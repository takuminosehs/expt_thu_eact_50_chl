import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path

# config.py からパスをインポート
from expt_thu_eact_50_chl.config import ORIGINAL_DATA_DIR

CURRENT_DIR = Path(__file__).parent.resolve()
PROCESSED_DIR = CURRENT_DIR / "processed_data"


def augment_events(events, H=260, W=346):
    """生イベントデータ [N, 4] (x, y, t, p) に対するデータ拡張（パターンB）

    ランダムにFlip, Zoom, Dropを組み合わせて適用します。
    """
    xs = events[:, 0].copy()
    ys = events[:, 1].copy()
    ts = events[:, 2].copy()
    ps = events[:, 3].copy()

    # 1. Horizontal Flip (確率50%)
    if np.random.rand() > 0.5:
        xs = (W - 1) - xs
        # 左右反転しても極性(p)はそのまま維持（幾何的な反転のみ）

    # 2. Event Zoom (確率50%)
    if np.random.rand() > 0.5:
        # 0.8倍〜1.2倍の範囲でランダムズーム
        scale = np.random.uniform(0.8, 1.2)
        cx, cy = W / 2.0, H / 2.0
        xs = cx + (xs - cx) * scale
        ys = cy + (ys - cy) * scale

    # 3. Event Drop / Random Mask (確率50%)
    if np.random.rand() > 0.5:
        # 10%〜30%のイベントをランダムに間引く
        drop_ratio = np.random.uniform(0.1, 0.3)
        n_events = len(events)
        if n_events > 0:
            keep_count = int(n_events * (1.0 - drop_ratio))
            keep_idx = np.random.choice(n_events, keep_count, replace=False)
            keep_idx.sort()  # 時系列順を維持するためにソート

            xs, ys, ts, ps = (
                xs[keep_idx],
                ys[keep_idx],
                ts[keep_idx],
                ps[keep_idx],
            )

    # ズーム等で画面外に飛び出したイベントをフィルタリング
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)

    return np.stack([xs[valid], ys[valid], ts[valid], ps[valid]], axis=1)


def events_to_robust_chsr(events, T_bins=224, H=260, W=346):
    """割り算を排除し、発散ノイズを防いだ堅牢な4チャネルCHSR"""
    v_tensor = np.zeros((4, T_bins, H), dtype=np.float32)
    if len(events) == 0:
        return v_tensor

    xs = events[:, 0].astype(np.int32)
    ys = events[:, 1].astype(np.int32)
    ts = events[:, 2]
    ps = events[:, 3].astype(np.int32)

    polarities = np.where(ps == 1, 1.0, -1.0)
    t_min, t_max = ts.min(), ts.max()
    t_total = t_max - t_min if t_max > t_min else 1e-6

    t_indices = ((ts - t_min) / t_total * T_bins).astype(np.int32)
    t_indices = np.clip(t_indices, 0, T_bins - 1)

    # 有効バウンダリチェック（念のため）
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys, ts, polarities, t_indices = (
        xs[valid],
        ys[valid],
        ts[valid],
        polarities[valid],
        t_indices[valid],
    )

    time_surface = np.zeros((H, W), dtype=np.float32)
    phi_x = np.sin(np.pi * np.arange(W) / W)

    for b in range(T_bins):
        bin_mask = t_indices == b
        bin_xs = xs[bin_mask]
        bin_ys = ys[bin_mask]
        bin_ts = ts[bin_mask]
        bin_pols = polarities[bin_mask]

        if len(bin_xs) == 0:
            continue

        norm_ts = (bin_ts - t_min) / t_total
        time_surface[bin_ys, bin_xs] = norm_ts

        # チャネル0: 極性マップ
        np.add.at(v_tensor[0, b, :], bin_ys, bin_pols)

        # チャネル3: ホログラフィックマップ
        np.add.at(v_tensor[3, b, :], bin_ys, phi_x[bin_xs])

        # チャネル1 & 2: 生勾配の蓄積
        gx = cv2.Sobel(time_surface, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(time_surface, cv2.CV_32F, 0, 1, ksize=3)

        np.add.at(v_tensor[1, b, :], bin_ys, gy[bin_ys, bin_xs])
        np.add.at(v_tensor[2, b, :], bin_ys, gx[bin_ys, bin_xs])

    # 標準化
    mean = np.mean(v_tensor)
    std = np.std(v_tensor) + 1e-5
    v_tensor = (v_tensor - mean) / std

    return v_tensor


def preprocess_and_save_augmented():
    print(
        f"🔍 大元データフォルダ（{ORIGINAL_DATA_DIR}）内の全データファイルをスキャン中..."
    )
    all_npy_paths = glob.glob(
        os.path.join(ORIGINAL_DATA_DIR, "**", "*.npy"), recursive=True
    )
    file_map = {os.path.basename(p): p for p in all_npy_paths}
    print(f"識別完了: {len(file_map)} 個のデータファイルを見つけました。")

    for mode in ["train", "test"]:
        txt_file = ORIGINAL_DATA_DIR / f"{mode}.txt"
        output_dir = PROCESSED_DIR / mode
        output_dir.mkdir(parents=True, exist_ok=True)

        if not txt_file.exists():
            print(f"❌ {txt_file} が見つかりません。")
            return

        with open(txt_file, "r") as f:
            lines = f.readlines()

        print(
            f"\n--- データ拡張版 【{mode.upper()}】 データのCHSR前処理を開始します ---"
        )

        for idx, line in enumerate(
            tqdm(lines, desc=f"Processing {mode}（Augmented Pipeline）")
        ):
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            filename_only = os.path.basename(parts[0])
            if filename_only not in file_map:
                continue
            filename_noext = os.path.splitext(filename_only)[0]

            # 1. 生イベントのロード
            events = np.load(file_map[filename_only])

            # --- オリジナルデータの前処理と保存 ---
            chsr_orig = events_to_robust_chsr(events, T_bins=224, H=260, W=346)
            save_name_orig = (
                f"{idx}_{filename_noext}_orig_label_{parts[1]}.npy"
            )
            np.save(output_dir / save_name_orig, chsr_orig)

            # --- 訓練データのみデータ拡張を施して増量保存 ---
            if mode == "train":
                events_aug = augment_events(events, H=260, W=346)
                chsr_aug = events_to_robust_chsr(
                    events_aug, T_bins=224, H=260, W=346
                )
                save_name_aug = (
                    f"{idx}_{filename_noext}_aug_label_{parts[1]}.npy"
                )
                np.save(output_dir / save_name_aug, chsr_aug)


if __name__ == "__main__":
    preprocess_and_save_augmented()
    print(
        f"\n🎉 データ拡張を適用した前処理が完了し、'{PROCESSED_DIR}' に保存されました！"
    )