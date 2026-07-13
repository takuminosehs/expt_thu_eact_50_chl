import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
# config.py から大元のデータディレクトリのみをインポート
from expt_thu_eact_50_chl.config import ORIGINAL_DATA_DIR

data_dir = ORIGINAL_DATA_DIR
CURRENT_DIR = Path(__file__).parent.resolve()
# 指示通り、train.pyと同じディレクトリ内の processed_data を出力先に指定
PROCESSED_DIR = CURRENT_DIR / "processed_data"

def events_to_robust_chsr_v5(events, T_bins=224, H=260, W=346):
    """
    幾何学的コーナー検出マップを5チャンネル目に追加した、改良型5チャネルCHSR表現
    """
    # チャンネル数を 4 から 5 へ拡張
    v_tensor = np.zeros((5, T_bins, H), dtype=np.float32)
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
        
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys, ts, polarities, t_indices = xs[valid], ys[valid], ts[valid], polarities[valid], t_indices[valid]

    time_surface = np.zeros((H, W), dtype=np.float32)
    phi_x = np.sin(np.pi * np.arange(W) / W)
    
    for b in range(T_bins):
        bin_mask = (t_indices == b)
        bin_xs = xs[bin_mask]
        bin_ys = ys[bin_mask]
        bin_ts = ts[bin_mask]
        bin_pols = polarities[bin_mask]
        
        if len(bin_xs) == 0:
            continue

        # Time Surfaceの更新
        norm_ts = (bin_ts - t_min) / t_total
        time_surface[bin_ys, bin_xs] = norm_ts
        
        # --- チャネル0: 極性マップ (+1/-1) ---
        np.add.at(v_tensor[0, b, :], bin_ys, bin_pols)
        
        # --- チャネル3: 論文準拠のホログラフィックマップ (位相加算) ---
        np.add.at(v_tensor[3, b, :], bin_ys, phi_x[bin_xs])
            
        # --- 速度（勾配）の計算 ---
        gx = cv2.Sobel(time_surface, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(time_surface, cv2.CV_32F, 0, 1, ksize=3)
        
        # --- チャネル1 & 2: 生勾配の蓄積 ---
        np.add.at(v_tensor[1, b, :], bin_ys, gy[bin_ys, bin_xs])
        np.add.at(v_tensor[2, b, :], bin_ys, gx[bin_ys, bin_xs])
        
        # --- チャネル4【新設】: 幾何学的ハリス・コーナー応答マップ ---
        # 蓄積された time_surface から局所的な構造テンソルを基にコーナー強度を算出
        harris_map = cv2.cornerHarris(time_surface, blockSize=3, ksize=3, k=0.04)
        # 負の応答（エッジや平坦領域）をクリップし、純粋な角（ポジティブ成分）のみを抽出
        harris_map = np.clip(harris_map, 0, None)
        # イベントが発生した位置の幾何学的コーナー強度のみを、他のチャネルと整合性を保って蓄積
        np.add.at(v_tensor[4, b, :], bin_ys, harris_map[bin_ys, bin_xs])
        
    # テンソル全体を標準化（Mean 0, Std 1）
    mean = np.mean(v_tensor)
    std = np.std(v_tensor) + 1e-5
    v_tensor = (v_tensor - mean) / std

    return v_tensor


def augment_raw_events(events, H=260, W=346):
    """
    生データ段階での物理的データ拡張
    """
    aug_events = events.copy()
    
    # 1. 左右反転
    if np.random.rand() < 0.5:
        aug_events[:, 0] = (W - 1) - aug_events[:, 0]
        
    # 2. ズーム（画面の中心を基準に拡大縮小）
    if np.random.rand() < 0.5:
        zoom_factor = np.random.uniform(0.8, 1.2)
        cx, cy = W / 2.0, H / 2.0
        aug_events[:, 0] = (aug_events[:, 0] - cx) * zoom_factor + cx
        aug_events[:, 1] = (aug_events[:, 1] - cy) * zoom_factor + cy
        
    # 3. イベントのランダム間引き
    if np.random.rand() < 0.5:
        drop_rate = np.random.uniform(0.1, 0.3)
        num_events = len(aug_events)
        keep_samples = int(num_events * (1 - drop_rate))
        
        keep_indices = np.random.choice(num_events, keep_samples, replace=False)
        keep_indices.sort()
        aug_events = aug_events[keep_indices]
        
    return aug_events


def preprocess_and_save():
    print(f"🔍 大元データフォルダ（{data_dir}）内の全データファイルをスキャン中...")
    # Pathlibで安全にパスを結合して探索
    all_npy_paths = glob.glob(str(data_dir / "**" / "*.npy"), recursive=True)
    file_map = {os.path.basename(p): p for p in all_npy_paths}
    print(f"識別完了: {len(file_map)} 個のデータファイルを見つけました。")

    np.random.seed(42)
    NUM_AUGMENTATIONS = 15 

    for mode in ["train", "test"]:
        txt_file = data_dir / f"{mode}.txt"
        output_dir = PROCESSED_DIR / mode
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not txt_file.exists():
            print(f"❌ {txt_file} が見つかりません。")
            return
            
        with open(txt_file, "r") as f:
            lines = f.readlines()
            
        print(f"\n--- HoloEv-Net-V4.2 (Harris Corner Enhanced) 用 【{mode.upper()}】 前処理開始 ---")
        
        for idx, line in enumerate(tqdm(lines, desc=f"Processing {mode}")):
            parts = line.strip().split()
            if len(parts) < 2: continue
            
            filename_only = os.path.basename(parts[0])
            if filename_only not in file_map: continue
            filename_noext = os.path.splitext(filename_only)[0]
                
            events = np.load(file_map[filename_only])
            
            # ① オリジナルデータの5ch生成・保存
            chsr_matrix = events_to_robust_chsr_v5(events, T_bins=224, H=260, W=346)
            filename = f"{idx}_{filename_noext}_orig_label_{parts[1]}.npy"
            np.save(str(output_dir / filename), chsr_matrix)

            # # ② 訓練モード時のみ、拡張データからも5chを再計算して保存
            # if mode == "train":
            #     for aug_idx in range(NUM_AUGMENTATIONS):
            #         aug_events = augment_raw_events(events, H=260, W=346)
            #         aug_chsr_matrix = events_to_robust_chsr_v5(aug_events, T_bins=224, H=260, W=346)
            #         filename_aug = f"{idx}_{filename_noext}_aug_{aug_idx}_label_{parts[1]}.npy"
            #         np.save(str(output_dir / filename_aug), aug_chsr_matrix)

if __name__ == "__main__":
    preprocess_and_save()
    print(f"\n🎉 5チャネル前処理が完了し、データが '{PROCESSED_DIR}' に保存されました！")