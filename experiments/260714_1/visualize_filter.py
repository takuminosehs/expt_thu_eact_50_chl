#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

# プロジェクトルートをPathに追加してconfigをインポート可能にする
# (今回のスクリプトはスタンドアロンでも動作しますが、今後の拡張性を考慮)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from config import ORIGINAL_DATA_DIR
except ImportError:
    ORIGINAL_DATA_DIR = PROJECT_ROOT / "data" / "THU-EACT-50-CHL"


def visualize_density_filter(
    event_path: Path,
    output_path: Path,
    t_bins_g: int = 224,
    b_bins_l: int = 4,
    h: int = 260,
    w: int = 346,
    padding: int = 10,
) -> None:
    """イベントデータから密度クラスターの抽出過程を可視化してPNG保存する"""
    # 1. データの読み込みと前処理
    events = np.load(event_path)
    if len(events) == 0:
        print(f"[Warning] {event_path.name} は空のデータです。")
        return

    xs = events[:, 0].astype(np.int32)
    ys = events[:, 1].astype(np.int32)
    ts = events[:, 2]
    ps = events[:, 3].astype(np.int32)

    polarities = np.where(ps == 1, 1.0, -1.0)
    t_min, t_max = ts.min(), ts.max()
    t_total = t_max - t_min if t_max > t_min else 1e-6

    # 画面外のインデックスを除外
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys, ts, polarities = xs[valid], ys[valid], ts[valid], polarities[valid]

    # ローカル側の時間ビンインデックスを計算
    t_indices_l = ((ts - t_min) / t_total * b_bins_l).astype(np.int32)
    t_indices_l = np.clip(t_indices_l, 0, b_bins_l - 1)

    # 可視化用のMatplotlibフィギュアを用意 (横2列 × 縦 B_bins_l 行)
    fig, axes = plt.subplots(
        b_bins_l, 2, figsize=(14, 3.5 * b_bins_l), sharex=True, sharey=True
    )
    if b_bins_l == 1:
        axes = np.expand_dims(axes, axis=0)

    print(f"Processing visualization for: {event_path.name}")

    # 2. 各時間ビンごとのフィルタリング挙動をシミュレート＆描画
    for b in range(b_bins_l):
        mask_l = t_indices_l == b

        # キャンバス生成 (背景ホワイト: BGR)
        img_all = np.ones((h, w, 3), dtype=np.uint8) * 255
        img_filtered = np.ones((h, w, 3), dtype=np.uint8) * 255

        if not np.any(mask_l):
            # イベントがない場合は白紙のまま
            axes[b, 0].imshow(img_all)
            axes[b, 0].set_title(f"Bin {b}: No Events")
            axes[b, 1].imshow(img_filtered)
            axes[b, 1].set_title(f"Bin {b}: Empty")
            continue

        b_xs, b_ys, b_pols = xs[mask_l], ys[mask_l], polarities[mask_l]

        # 高速マッピング (Positive=赤[0,0,255], Negative=青[255,0,0])
        pos_m = b_pols > 0
        neg_m = b_pols < 0
        img_all[b_ys[pos_m], b_xs[pos_m]] = [0, 0, 255]
        img_all[b_ys[neg_m], b_xs[neg_m]] = [255, 0, 0]

        # 密度クラスター抽出ロジックの再現
        bin_binary = np.zeros((h, w), dtype=np.uint8)
        bin_binary[b_ys, b_xs] = 255

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            bin_binary, connectivity=8
        )
        spatial_mask = np.zeros((h, w), dtype=np.float32)

        # 抽出されたクラスタの境界ボックスを描画
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 8:
                continue
            cx = stats[i, cv2.CC_STAT_LEFT]
            cy = stats[i, cv2.CC_STAT_TOP]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]

            x1, y1 = max(0, cx - padding), max(0, cy - padding)
            x2, y2 = min(w, cx + cw + padding), min(h, cy + ch + padding)
            spatial_mask[y1:y2, x1:x2] = 1.0

            # 元データ側に抽出判定枠（グリーン）を描画
            cv2.rectangle(img_all, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # マスク内の有効なイベントのみを右側にプロット
        valid_cluster_mask = spatial_mask[b_ys, b_xs] == 1.0
        if np.any(valid_cluster_mask):
            f_xs = b_xs[valid_cluster_mask]
            f_ys = b_ys[valid_cluster_mask]
            f_pols = b_pols[valid_cluster_mask]

            f_pos = f_pols > 0
            f_neg = f_pols < 0
            img_filtered[f_ys[f_pos], f_xs[f_pos]] = [0, 0, 255]
            img_filtered[f_ys[f_neg], f_xs[f_neg]] = [255, 0, 0]

        # OpenCV(BGR) から Matplotlib(RGB) へ変換して表示
        axes[b, 0].imshow(cv2.cvtColor(img_all, cv2.COLOR_BGR2RGB))
        axes[b, 0].set_title(
            f"Bin {b}: All Events & Bounding Boxes (Area >= 8)"
        )
        axes[b, 0].axis("off")

        axes[b, 1].imshow(cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB))
        axes[b, 1].set_title(f"Bin {b}: Filtered (Saved to Tensor)")
        axes[b, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Success] 可視化結果を保存しました: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="前処理における空間密度フィルタリングの可視化スクリプト"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="対象とする入力イベントデータ (.npy) のパス",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="出力するPNGのパスまたはディレクトリ (未指定時はスクリプトと同階層)",
    )
    args = parser.parse_args()

    # パスをPathlibオブジェクトに変換して絶対パス化
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"[Error] 入力ファイルが存在しません: {input_path}")
        sys.exit(1)

    # 出力先の決定ロジック
    script_dir = Path(__file__).parent.resolve()
    if args.output is None:
        # 未入力の場合はスクリプトと同じディレクトリにファイル名を生成
        output_path = script_dir / f"vis_{input_path.stem}.png"
    else:
        out_p = Path(args.output).resolve()
        # 拡張子がない、あるいは既存のディレクトリならディレクトリとみなす
        if out_p.suffix == "" or out_p.is_dir():
            out_p.mkdir(parents=True, exist_ok=True)
            output_path = out_p / f"vis_{input_path.stem}.png"
        else:
            out_p.parent.mkdir(parents=True, exist_ok=True)
            output_path = out_p

    visualize_density_filter(input_path, output_path)


if __name__ == "__main__":
    main()