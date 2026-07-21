import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from expt_thu_eact_50_chl import config

# 📂 パス・ディレクトリ設定 (config.pyおよびPathlib完全準拠)
PROJECT_ROOT = config.PROJECT_ROOT
CURRENT_DIR = Path(__file__).parent.resolve()

def spatio_temporal_filter(
    xs, ys, ts_sec, polarities, H=260, W=346, dt_ms=10.0, min_neighbors=2
):
    """時空間相関フィルタ (STCF): 孤立した熱雑音イベントを特定・除去する"""
    t_min, t_max = ts_sec.min(), ts_sec.max()
    duration = t_max - t_min + 1e-6
    num_bins = int(np.ceil(duration / (dt_ms / 1000.0)))
    num_bins = max(3, num_bins)

    t_bins = ((ts_sec - t_min) / duration * (num_bins - 1)).astype(np.int32)

    voxel = np.zeros((num_bins, H, W), dtype=np.uint8)
    voxel[t_bins, ys, xs] = 1

    neighbor_counts = np.zeros_like(voxel, dtype=np.uint8)
    for b in range(num_bins):
        neighbor_counts[b] = cv2.boxFilter(
            voxel[b], -1, (3, 3), normalize=False
        )

    total_counts = neighbor_counts.copy()
    total_counts[1:] += neighbor_counts[:-1]
    total_counts[:-1] += neighbor_counts[1:]

    keep_mask = total_counts[t_bins, ys, xs] >= min_neighbors
    return keep_mask


def compute_time_surface(xs, ys, ts_sec, H=260, W=346):
    """適応的 tau (tau = T_total / 3.0) を用いた Time-Surface の生成"""
    if len(xs) == 0:
        return np.zeros((H, W), dtype=np.float32), np.zeros(
            (H, W), dtype=np.float32
        )

    t_min, t_max = ts_sec.min(), ts_sec.max()
    t_total = t_max - t_min if t_max > t_min else 1e-6
    tau_adaptive = max(t_total / 3.0, 1e-4)

    t_last = np.zeros((H, W), dtype=np.float32)
    has_event = np.zeros((H, W), dtype=bool)

    t_last[ys, xs] = ts_sec
    has_event[ys, xs] = True

    ts_decay = np.exp(-(t_max - t_last) / tau_adaptive)
    ts_decay[~has_event] = 0.0

    return ts_decay, t_last


def compute_normal_flow(t_last, has_event):
    """生タイムスタンプの空間勾配からノーマルフロー (速度ベクトル) を計算"""
    gx_raw = cv2.Sobel(t_last, cv2.CV_32F, 1, 0, ksize=3)
    gy_raw = cv2.Sobel(t_last, cv2.CV_32F, 0, 1, ksize=3)
    grad_sq = gx_raw**2 + gy_raw**2

    valid_grad = grad_sq > 1e-5
    flow_x = np.zeros_like(gx_raw)
    flow_y = np.zeros_like(gy_raw)

    flow_x[valid_grad] = gx_raw[valid_grad] / grad_sq[valid_grad]
    flow_y[valid_grad] = gy_raw[valid_grad] / grad_sq[valid_grad]

    flow_x = np.clip(flow_x, -10.0, 10.0)
    flow_y = np.clip(flow_y, -10.0, 10.0)

    flow_x[~has_event] = 0.0
    flow_y[~has_event] = 0.0

    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    return flow_x, flow_y, magnitude


def visualize_and_save(input_path: Path, output_dir: Path | None = None):
    """イベントデータのSTCFノイズ除去状態とノーマルフローを可視化して保存"""
    if not input_path.exists():
        raise FileNotFoundError(
            f"指定されたファイルが存在しません: {input_path}"
        )

    # 1. データ読み込み (shape: [N, 4] -> x, y, ts_us, p)
    events = np.load(input_path)
    xs = events[:, 0].astype(np.int32)
    ys = events[:, 1].astype(np.int32)
    ts_sec = events[:, 2] / 1e6
    ps = events[:, 3].astype(np.int32)
    polarities = np.where(ps == 1, 1.0, -1.0)

    H, W = 260, 346

    # 2. STCF フィルタリング適用
    keep_mask = spatio_temporal_filter(xs, ys, ts_sec, polarities, H=H, W=W)
    noise_mask = ~keep_mask

    # 3. 処理前・処理後・ノイズ成分のマップ生成
    ts_raw, _ = compute_time_surface(xs, ys, ts_sec, H=H, W=W)
    ts_filtered, t_last_filtered = compute_time_surface(
        xs[keep_mask], ys[keep_mask], ts_sec[keep_mask], H=H, W=W
    )
    ts_noise, _ = compute_time_surface(
        xs[noise_mask], ys[noise_mask], ts_sec[noise_mask], H=H, W=W
    )

    has_event_filtered = np.zeros((H, W), dtype=bool)
    has_event_filtered[ys[keep_mask], xs[keep_mask]] = True

    # 4. ノーマルフロー計算
    flow_x, flow_y, flow_mag = compute_normal_flow(
        t_last_filtered, has_event_filtered
    )

    # 5. Matplotlib による 4 パネル比較描画
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"STCF Denoising & Normal Flow Visualization\nFile: {input_path.name}",
        fontsize=14,
        fontweight="bold",
    )

    # Panel 1: 生イベント (STCF 処理前 Time-Surface)
    im1 = axes[0, 0].imshow(ts_raw, cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title(
        f"1. Raw Events Time-Surface\n(Total Events: {len(xs)})"
    )
    axes[0, 0].axis("off")
    fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Panel 2: STCF 処理後 Time-Surface
    im2 = axes[0, 1].imshow(ts_filtered, cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title(
        f"2. STCF Filtered Time-Surface\n(Kept Events: {np.sum(keep_mask)})"
    )
    axes[0, 1].axis("off")
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Panel 3: 除去されたノイズ (STCF 差分)
    im3 = axes[1, 0].imshow(ts_noise, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title(
        f"3. Removed Noise Events\n(Removed Events: {np.sum(noise_mask)})"
    )
    axes[1, 0].axis("off")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Panel 4: ノーマルフロー強度マップ＋ベクトル場 (Quiver overlay)
    im4 = axes[1, 1].imshow(flow_mag, cmap="viridis")
    axes[1, 1].set_title("4. Normal Flow Magnitude & Vectors")
    axes[1, 1].axis("off")
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 視認性を保つためベクトルを一定間隔(12px)で間引いてQuiver描画
    step = 12
    y_grid, x_grid = np.mgrid[0:H:step, 0:W:step]
    u = flow_x[::step, ::step]
    v = flow_y[::step, ::step]
    axes[1, 1].quiver(
        x_grid,
        y_grid,
        u,
        v,
        color="cyan",
        pivot="middle",
        scale=30.0,
        headwidth=3,
    )

    plt.tight_layout()

    # 6. 画像の保存処理
    if output_dir is None:
        save_dir = CURRENT_DIR
    else:
        save_dir = Path(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    output_filename = save_dir / f"{input_path.stem}_visualization.png"
    plt.savefig(output_filename, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"✅ 可視化画像を正常に保存しました: {output_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="STCFノイズ除去とノーマルフローの可視化スクリプト"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="可視化対象の生イベント `.npy` ファイルパス (shape: [N, 4])",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="保存先ディレクトリパス (未指定の場合は入力ファイルと同じ親ディレクトリに保存)",
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    visualize_and_save(input_path, output_dir)


if __name__ == "__main__":
    main()