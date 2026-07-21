import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# config.py から必要パスをインポート (Pathlib準拠)
from expt_thu_eact_50_chl.config import MODEL_DIR, VEC_KM_FLOW_DIR

# 📂 ディレクトリ設定
CURRENT_DIR = Path(__file__).parent.resolve()

# ----------------------------------------------------------------------
# 💡 C++/CUDA 拡張モジュール (VecKM_flow) のロード
# ----------------------------------------------------------------------
if str(VEC_KM_FLOW_DIR) not in sys.path:
    sys.path.append(str(VEC_KM_FLOW_DIR))

try:
    from VecKM_flow import SliceNormalFlowEstimator
except ImportError as e:
    raise ImportError(
        f"❌ VecKM_flow モジュールのインポートに失敗しました。パスを確認してください: {VEC_KM_FLOW_DIR}"
    ) from e


def spatio_temporal_filter(
    xs: np.ndarray,
    ys: np.ndarray,
    ts_sec: np.ndarray,
    polarities: np.ndarray,
    H: int = 260,
    W: int = 346,
    dt_ms: float = 10.0,
    min_neighbors: int = 2,
) -> np.ndarray:
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


def compute_time_surface(
    xs: np.ndarray, ys: np.ndarray, ts_sec: np.ndarray, H: int = 260, W: int = 346
) -> tuple[np.ndarray, np.ndarray]:
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


def compute_normal_flow_veckm(
    xs: np.ndarray,
    ys: np.ndarray,
    ts_sec: np.ndarray,
    estimator: SliceNormalFlowEstimator,
    H: int = 260,
    W: int = 346,
    dt_window: float = 0.030,
    t_radius: float = 0.012,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """VecKM_flow (C++/CUDA) を用いてノーマルフローマップと強度を算出"""
    flow_x_map = np.zeros((H, W), dtype=np.float32)
    flow_y_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    if len(xs) == 0:
        return flow_x_map, flow_y_map, np.zeros((H, W), dtype=np.float32)

    t_min, t_max = ts_sec.min(), ts_sec.max()
    t_total = t_max - t_min if t_max > t_min else 1e-6

    has_event = np.zeros((H, W), dtype=bool)
    has_event[ys, xs] = True

    num_windows = max(1, int(np.ceil(t_total / dt_window)))
    all_indices = np.arange(len(ts_sec))

    for w in range(num_windows):
        t_center = t_min + (w + 0.5) * dt_window

        context_mask = (ts_sec >= t_center - dt_window) & (
            ts_sec <= t_center + dt_window
        )
        num_ctx = np.sum(context_mask)

        if num_ctx < 10:
            continue

        if num_ctx > 450000:
            dists = np.abs(ts_sec[context_mask] - t_center)
            threshold_dist = np.partition(dists, 450000)[450000]
            context_mask = context_mask & (
                np.abs(ts_sec - t_center) <= threshold_dist
            )

        context_indices = all_indices[context_mask]
        context_events_txy = np.stack(
            [
                ts_sec[context_mask],
                xs[context_mask].astype(np.float32),
                ys[context_mask].astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        context_events_txy = np.ascontiguousarray(context_events_txy)

        target_mask = (ts_sec >= t_center - dt_window / 2) & (
            ts_sec <= t_center + dt_window / 2
        )
        target_global_indices = all_indices[target_mask]
        target_indices_in_ctx = np.where(
            np.isin(context_indices, target_global_indices)
        )[0].astype(np.int32)

        if len(target_indices_in_ctx) == 0:
            continue

        try:
            flow = estimator.predict_flows(
                context_events_txy,
                context_events_txy.shape[0],
                target_indices_in_ctx,
                target_indices_in_ctx.shape[0],
                t_center,
                t_radius,
            )

            global_target_ids = context_indices[target_indices_in_ctx]
            target_xs = xs[global_target_ids]
            target_ys = ys[global_target_ids]

            np.add.at(flow_x_map, (target_ys, target_xs), flow[:, 0])
            np.add.at(flow_y_map, (target_ys, target_xs), flow[:, 1])
            np.add.at(count_map, (target_ys, target_xs), 1.0)

        except Exception:
            pass

    valid_counts = count_map > 0
    flow_x_map[valid_counts] /= count_map[valid_counts]
    flow_y_map[valid_counts] /= count_map[valid_counts]

    flow_x_map = np.clip(flow_x_map, -10.0, 10.0)
    flow_y_map = np.clip(flow_y_map, -10.0, 10.0)

    flow_x_map[~has_event] = 0.0
    flow_y_map[~has_event] = 0.0

    flow_mag = np.sqrt(flow_x_map**2 + flow_y_map**2)
    return flow_x_map, flow_y_map, flow_mag


def visualize_and_save(input_path: Path, output_dir: Path | None = None) -> None:
    """イベントデータのSTCFノイズ除去状態とVecKMノーマルフローを可視化して保存"""
    if not input_path.exists():
        raise FileNotFoundError(
            f"指定されたファイルが存在しません: {input_path}"
        )

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"❌ 事前学習済みモデルフォルダが見つかりません: {MODEL_DIR}"
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

    # 3. 処理前・処理後・ノイズ成分の Time-Surface 生成
    ts_raw, _ = compute_time_surface(xs, ys, ts_sec, H=H, W=W)
    ts_filtered, _ = compute_time_surface(
        xs[keep_mask], ys[keep_mask], ts_sec[keep_mask], H=H, W=W
    )
    ts_noise, _ = compute_time_surface(
        xs[noise_mask], ys[noise_mask], ts_sec[noise_mask], H=H, W=W
    )

    # 4. VecKM_flow Estimator の初期化とノーマルフロー計算
    estimator = SliceNormalFlowEstimator(
        str(MODEL_DIR), 500000, 640, 480, 64, 8
    )
    flow_x, flow_y, flow_mag = compute_normal_flow_veckm(
        xs[keep_mask], ys[keep_mask], ts_sec[keep_mask], estimator, H=H, W=W
    )

    # 5. Matplotlib による 4 パネル比較描画
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"STCF Denoising & VecKM Normal Flow Visualization\nFile: {input_path.name}",
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
    axes[1, 1].set_title("4. VecKM Normal Flow Magnitude & Vectors")
    axes[1, 1].axis("off")
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 視認性を保つためベクトルを一定間隔(12px)で間引いて Quiver 描画
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="STCFノイズ除去とVecKMノーマルフローの可視化スクリプト"
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
        help="保存先ディレクトリパス (未指定の場合は実行ファイルと同じ場所に保存)",
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    visualize_and_save(input_path, output_dir)


if __name__ == "__main__":
    main()