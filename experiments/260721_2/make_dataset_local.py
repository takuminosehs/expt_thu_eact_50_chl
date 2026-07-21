import ctypes
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from expt_thu_eact_50_chl import config

# 📂 パス・ディレクトリ設定 (config.pyおよびPathlib完全準拠)
ORIGINAL_DATA_DIR: Path = config.ORIGINAL_DATA_DIR
MODEL_DIR: Path = config.MODEL_DIR
VEC_KM_FLOW_DIR: Path = config.VEC_KM_FLOW_DIR

CURRENT_DIR = Path(__file__).parent.resolve()
PROCESSED_DIR = CURRENT_DIR / "processed_data"

# ----------------------------------------------------------------------
# 💡 C++/CUDA 共有ライブラリ (.so) および C拡張モジュールの完全自動ロード
# ----------------------------------------------------------------------
if str(VEC_KM_FLOW_DIR) not in sys.path:
    sys.path.append(str(VEC_KM_FLOW_DIR))

so_library_path = VEC_KM_FLOW_DIR / "libSliceNormalFlowEstimator.so"
if so_library_path.exists():
    ctypes.CDLL(str(so_library_path), mode=ctypes.RTLD_GLOBAL)
else:
    raise FileNotFoundError(
        f"❌ C++共有ライブラリが見つかりません: {so_library_path}"
    )

from VecKM_flow import SliceNormalFlowEstimator  # noqa: E402


def spatio_temporal_filter(
    xs: np.ndarray,
    ys: np.ndarray,
    ts_sec: np.ndarray,
    polarities: np.ndarray,
    H: int = 260,
    W: int = 346,
    dt_ms: float = 10.0,
    min_neighbors: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """時空間相関フィルタ (STCF): 孤立した熱雑音イベントを除去する"""
    t_min, t_max = ts_sec.min(), ts_sec.max()
    duration = t_max - t_min + 1e-6
    num_bins = int(np.ceil(duration / (dt_ms / 1000.0)))
    num_bins = max(3, num_bins)

    t_bins = ((ts_sec - t_min) / duration * (num_bins - 1)).astype(np.int32)

    voxel = np.zeros((num_bins, H, W), dtype=np.uint8)
    voxel[t_bins, ys, xs] = 1

    neighbor_counts = np.zeros_like(voxel, dtype=np.uint8)
    for b in range(num_bins):
        neighbor_counts[b] = cv2.boxFilter(voxel[b], -1, (3, 3), normalize=False)

    total_counts = neighbor_counts.copy()
    total_counts[1:] += neighbor_counts[:-1]
    total_counts[:-1] += neighbor_counts[1:]

    keep_mask = total_counts[t_bins, ys, xs] >= min_neighbors
    return (
        xs[keep_mask],
        ys[keep_mask],
        ts_sec[keep_mask],
        polarities[keep_mask],
    )


def generate_adaptive_4ch_local(
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    polarities: np.ndarray,
    estimator: SliceNormalFlowEstimator,
    H: int = 260,
    W: int = 346,
) -> np.ndarray:
    """VecKM ノーマルフローを用いた全範囲適応型 4 チャネル特徴量表現を生成

    Ch1: 適応的 Time-Surface
    Ch2: TSの空間勾配強度 (Sobelエッジ)
    Ch3: VecKM 推定ノーマルフロー X成分
    Ch4: VecKM 推定ノーマルフロー Y成分
    """
    v_tensor_local = np.zeros((4, H, W), dtype=np.float32)

    if len(xs) == 0:
        return v_tensor_local

    # タイムスタンプをマイクロ秒から秒へ正規化
    ts_sec = ts / 1e6

    # 1. 時空間ノイズ除去 (STCF)
    xs, ys, ts_sec, polarities = spatio_temporal_filter(
        xs, ys, ts_sec, polarities, H, W
    )

    if len(xs) == 0:
        return v_tensor_local

    t_min, t_max = ts_sec.min(), ts_sec.max()
    t_total = t_max - t_min if t_max > t_min else 1e-6

    t_last = np.zeros((H, W), dtype=np.float32)
    has_event = np.zeros((H, W), dtype=bool)

    t_last[ys, xs] = ts_sec
    has_event[ys, xs] = True

    # ─── Ch1: 適応的 Time-Surface ───
    tau_adaptive = max(t_total / 3.0, 1e-4)
    ts_decay = np.exp(-(t_max - t_last) / tau_adaptive)
    ts_decay[~has_event] = 0.0
    v_tensor_local[0] = ts_decay

    # ─── Ch2: TSの空間勾配強度 (Sobelエッジ) ───
    gx = cv2.Sobel(ts_decay, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ts_decay, cv2.CV_32F, 0, 1, ksize=3)
    ts_edge = np.sqrt(gx**2 + gy**2)
    v_tensor_local[1] = ts_edge

    # ─── Ch3 & Ch4: VecKM による高精度ノーマルフロー予測 ───
    flow_x_map = np.zeros((H, W), dtype=np.float32)
    flow_y_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    dt_window = 0.030  # 30ms 時間窓
    t_radius = 0.012   # 12ms コンテキスト半径

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

    v_tensor_local[2] = flow_x_map
    v_tensor_local[3] = flow_y_map

    # ─── チャネル単位での標準化 (平均0, 標準偏差1) ───
    for c in range(4):
        std = np.std(v_tensor_local[c]) + 1e-5
        v_tensor_local[c] = (v_tensor_local[c] - np.mean(v_tensor_local[c])) / std

    return v_tensor_local


def main() -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"❌ 事前学習済みモデルフォルダが見つかりません: {MODEL_DIR}"
        )

    print(f"🚀 VecKM SliceNormalFlowEstimator を初期化中: {MODEL_DIR}")
    # モデル設定: 640x480_24ms_C64_k8 (max_pts=500000, W=640, H=480, C=64, K=8)
    estimator = SliceNormalFlowEstimator(
        str(MODEL_DIR), 500000, 640, 480, 64, 8
    )

    all_npy_paths = list(ORIGINAL_DATA_DIR.rglob("*.npy"))
    file_map = {p.name: p for p in all_npy_paths}
    print(f"スキャン完了: {len(file_map)} 個のデータファイル")

    for mode in ["train", "test"]:
        txt_file = ORIGINAL_DATA_DIR / f"{mode}.txt"
        if not txt_file.exists():
            continue

        l_out_dir = PROCESSED_DIR / mode / "local"
        l_out_dir.mkdir(parents=True, exist_ok=True)

        lines = txt_file.read_text(encoding="utf-8").strip().splitlines()

        print(f"--- Generating {mode.upper()} Splits (VecKM Adaptive 4ch) ---")
        for idx, line in enumerate(tqdm(lines)):
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            filename_only = Path(parts[0]).name
            if filename_only not in file_map:
                continue
            filename_noext = Path(filename_only).stem

            events = np.load(file_map[filename_only])
            xs = events[:, 0].astype(np.int32)
            ys = events[:, 1].astype(np.int32)
            ts = events[:, 2]
            ps = events[:, 3].astype(np.int32)

            polarities = np.where(ps == 1, 1.0, -1.0)

            feat_l = generate_adaptive_4ch_local(
                xs, ys, ts, polarities, estimator
            )

            out_name = f"{idx}_{filename_noext}_label_{parts[1]}.npy"
            np.save(l_out_dir / out_name, feat_l)

    print("✅ 前処理データの保存が正常に完了しました！")


if __name__ == "__main__":
    main()