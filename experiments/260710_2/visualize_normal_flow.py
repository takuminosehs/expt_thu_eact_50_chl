import argparse
import sys
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- 🌟 共有ライブラリ (.so) を正常にロードするためのパス設定 🌟 ---
CURRENT_DIR = Path(__file__).parent.resolve()
os.environ["LD_LIBRARY_PATH"] = str(CURRENT_DIR) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.append(str(CURRENT_DIR))

# ビルドしたC++/CUDA高速ノーマルフロー抽出器をインポート
from VecKM_flow import SliceNormalFlowEstimator


def visualize_normal_flow(
    input_npy: Path,
    output_mp4: Path,
    model_dir: Path,
    bg_mode: str = "time_surface",
    scaling_mode: str = "log",
    scale: float = 15.0,
    fps: float = 30.0,
    T_bins: int = 224,
    H: int = 260,
    W: int = 346,
):
    """大元イベントデータからVecKMノーマルフローを計算し、非線形ベクトル動画を生成する"""
    print(f"📦 データを読み込み中: {input_npy}")
    events = np.load(input_npy)

    if len(events) == 0:
        print("❌ イベントデータが空です。")
        return

    # タイムスタンプ順にソート
    events = events[np.argsort(events[:, 2])]

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
    xs, ys, ts, polarities, t_indices = (
        xs[valid],
        ys[valid],
        ts[valid],
        polarities[valid],
        t_indices[valid],
    )

    # 抽出器の初期化 (346x260, D=64, k=8)
    print("🚀 SliceNormalFlowEstimator を初期化しています...")
    estimator = SliceNormalFlowEstimator(str(model_dir), 500000, W, H, 64, 8)

    # ビデオライターの初期化
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_mp4), fourcc, fps, (W, H))

    time_surface = np.zeros((H, W), dtype=np.float32)
    all_indices = np.arange(len(ts))
    t_radius = 0.012  # 24msモデル窓用の半径 (12ms)

    print(
        f"🎬 ノーマルフロー動画生成を開始します（背景: {bg_mode}, スケールモード: {scaling_mode}）"
    )

    for b in tqdm(range(T_bins), desc="Generating Frames"):
        bin_mask = t_indices == b
        bin_xs = xs[bin_mask]
        bin_ys = ys[bin_mask]
        bin_ts_sub = ts[bin_mask]
        bin_pols = polarities[bin_mask]

        if len(bin_xs) > 0:
            norm_ts = (bin_ts_sub - t_min) / t_total
            time_surface[bin_ys, bin_xs] = norm_ts

        # 背景画像のベース作成
        if bg_mode == "time_surface":
            bg_gray = (time_surface * 255).astype(np.uint8)
            frame = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2BGR)
        else:
            frame = np.zeros((H, W, 3), dtype=np.uint8)
            if len(bin_xs) > 0:
                for x, y, pol in zip(bin_xs, bin_ys, bin_pols):
                    frame[y, x] = [0, 0, 255] if pol > 0 else [255, 0, 0]

        # --- ノーマルフローの計算（make_dataset.pyのロジックを完全再現） ---
        if len(bin_xs) > 0:
            t_center = (bin_ts_sub.min() + bin_ts_sub.max()) / 2.0
            context_mask = (ts >= t_center - 0.030) & (ts <= t_center + 0.030)

            if np.sum(context_mask) > 450000:
                dists = np.abs(ts[context_mask] - t_center)
                threshold_dist = np.partition(dists, 450000)[450000]
                context_mask = context_mask & (
                    np.abs(ts - t_center) <= threshold_dist
                )

            context_indices = all_indices[context_mask]
            context_events_txy = np.stack(
                [ts[context_mask], xs[context_mask], ys[context_mask]], axis=1
            ).astype(np.float32)
            context_events_txy = np.ascontiguousarray(context_events_txy)

            bin_global_indices = all_indices[bin_mask]
            target_indices = np.where(
                np.isin(context_indices, bin_global_indices)
            )[0].astype(np.int32)

            if len(target_indices) > 0:
                try:
                    # C++モデルから直接ノーマルフローを予測
                    flow = estimator.predict_flows(
                        context_events_txy,
                        context_events_txy.shape[0],
                        target_indices,
                        target_indices.shape[0],
                        t_center,
                        t_radius,
                    )

                    global_target_indices = context_indices[target_indices]
                    flow_xs = xs[global_target_indices]
                    flow_ys = ys[global_target_indices]

                    # 各ターゲットイベントに対してベクトルを描画
                    for x, y, f_x, f_y in zip(
                        flow_xs, flow_ys, flow[:, 0], flow[:, 1]
                    ):
                        magnitude = np.sqrt(f_x**2 + f_y**2)

                        if magnitude > 1e-5:
                            # 非線形スケーリングの適用
                            if scaling_mode == "log":
                                new_magnitude = (
                                    np.log1p(magnitude * 10) * scale
                                )
                            elif scaling_mode == "sqrt":
                                new_magnitude = np.sqrt(magnitude) * scale
                            elif scaling_mode == "fixed":
                                new_magnitude = scale
                            else:
                                new_magnitude = magnitude * scale

                            dx = (f_x / magnitude) * new_magnitude
                            dy = (f_y / magnitude) * new_magnitude

                            start_pt = (int(x), int(y))
                            end_pt = (int(x + dx), int(y + dy))

                            if start_pt != end_pt:
                                # 視認性を上げるため、フロー強度で色分け
                                # 小さい（高速・鮮明）= 緑 / 大きい（低速・不確定）= 黄
                                color = (
                                    (0, 255, 0)
                                    if magnitude < 1.0
                                    else (0, 255, 255)
                                )

                                cv2.arrowedLine(
                                    frame,
                                    start_pt,
                                    end_pt,
                                    color,
                                    thickness=1,
                                    tipLength=0.3,
                                )
                except Exception as e:
                    pass

        video_writer.write(frame)

    video_writer.release()
    print(f"🎉 動画の書き出しが完了しました: {output_mp4}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VecKM Normal Flow Vector Visualizer"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="対象とする入力 .npy ファイル"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="出力する .mp4 ファイル"
    )
    parser.add_argument(
        "--bg-mode",
        type=str,
        choices=["time_surface", "event_count"],
        default="time_surface",
    )
    parser.add_argument(
        "--scaling-mode",
        type=str,
        choices=["log", "sqrt", "fixed", "linear"],
        default="log",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=12.0,
        help="矢印の長さ調整（fixedモード時は固定ピクセル長）",
    )

    args = parser.parse_args()

    visualize_normal_flow(
        input_npy=Path(args.input),
        output_mp4=Path(args.output),
        model_dir=CURRENT_DIR / "640x480_24ms_C64_k8",
        bg_mode=args.bg_mode,
        scaling_mode=args.scaling_mode,
        scale=args.scale,
    )