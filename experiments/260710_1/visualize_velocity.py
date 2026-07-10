import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def visualize_event_velocity(
    input_npy: Path,
    output_mp4: Path,
    bg_mode: str = "time_surface",
    scaling_mode: str = "log",
    scale: float = 15.0,
    fps: float = 30.0,
    T_bins: int = 224,
    H: int = 260,
    W: int = 346,
):
    """非線形スケーリングを用いて、大小の矢印の視認性を極限まで高めた速度ベクトル可視化"""
    print(f"📦 データを読み込み中: {input_npy}")
    events = np.load(input_npy)

    if len(events) == 0:
        print("❌ イベントデータが空です。")
        return

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

    # ビデオライターの初期化
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_mp4), fourcc, fps, (W, H))

    time_surface = np.zeros((H, W), dtype=np.float32)

    print(f"🎬 動画生成を開始します（背景: {bg_mode}, スケールモード: {scaling_mode}）")

    for b in tqdm(range(T_bins), desc="Generating Frames"):
        bin_mask = t_indices == b
        bin_xs = xs[bin_mask]
        bin_ys = ys[bin_mask]
        bin_ts = ts[bin_mask]
        bin_pols = polarities[bin_mask]

        if len(bin_xs) > 0:
            norm_ts = (bin_ts - t_min) / t_total
            time_surface[bin_ys, bin_xs] = norm_ts

        # 生勾配の計算
        gx = cv2.Sobel(time_surface, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(time_surface, cv2.CV_32F, 0, 1, ksize=3)

        # 背景画像の生成
        if bg_mode == "time_surface":
            bg_gray = (time_surface * 255).astype(np.uint8)
            frame = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2BGR)
        else:
            frame = np.zeros((H, W, 3), dtype=np.uint8)
            if len(bin_xs) > 0:
                for x, y, pol in zip(bin_xs, bin_ys, bin_pols):
                    frame[y, x] = [0, 0, 255] if pol > 0 else [255, 0, 0]

        # 速度ベクトルの描画
        if len(bin_xs) > 0:
            for x, y in zip(bin_xs, bin_ys):
                raw_dx = gx[y, x]
                raw_dy = gy[y, x]
                magnitude = np.sqrt(raw_dx**2 + raw_dy**2)

                if magnitude > 1e-5:
                    # --- 🌟 非線形ダイナミックレンジ圧縮の核心部分 🌟 ---
                    if scaling_mode == "log":
                        # 1. 対数スケーリング（最もおすすめ）
                        # 小さい値（高速移動）を底上げし、大きい値（低速・ノイズ）を強く抑制します
                        new_magnitude = np.log1p(magnitude * 10) * scale
                    elif scaling_mode == "sqrt":
                        # 2. 平方根スケーリング
                        # マイルドに大小の格差を縮めます
                        new_magnitude = np.sqrt(magnitude) * scale
                    elif scaling_mode == "fixed":
                        # 3. 固定長スケーリング
                        # 大小関係を完全に無視し、すべての矢印を「同じ長さ（scale）」にして方向だけを見せます
                        new_magnitude = scale
                    else:
                        new_magnitude = magnitude * scale

                    # 方向ベクトルを維持したまま、新しい長さに再計算
                    dx = (raw_dx / magnitude) * new_magnitude
                    dy = (raw_dy / magnitude) * new_magnitude

                    start_pt = (int(x), int(y))
                    end_pt = (int(x + dx), int(y + dy))

                    if start_pt != end_pt:
                        # 本来の「生勾配の大きさ」に応じて矢印の色を変化させるとさらに見やすくなります
                        # 小さい（高速）= 緑（0, 255, 0） / 大きい（低速）= 黄（0, 255, 255）
                        color = (0, 255, 0) if magnitude < 0.5 else (0, 255, 255)
                        
                        cv2.arrowedLine(
                            frame,
                            start_pt,
                            end_pt,
                            color,
                            thickness=1,
                            tipLength=0.3,
                        )

        video_writer.write(frame)

    video_writer.release()
    print(f"🎉 動画の書き出しが完了しました: {output_mp4}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event Stream Velocity Vector Visualizer")
    parser.add_argument("--input", type=str, required=True, help="対象とする入力 .npy ファイルのパス")
    parser.add_argument("--output", type=str, required=True, help="出力する .mp4 ファイルのパス")
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
        help="スケーリング方法: log(対数), sqrt(平方根), fixed(一律固定長), linear(前回の線形)",
    )
    parser.add_argument("--scale", type=float, default=15.0, help="矢印のベース長さ（fixedの時はそのまま矢印のピクセル長になります）")

    args = parser.parse_args()

    visualize_event_velocity(
        input_npy=Path(args.input),
        output_mp4=Path(args.output),
        bg_mode=args.bg_mode,
        scaling_mode=args.scaling_mode,
        scale=args.scale,
    )