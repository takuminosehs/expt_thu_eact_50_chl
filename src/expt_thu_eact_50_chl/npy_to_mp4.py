import os
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path

def npy_to_video(npy_path, output_mp4=None, H=260, W=346, fps=30):
    """
    イベントデータ(.npy)を読み込み、実時間スケールの動画(.mp4)に変換する
    """
    # パス文字列への安全な変換
    npy_path = str(npy_path)
    if not os.path.exists(npy_path):
        print(f"❌ エラー: ファイルが見つかりません -> {npy_path}")
        return

    # 出力ファイル名が指定されていない場合は自動生成
    if output_mp4 is None:
        output_mp4 = npy_path.replace(".npy", ".mp4")

    print(f"📂 読み込み中: {npy_path}")
    events = np.load(npy_path)
    
    if len(events) == 0:
        print("❌ イベントデータが空です。")
        return

    xs = events[:, 0].astype(np.int32)
    ys = events[:, 1].astype(np.int32)
    ts = events[:, 2]  # 単位：マイクロ秒(us)
    ps = events[:, 3]

    # 画面外の異常座標をカット
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys, ts, ps = xs[valid], ys[valid], ts[valid], ps[valid]

    t_min, t_max = ts.min(), ts.max()
    total_time_us = t_max - t_min
    
    # 30FPSで出力するための、1フレームあたりの時間幅（マイクロ秒）
    # 1秒 = 1,000,000マイクロ秒 / 30fps = 約33,333マイクロ秒
    frame_time_us = 1000000 / fps
    num_frames = int(total_time_us / frame_time_us) + 1

    print(f"🎬 動画の生成を開始します...")
    print(f"   - 記録時間: {total_time_us / 1e6:.2f} 秒")
    print(f"   - フレームレート: {fps} FPS")
    print(f"   - 総フレーム数: {num_frames} フレーム")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mp4, fourcc, fps, (W, H))

    for i in tqdm(range(num_frames), desc="レンダリング中"):
        t_start = t_min + i * frame_time_us
        t_end = t_start + frame_time_us

        # このフレームの時間幅に該当するイベントを抽出
        mask = (ts >= t_start) & (ts < t_end)
        bin_xs = xs[mask]
        bin_ys = ys[mask]
        bin_ps = ps[mask]

        # 背景を黒に設定
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # イベントを描画（OpenCVはBGR形式なので注意）
        # 正の極性(1) -> 赤(0, 0, 255) / 負の極性(-1等) -> 青(255, 0, 0)
        for x, y, p in zip(bin_xs, bin_ys, bin_ps):
            if p == 1:
                frame[y, x] = (0, 0, 255)  # 赤
            else:
                frame[y, x] = (255, 0, 0)  # 青

        out.write(frame)

    out.release()
    print(f"🎉 動画を保存しました: {output_mp4}")


if __name__ == "__main__":
    # --- 実行例 ---
    # ノイズ除去前の元データ
    npy_to_video("A0P8C0-2021_11_02_21_47_15.npy", "A0P8C0_raw.mp4")
    
    # ノイズ除去後のデータ（先ほど作成したもの）
    npy_to_video("A0P8C0-2021_11_02_21_47_15_hw_filtered.npy", "A0P8C0_filtered.mp4")