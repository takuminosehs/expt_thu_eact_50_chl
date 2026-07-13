import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# config.pyの内容は、必要に応じてインポートして拡張可能
from expt_thu_eact_50_chl.config import ORIGINAL_DATA_DIR, HW_DENOISED_NOPSI_DATA_DIR

# パス定義（制約に基づき Pathlib を徹底）
CURRENT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = CURRENT_DIR / "edge_visualize"


def create_edge_animation(npy_path: Path, T_bins=224, H=260, W=346, fps=30):
    """
    指定された生イベントデータから、ハリスコーナー検出を重ね合わせたmp4動画を生成する
    """
    if not npy_path.exists():
        print(f"❌ 指定されたファイルが見つかりません: {npy_path}")
        sys.exit(1)

    print(f"📦 データを読み込み中: {npy_path.name}")
    events = np.load(str(npy_path))

    if len(events) == 0:
        print(f"⚠️ イベントデータが空です。処理を中断します。")
        return

    # イベントから座標、時間情報を抽出
    xs = events[:, 0].astype(np.int32)
    ys = events[:, 1].astype(np.int32)
    ts = events[:, 2]

    t_min, t_max = ts.min(), ts.max()
    t_total = t_max - t_min if t_max > t_min else 1e-6

    # 224の時間ビンに厳密にインデックス化
    t_indices = ((ts - t_min) / t_total * T_bins).astype(np.int32)
    t_indices = np.clip(t_indices, 0, T_bins - 1)

    # 画面サイズ内の有効なイベントのみ抽出
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys, ts, t_indices = xs[valid], ys[valid], ts[valid], t_indices[valid]

    # 出力先ディレクトリの作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 指示通りの命名規則（もともとの名前の後に _edge_visualize を付与）
    filename_noext = npy_path.stem
    output_video_path = OUTPUT_DIR / f"{filename_noext}_edge_visualize.mp4"

    # OpenCV VideoWriter の初期化（2026年現在mp4コンテナで最も汎用性の高い mp4v を採用）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        str(output_video_path), fourcc, fps, (W, H)
    )

    time_surface = np.zeros((H, W), dtype=np.float32)

    print(f"🎬 幾何特徴の重ね書き動画を生成中（全 {T_bins} フレーム）...")
    
    # 時間ビンを時系列順にループ処理
    for b in tqdm(range(T_bins), desc="Generating Video"):
        bin_mask = (t_indices == b)
        bin_xs = xs[bin_mask]
        bin_ys = ys[bin_mask]
        bin_ts = ts[bin_mask]

        if len(bin_xs) > 0:
            # 前処理コードと完全に同一のロジックでTime Surfaceを更新
            norm_ts = (bin_ts - t_min) / t_total
            time_surface[bin_ys, bin_xs] = norm_ts

        # --- 可視化フレームのレンダリングプロセス ---
        
        # 1. 元データ（Time Surface）を背景画像[0-255]のBGRに変換してベース（下地）にする
        ts_visual = np.zeros((H, W), dtype=np.float32)
        cv2.normalize(time_surface, ts_visual, 0, 255, cv2.NORM_MINMAX)
        frame_img = cv2.cvtColor(ts_visual.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # 2. 前処理コードと全く同一条件でハリスコーナー強度を算出
        harris_map = cv2.cornerHarris(time_surface, blockSize=3, ksize=3, k=0.04)
        harris_map = np.clip(harris_map, 0, None)

        # --- 変更前 ---
        # if harris_map.max() > 0:
        #     corner_mask = harris_map > (0.01 * harris_map.max())
        #     frame_img[corner_mask] = [0, 0, 255]

        # --- 変更後（絶対閾値と厳格な相対閾値のハイブリッド） ---
        # 1e-5（0.00001）以下の微小な応答は、どれだけ周囲より高くてもノイズとみなして完全カット
        ABS_THRESHOLD = 1e-5  
        
        if harris_map.max() > ABS_THRESHOLD:
            # 最大値の 10%（0.10）以上、かつ絶対閾値を超える強い「角」だけを抽出
            corner_mask = (harris_map > (0.10 * harris_map.max())) & (harris_map > ABS_THRESHOLD)
            
            # 純粋なエッジ位置の1ピクセルだけを「赤色」に染める（パターン1を適用）
            frame_img[corner_mask] = [0, 0, 255]

        # ビデオライターにフレームを書き込み
        video_writer.write(frame_img)

    # リソース解放
    video_writer.release()
    print(f"\n🎉 動画生成が完了しました！")
    print(f"🎞️ 保存先: {output_video_path.resolve()}")


if __name__ == "__main__":
    # 使用方法1: コマンドラインから引数としてnpyのパスを直接指定して実行
    # 例: uv run animate_edge_visualization.py --path /path/to/data.npy
    parser = argparse.ArgumentParser(
        description="生イベントデータからハリスコーナー重ね合わせ動画(.mp4)を生成するスクリプト"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="可視化したい特定の生イベントファイル(.npy)へのパス",
    )
    args = parser.parse_args()

    if args.path:
        target_npy = Path(args.path)
    else:
        # 使用方法2: スクリプトを直接実行したい場合、以下のクォーテーション内を検証したい特定のパスに書き換えてください。
        # ─── スクリプト内直接指定用設定 ───
        SPECIFIC_NPY_PATH = (
            ORIGINAL_DATA_DIR / "A12P25C1-2021_11_08_11_12_14.npy"
            # HW_DENOISED_NOPSI_DATA_DIR / "A24P23C0-2021_11_08_09_19_40_hw_filtered.npy"
        )  # 例として設定しています。実際のファイル名に変更してください。
        # ─────────────────────────────────
        target_npy = SPECIFIC_NPY_PATH

    create_edge_animation(target_npy, T_bins=224, H=260, W=346, fps=30)