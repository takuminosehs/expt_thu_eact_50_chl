# src/expt_thu_eact_50_chl/generate_retry_videos.py
import json
import sys
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path

# プロジェクトルートのパス解決とインポート環境の整備
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 制約：config.py を必ずインポートする
from expt_thu_eact_50_chl import config

# ====================================================
# 実験設定 (指定されたパス・構成に完全準拠)
# ====================================================
DATA_SOURCE_DIR = config.ORIGINAL_DATA_DIR
JSON_PATH = PROJECT_ROOT / "experiments" / "260708_1" / "result" / "label_check" / "id_to_action_template.json"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "260708_1" / "result" / "label_check_retry_2"

NUM_RETRY_VIDEOS = 15  # 追加する本数
H_MAX = 260
W_MAX = 346
FPS = 30


def render_debug_video(npy_path: Path, output_mp4: Path, label_num: int) -> None:
    """
    イベントデータを読み込み、ラベル番号を特大サイズで焼き付けたデバッグ動画を生成する
    """
    events = np.load(npy_path)
    if len(events) == 0:
        return

    xs = events[:, 0].astype(np.int32)
    ys = events[:, 1].astype(np.int32)
    ts = events[:, 2]
    ps = events[:, 3]

    valid = (xs >= 0) & (xs < W_MAX) & (ys >= 0) & (ys < H_MAX)
    xs, ys, ts, ps = xs[valid], ys[valid], ts[valid], ps[valid]

    t_min, t_max = ts.min(), ts.max()
    total_time_us = t_max - t_min
    
    frame_time_us = 1000000 / FPS
    num_frames = int(total_time_us / frame_time_us) + 1

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_mp4), fourcc, FPS, (W_MAX, H_MAX))

    for i in range(num_frames):
        t_start = t_min + i * frame_time_us
        t_end = t_start + frame_time_us

        mask = (ts >= t_start) & (ts < t_end)
        bin_xs = xs[mask]
        bin_ys = ys[mask]
        bin_ps = ps[mask]

        frame = np.zeros((H_MAX, W_MAX, 3), dtype=np.uint8)

        for x, y, p in zip(bin_xs, bin_ys, bin_ps):
            if p == 1:
                frame[y, x] = (0, 0, 255)  # 赤
            else:
                frame[y, x] = (255, 0, 0)  # 青

        # 画面左上に「LABEL: XX」を特大サイズ（識別しやすいように今回はシアン色）で焼き付け
        text = f"LABEL: {label_num}"
        cv2.putText(
            frame, text, (15, 45), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3, cv2.LINE_AA
        )

        out.write(frame)

    out.release()


def main():
    # 1. 現在の進行状況（JSON）の読み込み
    if not JSON_PATH.exists():
        print(f"❌ エラー: 編集中のJSONファイルが見つかりません -> {JSON_PATH}")
        print("前回のスクリプトを実行して雛形が生成されているか、パスが正しいか確認してください。")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        current_mapping = json.load(f)

    # 値が空文字列 "" のラベルID（未判別クラス）を抽出
    unidentified_labels = [int(k) for k, v in current_mapping.items() if v.strip() == ""]
    
    print(f"🔍 編集中のJSONを解析しました。")
    print(f"   - 判別済み: {50 - len(unidentified_labels)} クラス")
    print(f"   - 未判別 (ターゲット): {len(unidentified_labels)} クラス")

    if not unidentified_labels:
        print("🎉 すべてのラベルの穴埋めが完了しています！追加動画の生成は不要です。")
        return

    # 2. 大元データフォルダのスキャン
    print(f"\n📂 大元データフォルダ ({DATA_SOURCE_DIR}) をスキャン中...")
    all_npy_paths = list(DATA_SOURCE_DIR.rglob("*.npy"))
    file_map = {p.name: p for p in all_npy_paths}

    txt_file = DATA_SOURCE_DIR / "train.txt"
    if not txt_file.exists():
        txt_file = DATA_SOURCE_DIR / "test.txt"
        
    with open(txt_file, "r") as f:
        lines = f.readlines()

    # ラベルごとのファイルリストを整理
    label_to_files = {i: [] for i in range(50)}
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        filename_only = Path(parts[0]).name
        label_id = int(parts[1])
        if filename_only in file_map and 0 <= label_id < 50:
            label_to_files[label_id].append(file_map[filename_only])

    # 出力先ディレクトリの作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. 未判別ラベルに対して追加動画タスクを構築
    video_tasks = []
    for label_id in unidentified_labels:
        paths = label_to_files[label_id]
        if not paths:
            continue
        
        # 🌟 前回の1本目（インデックス0）を避けるため、インデックス1から5本のデータを抽出
        retry_paths = paths[1 : 1 + NUM_RETRY_VIDEOS]
        
        for idx, p in enumerate(retry_paths):
            # 何本目の追加動画かわかりやすいようにファイル名を設定
            out_name = f"retry_label_{label_id:02d}_sample{idx+1:02d}_{p.stem}.mp4"
            video_tasks.append((p, OUTPUT_DIR / out_name, label_id))

    # 4. レンダリングの実行
    if not video_tasks:
        print("⚠️ 対象となる追加データが見つかりませんでした。")
        return

    print(f"\n🎬 未判別クラスの追加レンダリングを開始します（総動画本数: {len(video_tasks)} 本）...")
    for p, out_path, label_id in tqdm(video_tasks, desc="Rendering retry videos"):
        render_debug_video(p, out_path, label_id)

    print(f"\n🎉 補填動画の生成が完了しました！")
    print(f"📹 追加確認用動画の保存先 ➡️ {OUTPUT_DIR.resolve()}")
    print(f"💡 各フォルダの映像を頼りに、既存の '{JSON_PATH.name}' を最後まで完成させてください！")


if __name__ == "__main__":
    main()