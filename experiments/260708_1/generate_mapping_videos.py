# src/expt_thu_eact_50_chl/generate_mapping_videos.py
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
# 実験設定 (ユーザー指定のパス構造に完全準拠)
# ====================================================
DATA_SOURCE_DIR = config.ORIGINAL_DATA_DIR  # 大元データディレクトリ
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "260708_1" / "result" / "label_check"

# 特別に対象全本数を動画化するクラスの指定
FULL_CHECK_LABELS = {3, 36}

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

        # 🌟 目視確認しやすいよう、画面右上に「LABEL: XX」を特大サイズ（黄色）で焼き付け
        text = f"LABEL: {label_num}"
        cv2.putText(
            frame, text, (15, 45), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA
        )

        out.write(frame)

    out.release()


def main():
    print(f"📂 大元データフォルダ ({DATA_SOURCE_DIR}) をスキャン中...")
    all_npy_paths = list(DATA_SOURCE_DIR.rglob("*.npy"))
    file_map = {p.name: p for p in all_npy_paths}
    print(f"識別完了: {len(file_map)} 個の .npy ファイルを発見しました。")

    # train.txt を優先し、無ければ test.txt を探す
    txt_file = DATA_SOURCE_DIR / "train.txt"
    if not txt_file.exists():
        txt_file = DATA_SOURCE_DIR / "test.txt"
        
    if not txt_file.exists():
        print(f"❌ エラー: リストファイル（train.txt または test.txt）が {DATA_SOURCE_DIR} 内に見つかりません。")
        return

    print(f"📄 インデックスソースとして '{txt_file.name}' を読み込みます。")
    with open(txt_file, "r") as f:
        lines = f.readlines()

    # 各ラベルごとのファイルリストを整理
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

    # 1. 動画化タスクの構築
    video_tasks = []
    for label_id, paths in label_to_files.items():
        if not paths:
            print(f"⚠️ 警告: ラベル {label_id} に該当するファイルがデータセット内にありません。")
            continue
            
        if label_id in FULL_CHECK_LABELS:
            # 🌟 指定されたラベル (3, 36) は該当する全データを動画化対象にする
            print(f"📌 ラベル {label_id} は全件検証対象に設定されました（計 {len(paths)} 本）")
            for idx, p in enumerate(paths):
                out_name = f"full_check_label_{label_id:02d}_idx{idx+1:03d}_{p.stem}.mp4"
                video_tasks.append((p, OUTPUT_DIR / out_name, label_id))
        else:
            # それ以外のラベルは最初の1本のみサンプリング
            p = paths[0]
            out_name = f"sample_label_{label_id:02d}_{p.stem}.mp4"
            video_tasks.append((p, OUTPUT_DIR / out_name, label_id))

    # 2. 動画のレンダリング実行
    print(f"\n🎬 レンダリングを開始します（総動画本数: {len(video_tasks)} 本）...")
    for p, out_path, label_id in tqdm(video_tasks, desc="Rendering debug videos"):
        render_debug_video(p, out_path, label_id)

    # 3. 確認用マッピング雛形（JSON）の同時出力
    json_template = {str(i): "" for i in range(50)}
    json_out_path = OUTPUT_DIR / "id_to_action_template.json"
    with open(json_out_path, "w", encoding="utf-8") as jf:
        json.dump(json_template, jf, ensure_ascii=False, indent=4)

    print(f"\n🎉 すべての処理が完了しました！")
    print(f"📹 動画の保存先 ➡️ {OUTPUT_DIR.resolve()}")
    print(f"📝 雛形JSONパス ➡️ {json_out_path.resolve()}")


if __name__ == "__main__":
    main()