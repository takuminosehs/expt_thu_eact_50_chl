# src/expt_thu_eact_50_chl/error_visualizer.py
import json
import sys
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import expt_thu_eact_50_chl.config as config

import re
# プロジェクトルートのパス解決とインポート環境の整備
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _extract_original_filename(processed_filename: str) -> str:
    """
    ファイル名から特定のパターン（A...-20...）にマッチする部分を抽出する
    
    例: "63_A45P22C2-2021_11_07_11_58_38_hw_filtered_ch4_label_13.npy" 
    -> "A45P22C2-2021_11_07_11_58_38"
    """
    # 条件：A{数字}P{数字}C{数字}-20{数字}_{数字}_{数字}_{数字}_{数字}_{数字}
    pattern = r"A\d+P\d+C\d+-20\d+_\d+_\d+_\d+_\d+_\d+"
    # 最初のインデックス番号（例: "63_"）を剥ぎ取る
    match = re.search(pattern, processed_filename)
    if not match:
        raise ValueError(
            f"ファイル名のパターン抜き出しに失敗しました。 "
            f"パターンに一致する文字列が見つかりません。 (対象: '{processed_filename}')"
        )
        
    return match.group(0)


def generate_error_video(
    npy_path: Path, 
    output_mp4: Path, 
    ground_truth: int, 
    predicted: int, 
    confidence: float, 
    H: int = 260, 
    W: int = 346, 
    fps: int = 30
) -> None:
    """
    既存の npy_to_video のロジックをベースに、画面上部にメタデータを焼き込む拡張版関数
    """
    if not npy_path.exists():
        print(f"❌ エラー: 元ファイルが見つかりません -> {npy_path}")
        return

    events = np.load(npy_path)
    if len(events) == 0:
        print(f"❌ イベントデータが空です: {npy_path.name}")
        return

    xs = events[:, 0].astype(np.int32)
    ys = events[:, 1].astype(np.int32)
    ts = events[:, 2]  # マイクロ秒 (us)
    ps = events[:, 3]

    # 画面外の異常座標カット
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys, ts, ps = xs[valid], ys[valid], ts[valid], ps[valid]

    t_min, t_max = ts.min(), ts.max()
    total_time_us = t_max - t_min
    
    frame_time_us = 1000000 / fps
    num_frames = int(total_time_us / frame_time_us) + 1

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(output_mp4), fourcc, fps, (W, H))

    for i in range(num_frames):
        t_start = t_min + i * frame_time_us
        t_end = t_start + frame_time_us

        mask = (ts >= t_start) & (ts < t_end)
        bin_xs = xs[mask]
        bin_ys = ys[mask]
        bin_ps = ps[mask]

        # 背景黒のフレーム生成
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # イベント描画 (BGR: 正->赤 / 負->青)
        for x, y, p in zip(bin_xs, bin_ys, bin_ps):
            if p == 1:
                frame[y, x] = (0, 0, 255)
            else:
                frame[y, x] = (255, 0, 0)

        # 🌟 メタデータのテキスト焼き込み (視認性の高い緑色、左上に配置)
        # 複数行の情報を綺麗に並べるため、少しずつ y 座標をずらして描画します
        text_gt   = f"GT (True): {ground_truth}"
        text_pred = f"Predicted : {predicted}"
        text_conf = f"Confidence: {confidence * 100:.2f}%"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        color = (0, 255, 0)  # 緑色
        thickness = 1
        line_type = cv2.LINE_AA
        
        cv2.putText(frame, text_gt,   (15, 25), font, font_scale, color, thickness, line_type)
        cv2.putText(frame, text_pred, (15, 45), font, font_scale, color, thickness, line_type)
        cv2.putText(frame, text_conf, (15, 65), font, font_scale, color, thickness, line_type)

        out.write(frame)

    out.release()


def visualize_confident_errors(
    json_path: Path, 
    output_dir: Path, 
    num_videos: int | str = 10,
    target_npy_dir: Path = config.ORIGINAL_DATA_DIR
) -> None:
    """
    JSON結果を読み込み、誤答(false)データを確信度が高い順にソートし、指定本数を動画化する
    """
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    
    if not json_path.exists():
        raise FileNotFoundError(f"❌ 指定された解析JSONファイルが見つかりません: {json_path}")
        
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    # 1. 誤答 (is_correct == False) データのみを抽出
    error_records = [r for r in records if not r.get("is_correct", True)]
    print(f"📊 総誤答数: {len(error_records)} 件 / 全データ数: {len(records)} 件")
    
    if not error_records:
        print("🎉 素晴らしい！誤答データが1件もありませんでした。処理を終了します。")
        return
        
    # 2. 確信度 (confidence) が高い順（降順）に並び替え
    error_records.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)

    if isinstance(num_videos, str) and num_videos.lower() == "all":
        targets = error_records
        print(f"🎬 すべての誤答データ {len(targets)} 件を動画化します...")
    else:
        # 指定本数にスライス
        targets = error_records[:num_videos]
        print(f"🎬 確信度の高い上位 {len(targets)} 件の誤答データを動画化します...")
    
    # 3. 大元データの検索と動画化ループ
    # config.py で定義された大元ノイズ除去データディレクトリをベースにする
    source_dir = target_npy_dir
    
    for idx, item in enumerate(targets):
        processed_filename = item["filename"]

        try:
            orig_filename = _extract_original_filename(processed_filename)
        except ValueError as e:
            print(f"[WARNING] {e}")
            continue
        
        # 大元ディレクトリ配下から再帰的に該当ファイルを探索（サブフォルダ対策）
        found_paths = list(source_dir.rglob(f"*{orig_filename}*.npy"))
        
        if not found_paths:
            print(f"⚠️ 警告: 大元ファイルがディレクトリ '{source_dir.name}' 内に見つかりません: {orig_filename}")
            continue
            
        target_npy_path = found_paths[0]
        
        # 出力動画ファイル名の定義
        # ランキング順位、元ファイル名、確信度をファイル名に含めて分かりやすく保存します
        conf_pct = int(item['confidence'] * 100)
        out_mp4_name = f"GT{item['ground_truth']}_Pred{item['predicted']}_{target_npy_path.stem}_rank{idx+1:02d}_conf{conf_pct:02d}.mp4"
        output_mp4_path = output_dir / out_mp4_name
        
        print(f"[{idx+1}/{len(targets)}] レンダリング中: {out_mp4_name}")
        
        generate_error_video(
            npy_path=target_npy_path,
            output_mp4=output_mp4_path,
            ground_truth=item["ground_truth"],
            predicted=item["predicted"],
            confidence=item["confidence"]
        )
        
    print(f"🎉 すべてのエラー動画化が完了しました！保存先 ➡️ {output_dir.resolve()}")