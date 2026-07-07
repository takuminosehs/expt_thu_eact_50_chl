import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# 制約: config.pyのインポート
from expt_thu_eact_50_chl import config
from expt_thu_eact_50_chl.npy_to_mp4 import npy_to_video


def process_video_task(input_npy: Path, output_mp4: Path) -> str:
    """1つのファイルに対する動画化を実行するヘルパー関数"""
    if not input_npy.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_npy}")
    
    # 既に動画が存在する場合はスキップ（再実行時の時短のため）
    if output_mp4.exists():
        return f"[SKIP] 既に存在します: {output_mp4.name}"
        
    npy_to_video(input_npy, output_mp4)
    return f"[SUCCESS] 動画を作成しました: {output_mp4.name}"


def generate_videos_from_ranking(ranking_json_path: Path):
    """
    ランキングJSONを読み込み、オリジナルとノイズ除去後のデータをそれぞれ動画化する関数
    """
    if not ranking_json_path.exists():
        raise FileNotFoundError(f"ランキングファイルが見つかりません: {ranking_json_path}")

    with open(ranking_json_path, "r", encoding="utf-8") as f:
        ranking_data = json.load(f)

    current_dir = Path(__file__).parent
    movies_dir = current_dir / "movies"
    result_dir = current_dir / "result"
    
    # 出力先ディレクトリの作成
    movies_dir.mkdir(parents=True, exist_ok=True)

    # 処理対象のデータをリスト化（上位と下位を結合）
    target_items = ranking_data.get("top_reductions", []) + ranking_data.get("bottom_reductions", [])
    
    if not target_items:
        print("[WARN] JSON内に処理対象のデータが見つかりませんでした。")
        return

    # 並列処理用のタスクリストを作成
    # 要素は (入力npyパス, 出力mp4パス) のタプル
    tasks = []
    for item in target_items:
        orig_npy_name = item["input_file"]
        filt_npy_name = item["output_file"]
        
        # オリジナルデータのパス構築
        orig_npy_path = config.ORIGINAL_DATA_DIR / orig_npy_name
        orig_mp4_path = movies_dir / f"{Path(orig_npy_name).stem}.mp4"
        
        # フィルタリング後データのパス構築
        filt_npy_path = result_dir / filt_npy_name
        filt_mp4_path = movies_dir / f"{Path(orig_npy_name).stem}_filtered.mp4"
        
        tasks.append((orig_npy_path, orig_mp4_path))
        tasks.append((filt_npy_path, filt_mp4_path))

    print(f"[INFO] 合計 {len(tasks)} 件の動画化タスクを開始します...")

    # 並列処理で一気に動画化を実行
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_video_task, npy, mp4): (npy, mp4) for npy, mp4 in tasks}
        
        for future in as_completed(futures):
            npy, mp4 = futures[future]
            try:
                msg = future.result()
                print(msg)
            except Exception as e:
                print(f"[ERROR] {npy.name} の動画化中にエラーが発生しました: {e}")

    print(f"\n[INFO] 全ての動画化処理が完了しました。出力先: {movies_dir}")


if __name__ == "__main__":
    # 実行ディレクトリの ranking_reduction_rate.json を対象とする
    target_json = Path(__file__).parent / "ranking_reduction_rate.json"
    generate_videos_from_ranking(target_json)