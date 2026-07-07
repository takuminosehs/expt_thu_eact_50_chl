from expt_thu_eact_50_chl.denoise_hw import denoise_event_file, convert_labels_file
from expt_thu_eact_50_chl import config
from pathlib import Path
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

def process_single_file(file_path: Path, current_dir: Path) -> dict:
    """1つのファイルを処理し、メトリクス用の辞書を返すヘルパー関数"""
    output_path = current_dir / "result" / f"{file_path.stem}_hw_filtered.npy"
    
    if not file_path.exists():
        return {
            "status": "error",
            "input_file": file_path.name,
            "message": "ファイルが見つかりません"
        }
        
    try:
        # 先ほど変更した denoise_event_file (DenoiseResultを返す) を実行
        result = denoise_event_file(input_path=file_path, output_path=output_path, psi=0)
        
        # JSONシリアライズ可能な辞書形式にして返す
        return {
            "status": "success",
            "input_file": file_path.name,
            "output_file": result.output_path.name,
            "original_count": result.original_count,
            "filtered_count": result.filtered_count,
            "reduction_rate_percent": result.reduction_rate
        }
    except Exception as e:
        return {
            "status": "error",
            "input_file": file_path.name,
            "message": str(e)
        }

if __name__ == "__main__":
    target_dir = config.ORIGINAL_DATA_DIR
    pattern = re.compile(r".*\.npy$")
    current_dir = Path(__file__).parent
    (current_dir / "result").mkdir(parents=True, exist_ok=True)

    # 処理対象のファイルリストを作成
    tasks = [fp for fp in target_dir.iterdir() if fp.is_file() and pattern.match(fp.name)]
    
    # 処理結果のメトリクスを格納するリスト
    metrics_data = []

    # CPUの多コアを使って並列実行
    with ProcessPoolExecutor() as executor:
        # submitでタスクを投げ、futureオブジェクトを管理
        futures = {executor.submit(process_single_file, fp, current_dir): fp for fp in tasks}
        
        # 処理が完了したものから順次結果を取得 (as_completedを使用)
        for future in as_completed(futures):
            res = future.result()
            
            if res["status"] == "success":
                print(f"[SUCCESS] 保存先: {res['output_file']} (削減率: {res['reduction_rate_percent']:.2f}%)")
            else:
                print(f"[ERROR] {res['input_file']} の処理中にエラーが発生: {res['message']}")
                
            # ステータスに関わらずログとして記録
            metrics_data.append(res)

    # メトリクスをJSONファイルとして保存
    metrics_path = current_dir / "denoise_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4, ensure_ascii=False)
        
    print(f"\n[INFO] 全ての処理が完了しました。メトリクスを保存しました: {metrics_path}")

    # ラベル変換（ここは一瞬で終わるはずなので並列化不要）
    convert_labels_file(config.ORIGINAL_DATA_DIR / "train.txt", current_dir / "result" / "train.txt", "_hw_filtered")
    convert_labels_file(config.ORIGINAL_DATA_DIR / "test.txt", current_dir / "result" / "test.txt", "_hw_filtered")