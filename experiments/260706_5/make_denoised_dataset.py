from expt_thu_eact_50_chl.denoise_hw import denoise_event_file, convert_labels_file
from expt_thu_eact_50_chl import config
from pathlib import Path
import re
from concurrent.futures import ProcessPoolExecutor

def process_single_file(file_path, current_dir):
    """1つのファイルを処理するヘルパー関数"""
    output_path = current_dir / "result" / f"{file_path.stem}_hw_filtered.npy"
    if file_path.exists():
        output = denoise_event_file(input_path=file_path, output_path=output_path, dt_ms=80.0)
        return f"[SUCCESS] 保存先: {output}"
    return f"[NOTE] ファイルが見つかりません: {file_path}"

if __name__ == "__main__":
    target_dir = config.ORIGINAL_DATA_DIR
    pattern = re.compile(r".*\.npy$")
    current_dir = Path(__file__).parent
    (current_dir / "result").mkdir(parents=True, exist_ok=True)

    # 処理対象のファイルリストを作成
    tasks = [fp for fp in target_dir.iterdir() if fp.is_file() and pattern.match(fp.name)]

    # CPUの多コアを使って並列実行（GPU化する場合も、ここから並列でGPUに投げると効率的です）
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_file, fp, current_dir) for fp in tasks]
        for future in futures:
            print(future.result())

    # ラベル変換（ここは一瞬で終わるはずなので並列化不要）
    convert_labels_file(config.ORIGINAL_DATA_DIR / "train.txt", current_dir / "result" / "train.txt", "_hw_filtered")
    convert_labels_file(config.ORIGINAL_DATA_DIR / "test.txt", current_dir / "result" / "test.txt", "_hw_filtered")