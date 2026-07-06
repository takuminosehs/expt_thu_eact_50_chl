from expt_thu_eact_50_chl.denoise_hw import denoise_event_file
from expt_thu_eact_50_chl import config
from pathlib import Path
import re

if __name__ == "__main__":
    target_dir = config.ORIGINAL_DATA_DIR
    pattern = re.compile(r".*\.npy$")
    current_dir = Path(__file__).parent
    # 使用例：config.pyに定義されたディレクトリからサンプルを処理する場合
    # ※実際のファイル名に合わせて変更してください
    for file_path in target_dir.iterdir():
        if file_path.is_file() and pattern.match(file_path.name):
            output_path = current_dir / "result" / str(file_path.stem + "_hw_filtered.npy")
            if file_path.exists():
                output = denoise_event_file(input_path=file_path, output_path=output_path)
                print(f"[SUCCESS] 保存先: {output}")
            else:
                print(f"[NOTE] サンプルファイルが指定のデータディレクトリ内に見つかりません: {file_path}")