from expt_thu_eact_50_chl.json_excel_exporter import convert_prediction_json_to_excel
from pathlib import Path

current_dir = Path(__file__).parent.resolve()

if __name__ == "__main__":
    # 1. デフォルト設定での呼び出し（カレントディレクトリ配下を対象とする）
    try:
        convert_prediction_json_to_excel(json_path=current_dir / "result" / "prediction_analysis.json", excel_path=current_dir / "result" / "prediction_analysis.xlsx")
    except FileNotFoundError as e:
        print(f"[INFO] デフォルト実行テスト: {e} (ファイルがないためスキップしました)")