import json
from pathlib import Path
import pandas as pd

def convert_prediction_json_to_excel(
    json_path: str | Path = "prediction_analysis.json",
    excel_path: str | Path = "prediction_analysis.xlsx"
) -> None:
    """
    推論結果のJSONファイルを読み込み、見やすいExcelファイル(.xlsx)として出力する関数。
    
    Args:
        json_path (str | Path): 入力JSONファイルのパス。デフォルトはカレントディレクトリの 'prediction_analysis.json'
        excel_path (str | Path): 出力Excelファイルのパス。デフォルトはカレントディレクトリの 'prediction_analysis.xlsx'
    """
    # 文字列で渡された場合でも pathlib.Path オブジェクトに統一
    json_path = Path(json_path)
    excel_path = Path(excel_path)
    
    # 入力ファイルの存在チェック
    if not json_path.exists():
        raise FileNotFoundError(f"❌ 入力JSONファイルが見つかりません: {json_path.resolve()}")
        
    print(f"📖 JSONファイルを読み込んでいます: {json_path.name}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # データをDataFrameに変換
    df = pd.DataFrame(data)
    
    # 出力先ディレクトリが存在しない場合は自動作成
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Excelへの書き出しとレイアウトの微調整
    sheet_name = "Prediction_Results"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        
        # 🌟 再利用時に嬉しい親切設計: 文字数に合わせて列の幅を自動調整する
        worksheet = writer.sheets[sheet_name]
        for col in worksheet.columns:
            # 列の中で最も長い文字列の長さを取得 (ヘッダー含む)
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = col[0].column_letter
            # 少し余裕（+3）を持たせて幅を設定（最低幅は10）
            worksheet.column_dimensions[col_letter].width = max(max_len + 3, 10)
            
    print(f"🎉 Excelファイルの出力が完了しました！ ➡️ {excel_path.resolve()}")

# --- スクリプトとして直接実行された場合のテストコード ---
if __name__ == "__main__":
    # 1. デフォルト設定での呼び出し（カレントディレクトリ配下を対象とする）
    try:
        convert_prediction_json_to_excel()
    except FileNotFoundError as e:
        print(f"[INFO] デフォルト実行テスト: {e} (ファイルがないためスキップしました)")

    # 2. パスを明示的に指定して呼び出す例（別ディレクトリへの出力など）
    # target_json = Path("/path/to/experiments/260707_1/prediction_analysis.json")
    # output_xlsx = Path("/path/to/output/result.xlsx")
    # convert_prediction_json_to_excel(json_path=target_json, excel_path=output_xlsx)