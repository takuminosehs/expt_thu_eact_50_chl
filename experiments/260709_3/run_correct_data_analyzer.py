# 検証用の実行例サンプル
from pathlib import Path
from expt_thu_eact_50_chl.model_check.correct_data_analyzer import analyze_and_export_to_json
from train import HoloEvNetBaseV4_SpatialOnly  # モデル定義をインポート

if __name__ == "__main__":
    # パスの定義
    current_dir = Path(__file__).parent.resolve()
    
    # 260706_2 の実験結果保存先などを指定
    model_file = current_dir / "best_model_spatial_only.pth"
    test_data_folder = current_dir.parent/ "260707_1" /"processed_data" / "test"
    output_json = current_dir / "result" / "prediction_analysis.json"
    
    # 今回試したチャネル数に合わせてモデルを初期化してラッパーに引き渡す
    model_inst = HoloEvNetBaseV4_SpatialOnly(num_classes=50)
    
    # 実行
    analyze_and_export_to_json(
        model_path=model_file,
        data_dir=test_data_folder,
        output_json_path=output_json,
        model_instance=model_inst
    )