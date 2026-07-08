# experiments/260707_4/run_error_visualizer.py
import sys
from pathlib import Path

# プロジェクトルートのパス解決
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 作成した共通モジュールから関数をインポート
from src.expt_thu_eact_50_chl.model_check.error_visualizer import visualize_confident_errors

if __name__ == "__main__":
    current_dir = Path(__file__).parent.resolve()
    
    # 前回のステップで出力した予測結果のJSONパス
    analysis_json = current_dir / "result" / "prediction_analysis.json"
    
    # 誤答動画の保存先フォルダ
    error_video_dir = current_dir / "result" / "error_videos"
    
    # 🌟 確信度の高い順に何本動画化するかを指定
    NUM_VIDEOS_TO_GENERATE = 50
    
    # 実行
    visualize_confident_errors(
        json_path=analysis_json,
        output_dir=error_video_dir,
        num_videos=NUM_VIDEOS_TO_GENERATE
    )