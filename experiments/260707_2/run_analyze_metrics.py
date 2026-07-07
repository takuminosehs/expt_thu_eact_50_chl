from expt_thu_eact_50_chl import analyze_metrics
from pathlib import Path

if __name__ == "__main__":
    current_dir = Path(__file__).parent.resolve()
    analyze_metrics.generate_ranking_file(input_path=current_dir / "denoise_metrics.json", output_path=current_dir / "ranking_reduction_rate.json", n=5)