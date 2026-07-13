from pathlib import Path

# プロジェクト内主要ディレクトリへのパス
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# データソースとして使えるディレクトリ群（オーギュメンテーション後データ含む）
ORIGINAL_DATA_DIR = PROJECT_ROOT / "data" / "THU-EACT-50-CHL"

HW_DENOISED_DATA_DIR = PROJECT_ROOT / "experiments" / "260706_1" / "result"
HW_DENOISED_DT20_DATA_DIR = PROJECT_ROOT / "experiments" / "260706_3" / "result"
HW_DENOISED_DT80_DATA_DIR = PROJECT_ROOT / "experiments" / "260706_5" / "result"
HW_DENOISED_NOPSI_DATA_DIR = PROJECT_ROOT / "experiments" / "260706_7" / "result"


# 前処理後データも使いまわせるようにする。
# オーギュメンテーションなし、極性マップ統合、速度ベース4ch
PROCESSED_DATA_DIR_260707_1 = PROJECT_ROOT / "experiments" / "260707_1" / "processed_data"
PROCESSED_DATA_DIR_260710_9 = PROJECT_ROOT / "experiments" / "260710_9" / "processed_data"