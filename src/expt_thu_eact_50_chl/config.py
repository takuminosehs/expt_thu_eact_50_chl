from pathlib import Path

# プロジェクト内主要ディレクトリへのパス
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# データソースとして使えるディレクトリ群（オーギュメンテーション後データ含む）
ORIGINAL_DATA_DIR = PROJECT_ROOT / "data" / "THU-EACT-50-CHL"
