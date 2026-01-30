from pathlib import Path
# Path to the top level steam_real folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Data folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"
# Models folder
MODELS_DIR = PROJECT_ROOT / "models"
# Churn definition in days (will be used later in features.py)
CHURN_THRESHOLD_DAYS = 30