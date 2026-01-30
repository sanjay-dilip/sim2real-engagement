from pathlib import Path
import numpy as np
# Root of the anime_simulated project
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PACKAGE_ROOT / "models"
# Make sure these exist at runtime
for d in [DATA_DIR, EXTERNAL_DIR, RAW_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)
# File names
ANIME_JSON_FILE = "anime-offline-database.json"
ANIME_MASTER_FILE = "anime_master.parquet"
EPISODES_FILE = "episodes.parquet"
VIEWING_LOGS_FILE = "viewing_logs.parquet"
ML_DATASET_FILE = "ml_dataset.parquet"
MODEL_FILE = "next_episode_model.pkl"
# RNG seed
RNG_SEED = 42
def get_rng(seed: int = RNG_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)