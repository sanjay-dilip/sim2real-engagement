"""
Entry point to run the full anime_simulated pipeline.
Steps:
1. Metadata pipeline
   - Load anime-offline-database.json
   - Build anime_master and episodes tables
   - Save to data/processed/
2. Simulation pipeline
   - Load processed anime metadata
   - Simulate user viewing logs
   - Save viewing_logs.parquet
3. Feature pipeline
   - Build ML dataset from viewing_logs
   - Save ml_dataset.parquet
4. Modeling pipeline
   - Train a classifier to predict label_next_episode
   - Print validation metrics
   - Save trained model to models/next_episode_model.pkl
Usage (run from project root, where anime_simulated/ lives):
    python -m anime_simulated.run_anime_pipeline
You can also import run_all() from other code or notebooks.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple
from anime_simulated.src.metadata_pipeline import run_metadata_pipeline
from anime_simulated.src.simulation_pipeline import run_simulation_pipeline
from anime_simulated.src.features import build_ml_dataset
from anime_simulated.src.models import train_model
def run_all(
    n_users: int = 5000,
    build_features: bool = True,
    train_model_flag: bool = True,
) -> Dict[str, Any]:
    """
    Run the end-to-end pipeline for the anime_simulated project.
    Args:
        n_users:
            Number of synthetic users to simulate in the viewing logs.
        build_features:
            If True, run the feature pipeline to create ml_dataset.parquet.
        train_model_flag:
            If True, train the model on the ML dataset and save it.
    Returns:
        A dict with the main artefacts:
            {
                "anime_df": anime_df,
                "episodes_df": episodes_df,
                "logs_df": logs_df,
                "ml_df": ml_df or None,
                "model": model or None,
                "metrics": metrics or None,
            }
    """
    # 1) Metadata (anime + episodes)
    anime_df, episodes_df = run_metadata_pipeline()
    # 2) Simulation (synthetic viewing logs)
    logs_df = run_simulation_pipeline(n_users=n_users)
    ml_df = None
    model = None
    metrics: Dict[str, Any] | None = None
    # 3) Feature pipeline (ML dataset)
    if build_features:
        ml_df = build_ml_dataset()
    # 4) Modeling pipeline (train and save model)
    if train_model_flag:
        model, metrics = train_model()
    return {
        "anime_df": anime_df,
        "episodes_df": episodes_df,
        "logs_df": logs_df,
        "ml_df": ml_df,
        "model": model,
        "metrics": metrics,
    }
if __name__ == "__main__":
    # Default: full pipeline with 5000 users, features, and model training
    run_all()