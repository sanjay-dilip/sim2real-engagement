from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.config import PROCESSED_DIR
def load_session_events(path: Path | None = None) -> pd.DataFrame:
    """
    Load the session_events parquet created by ingestion.
    """
    if path is None:
        path = PROCESSED_DIR / "session_events.parquet"
    df = pd.read_parquet(path)
    return df
def build_user_features(session_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build user level features and a proxy churn label.
    Features:
      - total_playtime_value: total minutes (or units) played
      - sessions: total number of play events
      - unique_games: how many different games the user played
      - avg_session_length: total_playtime_value / sessions
      - playtime_per_game: total_playtime_value / unique_games
    Label:
      - churned = 1 for users in the bottom 20 percent of total_playtime_value
      - churned = 0 otherwise
    """
    # Aggregate to user level
    user_df = (
        session_df.groupby("user_id")
        .agg(
            total_playtime_value=("total_playtime_value", "sum"),
            sessions=("sessions", "sum"),
            unique_games=("game_name", "nunique"),
        )
        .reset_index()
    )
    # Derived features
    user_df["avg_session_length"] = (
        user_df["total_playtime_value"]
        / user_df["sessions"].clip(lower=1)
    )
    user_df["playtime_per_game"] = (
        user_df["total_playtime_value"]
        / user_df["unique_games"].clip(lower=1)
    )
    # Proxy churn label based on total playtime
    cutoff = user_df["total_playtime_value"].quantile(0.20)
    user_df["churned"] = (user_df["total_playtime_value"] <= cutoff).astype(int)
    return user_df
def save_ml_dataset(df: pd.DataFrame) -> Path:
    """
    Save the ML ready dataset.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "ml_dataset.parquet"
    df.to_parquet(out_path, index=False)
    return out_path
def run_feature_pipeline(session_events_path: Path | None = None) -> Path:
    """
    Full feature pipeline:
      1. load session_events
      2. build user features
      3. save ml_dataset.parquet
    """
    print("[features] Loading session events...")
    if session_events_path is not None:
        session_df = pd.read_parquet(session_events_path)
    else:
        session_df = load_session_events()
    print(f"[features] Session rows: {len(session_df)}")
    print("[features] Building user features and churn label...")
    ml_df = build_user_features(session_df)
    print(f"[features] Users in ML dataset: {len(ml_df)}")
    print("[features] Saving ml_dataset.parquet...")
    out_path = save_ml_dataset(ml_df)
    print(f"[features] Done. Saved to {out_path}")
    return out_path
if __name__ == "__main__":
    run_feature_pipeline()