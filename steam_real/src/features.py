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
def load_purchase_table(path: Path | None = None) -> pd.DataFrame:
    """
    Load the purchases parquet created by ingestion (owned_games per user).
    """
    if path is None:
        path = PROCESSED_DIR / "purchases.parquet"
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
def add_alternate_churn_features(
    user_df: pd.DataFrame, purchase_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add a second, non-circular churn definition based on library breadth
    rather than playtime depth.
    Adds:
      - owned_games: distinct games purchased by the user (from purchase
        events, independent of playtime)
      - play_ratio: unique_games / max(owned_games, 1) -- how much of a
        user's purchased library they actually played
      - churned_v2: 1 for users in the bottom 20 percent of play_ratio,
        0 otherwise
    Restricted to the same population as `churned` (v1) -- users already
    present in user_df, i.e. users with at least one play event. Every user
    in that population had at least one purchase record during EDA on
    steam-200k.csv, so owned_games is not expected to be missing in
    practice; the merge is still a left join with fillna(0) as a safety net
    for malformed/future data.
    Leakage note: churned_v2 is a direct function of unique_games and
    owned_games. Any model trained against churned_v2 must exclude
    unique_games, owned_games, and play_ratio from its feature set.
    """
    out = user_df.merge(purchase_df, on="user_id", how="left")
    out["owned_games"] = out["owned_games"].fillna(0).astype(int)
    out["play_ratio"] = out["unique_games"] / out["owned_games"].clip(lower=1)
    cutoff = out["play_ratio"].quantile(0.20)
    out["churned_v2"] = (out["play_ratio"] <= cutoff).astype(int)
    return out
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
    print("[features] Adding alternate (library breadth) churn label...")
    purchase_df = load_purchase_table()
    ml_df = add_alternate_churn_features(ml_df, purchase_df)
    print("[features] Saving ml_dataset.parquet...")
    out_path = save_ml_dataset(ml_df)
    print(f"[features] Done. Saved to {out_path}")
    return out_path
if __name__ == "__main__":
    run_feature_pipeline()