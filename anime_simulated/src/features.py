"""
Feature pipeline for the anime_simulated project.
Takes viewing logs from data/processed/viewing_logs.parquet and builds
a clean ML dataset for predicting label_next_episode.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from .config import PROCESSED_DIR, VIEWING_LOGS_FILE, ML_DATASET_FILE
def load_viewing_logs() -> pd.DataFrame:
    """
    Load the synthetic viewing logs written by the simulation pipeline.
    """
    path = PROCESSED_DIR / VIEWING_LOGS_FILE
    df = pd.read_parquet(path)
    return df
# -----------------------------
# Anime level features
# -----------------------------
def add_anime_level_features(logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple anime level aggregates:
      - anime_num_watch_events: total rows in logs for this anime
      - anime_num_users: number of unique users who watched this anime
      - anime_mean_p_continue: mean simulated continuation prob
    """
    anime_stats = (
        logs_df.groupby("anime_row_id")
        .agg(
            anime_num_watch_events=("user_id", "size"),
            anime_num_users=("user_id", "nunique"),
            anime_mean_p_continue=("p_continue", "mean"),
        )
        .reset_index()
    )
    out = logs_df.merge(anime_stats, on="anime_row_id", how="left")
    return out
# -----------------------------
# User level features
# -----------------------------
def add_user_level_features(logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add user level behavior features based on past activity.
    To avoid leakage, we:
      - sort by (user_id, watch_start_time)
      - compute cumulative stats
      - make sure each row uses only information from previous rows
    """
    df = logs_df.copy()
    # Make sure timestamps are datetime
    df["watch_start_time"] = pd.to_datetime(df["watch_start_time"])
    # Sort in time order per user
    df = df.sort_values(["user_id", "watch_start_time"])
    # 1) Episodes watched so far per user (before this event)
    # cumcount() starts at 0, so it is already "previous episodes"
    df["user_prev_episodes"] = df.groupby("user_id").cumcount()
    # 2) Episodes watched so far per user for this anime
    df["user_prev_episodes_this_anime"] = df.groupby(
        ["user_id", "anime_row_id"]
    ).cumcount()
    # 3) Previous continuation rate for the user (historical label_next_episode)
    # First, cumulative sum and count of label_next_episode
    df["user_cum_label_sum"] = df.groupby("user_id")["label_next_episode"].cumsum()
    df["user_cum_label_count"] = df.groupby("user_id").cumcount() + 1
    # Previous sum and count (exclude current event)
    df["user_prev_label_sum"] = df["user_cum_label_sum"] - df["label_next_episode"]
    df["user_prev_label_count"] = df["user_cum_label_count"] - 1
    # Avoid divide by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        df["user_prev_cont_rate"] = (
            df["user_prev_label_sum"] / df["user_prev_label_count"].replace(0, np.nan)
        )
    # For first events, there is no history. Fill with global mean continuation.
    global_mean = df["label_next_episode"].mean()
    df["user_prev_cont_rate"] = df["user_prev_cont_rate"].fillna(global_mean)
    # Drop helper columns
    df = df.drop(
        columns=[
            "user_cum_label_sum",
            "user_cum_label_count",
            "user_prev_label_sum",
            "user_prev_label_count",
        ]
    )
    return df
# -----------------------------
# Build ML dataset
# -----------------------------
def build_ml_dataset() -> pd.DataFrame:
    """
    Main feature pipeline.
    Steps:
      1. Load viewing logs
      2. Add anime level features
      3. Add user level features
      4. Select feature columns and target
      5. Save to data/processed/ml_dataset.parquet
    """
    logs_df = load_viewing_logs()
    # Add anime level features
    df = add_anime_level_features(logs_df)
    # Add user level features
    df = add_user_level_features(df)
    # Ensure consistent types for some columns
    df["episode_number"] = df["episode_number"].astype(int)
    df["watch_time_sec"] = df["watch_time_sec"].astype(int)
    df["completed_fraction"] = df["completed_fraction"].astype(float)
    df["engagement_level"] = df["engagement_level"].astype(float)
    df["anime_num_watch_events"] = df["anime_num_watch_events"].astype(int)
    df["anime_num_users"] = df["anime_num_users"].astype(int)
    df["anime_mean_p_continue"] = df["anime_mean_p_continue"].astype(float)
    df["user_prev_episodes"] = df["user_prev_episodes"].astype(int)
    df["user_prev_episodes_this_anime"] = df["user_prev_episodes_this_anime"].astype(int)
    df["user_prev_cont_rate"] = df["user_prev_cont_rate"].astype(float)
    # Target
    target_col = "label_next_episode"
    # Simple feature set for v1
    feature_cols = [
        "episode_number",
        "watch_time_sec",
        "completed_fraction",
        "engagement_level",
        "anime_num_watch_events",
        "anime_num_users",
        "anime_mean_p_continue",
        "user_prev_episodes",
        "user_prev_episodes_this_anime",
        "user_prev_cont_rate",
    ]
    # Keep only what we need for modeling plus ids for reference
    ml_df = df[
        [
            "user_id",
            "anime_row_id",
            "anime_title",
            "watch_start_time",
            target_col,
        ]
        + feature_cols
    ].copy()
    # Save
    out_path = PROCESSED_DIR / ML_DATASET_FILE
    ml_df.to_parquet(out_path, index=False)
    print(f"Saved ML dataset to {out_path}")
    print(f"Shape: {ml_df.shape[0]:,} rows x {ml_df.shape[1]} columns")
    return ml_df
if __name__ == "__main__":
    # Allow quick manual run:
    build_ml_dataset()