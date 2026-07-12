from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.config import RAW_DIR, PROCESSED_DIR
def load_raw_logs(path: Path | None = None) -> pd.DataFrame:
    """
    Load the steam-200k dataset.
    Expected format (no header in file):
        user_id, game_name, behavior, value, ...
    Returns all rows (both "purchase" and "play" behavior) -- downstream
    functions filter to the behavior type they need. Previously this
    filtered to "play" rows here, which silently discarded every
    "purchase" row (and any user who purchased but never played) before
    any other code saw them.
    """
    if path is None:
        path = RAW_DIR / "steam-200k.csv"
    # steam-200k usually has no header row
    df = pd.read_csv(path, header=None)
    # Give basic names to the first four columns
    # If your file has more columns, they will just keep default names
    base_cols = ["user_id", "game_name", "behavior", "value"]
    for i, col in enumerate(base_cols):
        if i < df.shape[1]:
            df.rename(columns={df.columns[i]: col}, inplace=True)
    return df
def clean_and_build_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn raw "play" behavior logs into a simple session table.
    Since steam-200k does not have timestamps, we treat each play row as a
    play event and aggregate to user + game level. "purchase" rows are
    excluded here -- see build_purchase_table for those.
    """
    play_df = df[df["behavior"] == "play"].copy() if "behavior" in df.columns else df.copy()
    # Make sure value is numeric (minutes or hours depending on dataset)
    if "value" in play_df.columns:
        play_df["value"] = pd.to_numeric(play_df["value"], errors="coerce").fillna(0)
    else:
        play_df["value"] = 0.0
    session_df = (
        play_df.groupby(["user_id", "game_name"])
        .agg(
            total_playtime_value=("value", "sum"),
            sessions=("value", "count"),
        )
        .reset_index()
    )
    return session_df
def build_purchase_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a user-level table of distinct owned games, from "purchase"
    behavior rows. Used by features.py to compute play_ratio -- an
    alternate churn signal based on library breadth rather than playtime,
    without assuming timestamps or session data that steam-200k doesn't have.
    """
    purchase_df = df[df["behavior"] == "purchase"] if "behavior" in df.columns else df.iloc[0:0]
    owned_df = (
        purchase_df.groupby("user_id")["game_name"]
        .nunique()
        .rename("owned_games")
        .reset_index()
    )
    return owned_df
def save_session_events(session_df: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "session_events.parquet"
    session_df.to_parquet(out_path, index=False)
    return out_path
def save_purchase_table(owned_df: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "purchases.parquet"
    owned_df.to_parquet(out_path, index=False)
    return out_path
def run_ingestion() -> Path:
    print("[ingestion] Loading steam-200k logs...")
    raw_df = load_raw_logs()
    print(f"[ingestion] Raw rows: {len(raw_df)}")
    print("[ingestion] Aggregating to user + game sessions...")
    session_df = clean_and_build_sessions(raw_df)
    print(f"[ingestion] Session rows: {len(session_df)}")
    print("[ingestion] Saving session_events.parquet...")
    out_path = save_session_events(session_df)
    print(f"[ingestion] Done. Saved to {out_path}")
    print("[ingestion] Building purchase table (owned games per user)...")
    owned_df = build_purchase_table(raw_df)
    print(f"[ingestion] Purchase rows: {len(owned_df)}")
    print("[ingestion] Saving purchases.parquet...")
    purchase_path = save_purchase_table(owned_df)
    print(f"[ingestion] Done. Saved to {purchase_path}")
    return out_path
if __name__ == "__main__":
    run_ingestion()
