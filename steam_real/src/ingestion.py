from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.config import RAW_DIR, PROCESSED_DIR
def load_raw_logs(path: Path | None = None) -> pd.DataFrame:
    """
    Load the steam-200k dataset.
    Expected format (no header in file):
        user_id, game_name, behavior, value, ...
    We will:
      - assign column names
      - keep only rows where behavior == "play"
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
    # Keep only "play" events
    if "behavior" in df.columns:
        df = df[df["behavior"] == "play"]
    return df
def clean_and_build_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn raw logs into a simple session table.
    Since steam-200k does not have timestamps, we treat each row as a play
    event and aggregate to user + game level.
    """
    # Make sure value is numeric (minutes or hours depending on dataset)
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    else:
        df["value"] = 0.0
    session_df = (
        df.groupby(["user_id", "game_name"])
        .agg(
            total_playtime_value=("value", "sum"),
            sessions=("value", "count"),
        )
        .reset_index()
    )
    return session_df
def save_session_events(session_df: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "session_events.parquet"
    session_df.to_parquet(out_path, index=False)
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
    return out_path
if __name__ == "__main__":
    run_ingestion()