from pathlib import Path
import json
import pandas as pd
from .config import EXTERNAL_DIR, PROCESSED_DIR, ANIME_JSON_FILE, ANIME_MASTER_FILE, EPISODES_FILE
def load_raw_metadata(path: Path | None = None) -> dict:
    """
    Load the anime-offline-database JSON.
    Returns the full root dict so we can access `root['data']`.
    """
    if path is None:
        path = EXTERNAL_DIR / ANIME_JSON_FILE
    with open(path, "r", encoding="utf-8") as f:
        root = json.load(f)
    return root
def build_anime_master_table(anime_list: list[dict]) -> pd.DataFrame:
    """
    Build one row per anime with key fields.
    """
    rows = []
    for idx, anime in enumerate(anime_list):
        anime_season = anime.get("animeSeason") or {}
        duration = anime.get("duration") or {}
        score = anime.get("score") or {}
        rows.append(
            {
                "anime_row_id": idx,
                "title": anime.get("title"),
                "type": anime.get("type"),
                "episodes": anime.get("episodes"),
                "status": anime.get("status"),
                "season": anime_season.get("season"),
                "year": anime_season.get("year"),
                "duration_sec": duration.get("value"),
                "score_mean": score.get("arithmeticMean"),
                "score_median": score.get("median"),
                "num_tags": len(anime.get("tags", [])),
                "num_synonyms": len(anime.get("synonyms", [])),
                "num_sources": len(anime.get("sources", [])),
            }
        )
    df = pd.DataFrame(rows)
    return df
def build_episode_table(anime_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each anime into per-episode rows.
    """
    episode_rows = []
    for row in anime_df.itertuples(index=False):
        if pd.isna(row.episodes):
            continue
        try:
            n_eps = int(row.episodes)
        except (TypeError, ValueError):
            continue
        if n_eps <= 0:
            continue
        for ep in range(1, n_eps + 1):
            episode_rows.append(
                {
                    "anime_row_id": row.anime_row_id,
                    "title": row.title,
                    "type": row.type,
                    "year": row.year,
                    "episode_number": ep,
                }
            )
    episodes_df = pd.DataFrame(episode_rows)
    return episodes_df
def run_metadata_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full metadata step: load JSON, build master + episode tables, save to processed.
    """
    root = load_raw_metadata()
    anime_list = root["data"]
    anime_df = build_anime_master_table(anime_list)
    episodes_df = build_episode_table(anime_df)
    anime_path = PROCESSED_DIR / ANIME_MASTER_FILE
    episodes_path = PROCESSED_DIR / EPISODES_FILE
    anime_df.to_parquet(anime_path, index=False)
    episodes_df.to_parquet(episodes_path, index=False)
    print(f"Saved anime master to {anime_path}")
    print(f"Saved episodes table to {episodes_path}")
    return anime_df, episodes_df