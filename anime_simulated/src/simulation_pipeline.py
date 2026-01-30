from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from .config import PROCESSED_DIR, ANIME_MASTER_FILE, VIEWING_LOGS_FILE, RNG_SEED, get_rng
# Base continuation probabilities by episode index (1-based)
BASE_RETENTION = {
    1: 0.80,
    2: 0.72,
    3: 0.65,
    4: 0.60,
    5: 0.55,
    6: 0.50,
    7: 0.47,
    8: 0.44,
}
def base_retention_prob(episode_number: int) -> float:
    if episode_number in BASE_RETENTION:
        return BASE_RETENTION[episode_number]
    return 0.40  # flatten for later episodes
def load_anime_for_simulation() -> pd.DataFrame:
    """
    Load processed anime master and filter to a subset used for simulation.
    """
    anime_df = pd.read_parquet(PROCESSED_DIR / ANIME_MASTER_FILE)
    sim_anime_df = anime_df.query("type == 'TV'").copy()
    sim_anime_df = sim_anime_df[
        (sim_anime_df["episodes"].fillna(0).astype(int) >= 8)
        & (~sim_anime_df["year"].isna())
    ].copy()
    sim_anime_df["episodes"] = sim_anime_df["episodes"].astype(int)
    sim_anime_df["year"] = sim_anime_df["year"].astype(int)
    # score based sampling weights
    sim_anime_df["score_weight"] = sim_anime_df["score_mean"].fillna(
        sim_anime_df["score_mean"].median()
    )
    sim_anime_df["score_weight"] = np.clip(
        sim_anime_df["score_weight"] - sim_anime_df["score_weight"].min() + 0.1,
        a_min=0.1,
        a_max=None,
    )
    sim_anime_df["score_weight"] = (
        sim_anime_df["score_weight"] / sim_anime_df["score_weight"].sum()
    )
    return sim_anime_df
def create_users(n_users: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Create synthetic users with an engagement level in [0, 1].
    """
    users_df = pd.DataFrame(
        {
            "user_id": np.arange(1, n_users + 1),
            "engagement_level": rng.beta(a=2.0, b=2.0, size=n_users),
        }
    )
    return users_df
def simulate_user_viewing_logs(
    users_df: pd.DataFrame,
    sim_anime_df: pd.DataFrame,
    rng: np.random.Generator,
    max_shows_per_user: int = 5,
    simulation_days: int = 30,
) -> pd.DataFrame:
    """
    Main simulation function. Returns a viewing log DataFrame.
    """
    logs = []
    anime_values = sim_anime_df[["anime_row_id", "title", "episodes", "score_weight"]].to_numpy()
    anime_weights = anime_values[:, -1].astype(float)
    start_date = datetime(2024, 1, 1)
    for user in users_df.itertuples(index=False):
        k = rng.integers(1, max_shows_per_user + 1)
        chosen_idx = rng.choice(
            anime_values.shape[0],
            size=min(k, anime_values.shape[0]),
            replace=False,
            p=anime_weights,
        )
        for idx in chosen_idx:
            anime_row_id, title, n_eps, _w = anime_values[idx]
            n_eps = int(n_eps)
            start_day_offset = int(rng.integers(0, simulation_days))
            watch_start_time = start_date + timedelta(days=start_day_offset)
            episode_number = 1
            max_episodes_for_anime = min(n_eps, 48)
            while episode_number <= max_episodes_for_anime:
                base_p = base_retention_prob(episode_number)
                engagement_factor = 0.6 + 0.8 * float(user.engagement_level)
                noise = rng.normal(loc=0.0, scale=0.03)
                p_continue = base_p * engagement_factor + noise
                p_continue = float(np.clip(p_continue, 0.01, 0.99))
                duration_sec_default = 20 * 60
                frac_watched = float(
                    np.clip(rng.normal(loc=0.85, scale=0.10), 0.2, 1.0)
                )
                watch_time_sec = int(duration_sec_default * frac_watched)
                continue_flag = rng.random() < p_continue
                logs.append(
                    {
                        "user_id": user.user_id,
                        "engagement_level": float(user.engagement_level),
                        "anime_row_id": int(anime_row_id),
                        "anime_title": title,
                        "episode_number": int(episode_number),
                        "watch_start_time": watch_start_time,
                        "watch_time_sec": watch_time_sec,
                        "completed_fraction": frac_watched,
                        "label_next_episode": int(continue_flag),
                        "p_continue": p_continue,
                    }
                )
                if not continue_flag:
                    break
                episode_number += 1
                watch_start_time = watch_start_time + timedelta(
                    hours=float(rng.uniform(0.5, 48.0))
                )
    logs_df = pd.DataFrame(logs)
    return logs_df
def run_simulation_pipeline(
    n_users: int = 5000,
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    """
    Full simulation step: load anime, create users, simulate logs, save to processed.
    """
    rng = get_rng(seed)
    sim_anime_df = load_anime_for_simulation()
    users_df = create_users(n_users=n_users, rng=rng)
    logs_df = simulate_user_viewing_logs(
        users_df=users_df,
        sim_anime_df=sim_anime_df,
        rng=rng,
    )
    out_path = PROCESSED_DIR / VIEWING_LOGS_FILE
    logs_df.to_parquet(out_path, index=False)
    print(f"Simulated {len(logs_df):,} rows of viewing logs")
    print(f"Saved viewing logs to {out_path}")
    return logs_df