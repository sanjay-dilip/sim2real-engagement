"""
Tests for anime_simulated's label-generation and feature-engineering logic.

Uses small, synthetic fixtures constructed directly in-memory -- not the full
84MB anime-offline-database.json -- since these tests protect the
correctness/determinism of the label/feature *functions*, not the raw dataset.
"""
import numpy as np
import pandas as pd

from anime_simulated.src.simulation_pipeline import (
    BASE_RETENTION,
    base_retention_prob,
    simulate_user_viewing_logs,
)
from anime_simulated.src.features import (
    add_anime_level_features,
    add_user_level_features,
)


def _tiny_anime_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "anime_row_id": [1, 2],
            "title": ["Show A", "Show B"],
            "episodes": [12, 12],
            "score_weight": [0.5, 0.5],
        }
    )


def _tiny_users_fixture(n_users: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    return pd.DataFrame(
        {
            "user_id": np.arange(1, n_users + 1),
            "engagement_level": rng.beta(a=2.0, b=2.0, size=n_users),
        }
    )


def test_base_retention_prob_known_episodes():
    """
    base_retention_prob should return the exact BASE_RETENTION table value
    for tabled episodes, and flatten to 0.40 beyond the table.
    """
    for episode_number, expected in BASE_RETENTION.items():
        assert base_retention_prob(episode_number) == expected
    assert base_retention_prob(9) == 0.40
    assert base_retention_prob(48) == 0.40


def test_simulation_is_reproducible_given_same_seed():
    """
    label_next_episode is drawn from rng.random() < p_continue -- this test
    guards that the same seed always produces the identical simulated log,
    since every downstream metric (Issue 1's baseline_metrics.json, and any
    future sensitivity/stability analysis) depends on that determinism.
    """
    sim_anime_df = _tiny_anime_fixture()
    users_df = _tiny_users_fixture()

    logs_a = simulate_user_viewing_logs(
        users_df=users_df,
        sim_anime_df=sim_anime_df,
        rng=np.random.default_rng(42),
    )
    logs_b = simulate_user_viewing_logs(
        users_df=users_df,
        sim_anime_df=sim_anime_df,
        rng=np.random.default_rng(42),
    )

    pd.testing.assert_frame_equal(logs_a, logs_b)


def test_simulation_different_seeds_produce_different_logs():
    """
    Sanity check on the reproducibility test above: different seeds should
    (with overwhelming probability, given rng.random() draws and Gaussian
    noise throughout) NOT produce an identical log.
    """
    sim_anime_df = _tiny_anime_fixture()
    users_df = _tiny_users_fixture()

    logs_a = simulate_user_viewing_logs(
        users_df=users_df,
        sim_anime_df=sim_anime_df,
        rng=np.random.default_rng(1),
    )
    logs_b = simulate_user_viewing_logs(
        users_df=users_df,
        sim_anime_df=sim_anime_df,
        rng=np.random.default_rng(2),
    )

    assert not logs_a.equals(logs_b)


def _tiny_logs_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "anime_row_id": [10, 10, 20, 10, 10],
            "anime_title": ["A", "A", "B", "A", "A"],
            "watch_start_time": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-01",
                    "2024-01-05",
                ]
            ),
            "episode_number": [1, 2, 1, 1, 2],
            "watch_time_sec": [1000, 1000, 1000, 1000, 1000],
            "completed_fraction": [0.9, 0.9, 0.9, 0.9, 0.9],
            "engagement_level": [0.5, 0.5, 0.5, 0.7, 0.7],
            "label_next_episode": [1, 0, 1, 0, 1],
            "p_continue": [0.8, 0.6, 0.7, 0.4, 0.5],
        }
    )


def test_add_anime_level_features_aggregates_correctly():
    """
    anime_num_watch_events/anime_num_users/anime_mean_p_continue should be
    computed per anime_row_id and broadcast back onto every matching row.
    """
    logs_df = _tiny_logs_fixture()
    out = add_anime_level_features(logs_df)

    # anime_row_id 10 appears in rows 0, 1, 3, 4 (4 events, 2 unique users)
    anime_10_rows = out[out["anime_row_id"] == 10]
    assert (anime_10_rows["anime_num_watch_events"] == 4).all()
    assert (anime_10_rows["anime_num_users"] == 2).all()
    expected_mean_p = np.mean([0.8, 0.6, 0.4, 0.5])
    assert np.isclose(anime_10_rows["anime_mean_p_continue"].iloc[0], expected_mean_p)

    # anime_row_id 20 appears in row 2 only (1 event, 1 unique user)
    anime_20_rows = out[out["anime_row_id"] == 20]
    assert (anime_20_rows["anime_num_watch_events"] == 1).all()
    assert (anime_20_rows["anime_num_users"] == 1).all()
    assert np.isclose(anime_20_rows["anime_mean_p_continue"].iloc[0], 0.7)


def test_add_user_level_features_uses_only_past_events():
    """
    user_prev_episodes / user_prev_cont_rate must reflect only events before
    the current row (leakage guard on the existing feature logic) -- the
    first event for a user should show zero prior episodes and the global
    mean continuation rate, not any information from later rows.
    """
    logs_df = _tiny_logs_fixture()
    out = add_user_level_features(logs_df)
    out = out.sort_values(["user_id", "watch_start_time"]).reset_index(drop=True)

    user_1_rows = out[out["user_id"] == 1].reset_index(drop=True)
    # First event for user 1: no prior history at all.
    assert user_1_rows.loc[0, "user_prev_episodes"] == 0
    assert user_1_rows.loc[0, "user_prev_episodes_this_anime"] == 0
    global_mean = logs_df["label_next_episode"].mean()
    assert np.isclose(user_1_rows.loc[0, "user_prev_cont_rate"], global_mean)

    # Second event for user 1: one prior event, which had label_next_episode=1.
    assert user_1_rows.loc[1, "user_prev_episodes"] == 1
    assert np.isclose(user_1_rows.loc[1, "user_prev_cont_rate"], 1.0)
