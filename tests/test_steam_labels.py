"""
Tests for steam_real's churn-label and feature-aggregation logic.

Uses small, synthetic fixtures -- not the full steam-200k.csv -- since these
tests protect the correctness of the label/feature *functions*, not the raw
dataset.
"""
import pandas as pd

from src.features import build_user_features


def test_churn_label_is_bottom_20_percent_of_playtime():
    """
    churned should be 1 for exactly the users at or below the 20th
    percentile of total_playtime_value, per src/features.py's documented
    definition, and 0 otherwise.
    """
    session_df = pd.DataFrame(
        {
            "user_id": list(range(1, 11)),
            "game_name": ["Game A"] * 10,
            "total_playtime_value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
            "sessions": [1] * 10,
        }
    )

    user_df = build_user_features(session_df)

    cutoff = user_df["total_playtime_value"].quantile(0.20)
    expected_churned = (user_df["total_playtime_value"] <= cutoff).astype(int)

    assert (user_df["churned"] == expected_churned).all()
    # With 10 evenly-spaced values, the bottom 20th percentile should flag
    # at least the single lowest-playtime user.
    assert user_df.loc[user_df["total_playtime_value"] == 1, "churned"].iloc[0] == 1
    assert user_df.loc[user_df["total_playtime_value"] == 100, "churned"].iloc[0] == 0


def test_churn_label_handles_ties_at_cutoff():
    """
    Users tied exactly at the cutoff value should all be labeled churned
    (the label uses <=, not <).
    """
    session_df = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5],
            "game_name": ["Game A"] * 5,
            # three users tied at the low end
            "total_playtime_value": [10, 10, 10, 50, 90],
            "sessions": [1] * 5,
        }
    )

    user_df = build_user_features(session_df)
    cutoff = user_df["total_playtime_value"].quantile(0.20)

    tied_users = user_df[user_df["total_playtime_value"] == 10]
    if cutoff >= 10:
        assert (tied_users["churned"] == 1).all()


def test_build_user_features_aggregation_correctness():
    """
    build_user_features should correctly aggregate a multi-row-per-user
    session table into user-level totals and derived ratios.
    """
    session_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "game_name": ["Game A", "Game B", "Game A"],
            "total_playtime_value": [100.0, 50.0, 30.0],
            "sessions": [2, 1, 3],
        }
    )

    user_df = build_user_features(session_df).set_index("user_id")

    # User 1: two games, playtime 100 + 50 = 150, sessions 2 + 1 = 3
    assert user_df.loc[1, "total_playtime_value"] == 150.0
    assert user_df.loc[1, "sessions"] == 3
    assert user_df.loc[1, "unique_games"] == 2
    assert user_df.loc[1, "avg_session_length"] == 150.0 / 3
    assert user_df.loc[1, "playtime_per_game"] == 150.0 / 2

    # User 2: one game, playtime 30, sessions 3
    assert user_df.loc[2, "total_playtime_value"] == 30.0
    assert user_df.loc[2, "sessions"] == 3
    assert user_df.loc[2, "unique_games"] == 1
    assert user_df.loc[2, "avg_session_length"] == 30.0 / 3
    assert user_df.loc[2, "playtime_per_game"] == 30.0 / 1


def test_build_user_features_zero_playtime_does_not_divide_by_zero():
    """
    sessions/unique_games are clipped to a minimum of 1 before dividing, so
    a user with zero playtime should not produce inf/NaN derived features.
    """
    session_df = pd.DataFrame(
        {
            "user_id": [1],
            "game_name": ["Game A"],
            "total_playtime_value": [0.0],
            "sessions": [0],
        }
    )

    user_df = build_user_features(session_df)

    assert user_df.loc[0, "avg_session_length"] == 0.0
    assert user_df.loc[0, "playtime_per_game"] == 0.0
