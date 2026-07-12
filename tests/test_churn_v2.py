"""
Tests for the alternate (v2, library-breadth) Steam churn definition and its
leakage guard.
"""
import pandas as pd

from src.features import add_alternate_churn_features
from src.sensitivity import V1_FEATURE_COLS, V2_FEATURE_COLS, COMMON_FEATURE_COLS


def _user_df_fixture() -> pd.DataFrame:
    # Mirrors build_user_features' output shape (minus churned, added separately
    # by tests that need it) -- unique_games stands in for "played_games".
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5],
            "total_playtime_value": [10.0, 20.0, 30.0, 40.0, 50.0],
            "sessions": [1, 2, 3, 4, 5],
            "unique_games": [1, 2, 3, 8, 10],
            "avg_session_length": [10.0, 10.0, 10.0, 10.0, 10.0],
            "playtime_per_game": [10.0, 10.0, 10.0, 5.0, 5.0],
        }
    )


def _purchase_df_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5],
            "owned_games": [1, 2, 6, 8, 20],
        }
    )


def test_play_ratio_computed_correctly():
    user_df = _user_df_fixture()
    purchase_df = _purchase_df_fixture()

    out = add_alternate_churn_features(user_df, purchase_df).set_index("user_id")

    assert out.loc[1, "play_ratio"] == 1 / 1  # 1.0
    assert out.loc[2, "play_ratio"] == 2 / 2  # 1.0
    assert out.loc[3, "play_ratio"] == 3 / 6  # 0.5
    assert out.loc[4, "play_ratio"] == 8 / 8  # 1.0
    assert out.loc[5, "play_ratio"] == 10 / 20  # 0.5


def test_churned_v2_is_bottom_20_percent_of_play_ratio():
    user_df = _user_df_fixture()
    purchase_df = _purchase_df_fixture()

    out = add_alternate_churn_features(user_df, purchase_df)
    cutoff = out["play_ratio"].quantile(0.20)
    expected = (out["play_ratio"] <= cutoff).astype(int)

    assert (out["churned_v2"] == expected).all()
    # Users 3 and 5 (play_ratio=0.5) are the lowest -- they should be the
    # ones flagged, not users 1/2/4 (play_ratio=1.0).
    flagged = set(out.loc[out["churned_v2"] == 1, "user_id"])
    assert flagged == {3, 5}


def test_owned_games_missing_purchase_record_fills_zero():
    """
    Safety-net behavior: a user with play events but no matching purchase
    record (not expected in steam-200k.csv per EDA, but not assumed) should
    get owned_games=0, and play_ratio should not raise or produce inf/NaN
    (owned_games is clipped to a minimum of 1 before dividing).
    """
    user_df = _user_df_fixture()
    purchase_df = _purchase_df_fixture().iloc[1:]  # drop user 1's purchase row

    out = add_alternate_churn_features(user_df, purchase_df).set_index("user_id")

    assert out.loc[1, "owned_games"] == 0
    assert out.loc[1, "play_ratio"] == 1 / 1  # unique_games=1, clipped denom=1
    assert not out["play_ratio"].isna().any()
    assert not (out["play_ratio"] == float("inf")).any()


def test_v2_feature_set_excludes_label_construction_columns():
    """
    Leakage guard: churned_v2 is built directly from unique_games and
    owned_games (via play_ratio). If any of unique_games, owned_games, or
    play_ratio ever end up back in V2_FEATURE_COLS, the v2 model would
    trivially predict its own label -- this test fails loudly if that
    regresses.
    """
    leakage_cols = {"unique_games", "owned_games", "play_ratio"}
    assert leakage_cols.isdisjoint(set(V2_FEATURE_COLS))


def test_v1_feature_set_unchanged():
    """
    v1's feature set must remain exactly what modeling.py's FEATURE_COLS
    already used -- this issue must not change v1's inputs.
    """
    assert V1_FEATURE_COLS == [
        "total_playtime_value",
        "sessions",
        "unique_games",
        "avg_session_length",
        "playtime_per_game",
    ]


def test_common_feature_cols_is_intersection():
    assert set(COMMON_FEATURE_COLS) == set(V1_FEATURE_COLS) & set(V2_FEATURE_COLS)
    assert "unique_games" not in COMMON_FEATURE_COLS
