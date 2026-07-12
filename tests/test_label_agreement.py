"""
Tests for label-agreement and segment-comparison calculations between the
two Steam churn definitions.
"""
import pandas as pd

from src.label_agreement import compare_label_definitions, compare_segment_churn_rates


def test_compare_label_definitions_known_case():
    """
    Hand-computed fixture: 10 users, churned/churned_v2 chosen so agreement
    counts, disagreement direction, and Cohen's kappa can be verified by hand.

    y1 (churned):    1 1 1 0 0 0 0 0 0 0   (3 positive)
    y2 (churned_v2): 1 1 0 1 0 0 0 0 0 0   (3 positive)

    Agreement: users 1,2 (both 1), users 5-10 (both 0) -> 8 agree, 2 disagree.
    newly_churned_under_v2 (y1=0, y2=1): user 4 -> 1
    no_longer_churned_under_v2 (y1=1, y2=0): user 3 -> 1
    """
    df = pd.DataFrame(
        {
            "user_id": range(1, 11),
            "churned": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "churned_v2": [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        }
    )

    result = compare_label_definitions(df)

    assert result["n_users"] == 10
    assert result["prevalence_v1"] == 0.3
    assert result["prevalence_v2"] == 0.3
    assert result["n_agree"] == 8
    assert result["n_disagree"] == 2
    assert result["agreement_rate"] == 0.8
    assert result["disagreement_rate"] == 0.2
    assert result["newly_churned_under_v2"] == 1
    assert result["no_longer_churned_under_v2"] == 1
    cm = result["confusion_matrix_v1_vs_v2"]
    assert cm == {"v1_0_v2_0": 6, "v1_0_v2_1": 1, "v1_1_v2_0": 1, "v1_1_v2_1": 2}


def test_compare_label_definitions_perfect_agreement_gives_kappa_one():
    df = pd.DataFrame(
        {
            "user_id": range(1, 11),
            "churned": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "churned_v2": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    result = compare_label_definitions(df)
    assert result["cohens_kappa"] == 1.0
    assert result["n_disagree"] == 0


def test_compare_segment_churn_rates_reports_size_and_rate_per_tier():
    df = pd.DataFrame(
        {
            "user_id": range(1, 11),
            "total_playtime_value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "churned": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "churned_v2": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    result = compare_segment_churn_rates(df)

    segments = pd.DataFrame(result["segments"]).set_index("playtime_tier")
    assert segments["segment_size"].sum() == 10
    # Every tier's churn_rate_v1/v2 should be a valid proportion in [0, 1].
    assert (segments["churn_rate_v1"] >= 0).all() and (segments["churn_rate_v1"] <= 1).all()
    assert (segments["churn_rate_v2"] >= 0).all() and (segments["churn_rate_v2"] <= 1).all()

    assert result["high_risk_v1_count"] == 2
    assert result["high_risk_v2_count"] == 5
