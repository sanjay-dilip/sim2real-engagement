"""
Label-agreement analysis between the two Steam churn definitions:
  - churned (v1): bottom 20th percentile of total_playtime_value
  - churned_v2:   bottom 20th percentile of play_ratio (library breadth)

Computed over the full user population (not just a validation split) --
label agreement is a population-level property of the two definitions, not
a model-evaluation metric.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def compare_label_definitions(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare `churned` (v1) against `churned_v2` on the same DataFrame.
    Expects both columns to already exist (see features.add_alternate_churn_features).
    """
    y1 = df["churned"].to_numpy()
    y2 = df["churned_v2"].to_numpy()
    n = len(df)

    agreement_mask = y1 == y2
    n_agree = int(agreement_mask.sum())
    n_disagree = int(n - n_agree)

    # Direction of change: users v1 called non-churned but v2 calls churned,
    # and vice versa.
    newly_churned_under_v2 = int(((y1 == 0) & (y2 == 1)).sum())
    no_longer_churned_under_v2 = int(((y1 == 1) & (y2 == 0)).sum())

    cm = confusion_matrix(y1, y2, labels=[0, 1])

    result: Dict[str, Any] = {
        "n_users": int(n),
        "prevalence_v1": float(y1.mean()),
        "prevalence_v2": float(y2.mean()),
        "agreement_rate": n_agree / n,
        "disagreement_rate": n_disagree / n,
        "n_agree": n_agree,
        "n_disagree": n_disagree,
        "n_label_changed": n_disagree,
        "pct_label_changed": round(100 * n_disagree / n, 2),
        "newly_churned_under_v2": newly_churned_under_v2,
        "no_longer_churned_under_v2": no_longer_churned_under_v2,
        "cohens_kappa": float(cohen_kappa_score(y1, y2)),
        "confusion_matrix_v1_vs_v2": {
            "v1_0_v2_0": int(cm[0, 0]),
            "v1_0_v2_1": int(cm[0, 1]),
            "v1_1_v2_0": int(cm[1, 0]),
            "v1_1_v2_1": int(cm[1, 1]),
        },
    }
    return result


def compare_segment_churn_rates(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare churn rate and segment size under v1 vs v2, using the existing
    playtime-quintile tier segmentation. Note this segmentation is built
    from total_playtime_value, the same variable v1's label is built from
    -- interpret the v1-vs-tier relationship with that circularity in mind.
    v2's relationship to tier is not circular (v2 is built from play_ratio,
    an independent variable), so a tier-level difference here is more load-
    bearing evidence of what "risk segment" a definition actually captures.
    """
    tiered = df.copy()
    tiered["playtime_tier"] = pd.qcut(
        tiered["total_playtime_value"],
        q=5,
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
    )
    by_tier = tiered.groupby("playtime_tier", observed=True).agg(
        segment_size=("user_id", "size"),
        churn_rate_v1=("churned", "mean"),
        churn_rate_v2=("churned_v2", "mean"),
    )
    by_tier["churn_rate_v1"] = by_tier["churn_rate_v1"].astype(float)
    by_tier["churn_rate_v2"] = by_tier["churn_rate_v2"].astype(float)

    high_risk_v1 = set(df.loc[df["churned"] == 1, "user_id"])
    high_risk_v2 = set(df.loc[df["churned_v2"] == 1, "user_id"])
    overlap = high_risk_v1 & high_risk_v2

    return {
        "segments": by_tier.reset_index().to_dict(orient="records"),
        "high_risk_v1_count": len(high_risk_v1),
        "high_risk_v2_count": len(high_risk_v2),
        "high_risk_overlap_count": len(overlap),
        "high_risk_overlap_pct_of_v1": round(100 * len(overlap) / max(len(high_risk_v1), 1), 2),
        "high_risk_overlap_pct_of_v2": round(100 * len(overlap) / max(len(high_risk_v2), 1), 2),
    }
