"""
Cross-definition sensitivity analysis: trains a model against each Steam
churn definition (v1: playtime percentile, v2: library-breadth percentile)
on the SAME train/validation split, using the same model family and
hyperparameters, and compares performance and feature importance.

IMPORTANT: this does not touch or retrain the original churn_model.pkl
produced by modeling.py / documented in Issue 1's baseline_metrics.json.
That artifact used a *stratified* split (stratified by `churned`), which is
appropriate for a single model but not for a fair v1-vs-v2 comparison --
stratifying by one label would bias which rows land in validation relative
to the other label. This module re-trains a v1 model (same feature set,
same hyperparameters as modeling.py) on a *shared, unstratified* split so
v1 and v2 are evaluated on identical held-out users. The two v1 numbers
(original vs. shared-split) are expected to be close but not necessarily
identical -- both are reported, and neither replaces the other.

Leakage guard: churned_v2 is a direct function of unique_games and
owned_games (see features.add_alternate_churn_features), so the v2 feature
set excludes unique_games, owned_games, and play_ratio.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import MODELS_DIR, PROCESSED_DIR
from src.evaluation import (
    RANDOM_STATE,
    compute_permutation_importance,
    evaluate_classifier,
    shared_train_val_split,
)
from src.label_agreement import compare_label_definitions, compare_segment_churn_rates

RESULTS_DIR = MODELS_DIR.parent / "results"
RESULTS_FILE = "sensitivity_analysis.json"

V1_LABEL_COL = "churned"
V2_LABEL_COL = "churned_v2"

# v1: unchanged from modeling.py's FEATURE_COLS. unique_games is safe here
# -- only total_playtime_value is used to build the v1 label.
V1_FEATURE_COLS: List[str] = [
    "total_playtime_value",
    "sessions",
    "unique_games",
    "avg_session_length",
    "playtime_per_game",
]

# v2: excludes unique_games/owned_games/play_ratio, all of which are used
# to construct churned_v2 (see features.add_alternate_churn_features).
V2_FEATURE_COLS: List[str] = [
    "total_playtime_value",
    "sessions",
    "avg_session_length",
    "playtime_per_game",
]

# Features present in both sets -- the only fair basis for comparing
# importance *rankings* between v1 and v2.
COMMON_FEATURE_COLS: List[str] = [c for c in V1_FEATURE_COLS if c in V2_FEATURE_COLS]


def _make_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)


def _train_and_evaluate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
) -> Dict[str, Any]:
    X_train, y_train = train_df[feature_cols], train_df[label_col]
    X_val, y_val = val_df[feature_cols], val_df[label_col]

    model = _make_random_forest()
    model.fit(X_train, y_train)
    metrics = evaluate_classifier(model, X_val, y_val)
    importance = compute_permutation_importance(model, X_val, y_val, feature_cols)

    dummy = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)
    dummy_metrics = evaluate_classifier(dummy, X_val, y_val)

    return {
        "feature_cols": feature_cols,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "prevalence_train": float(y_train.mean()),
        "metrics": metrics,
        "permutation_importance": importance,
        "dummy_baseline_metrics": dummy_metrics,
    }


def run_sensitivity_analysis(df: pd.DataFrame | None = None) -> Path:
    if df is None:
        df = pd.read_parquet(PROCESSED_DIR / "ml_dataset.parquet")

    train_df, val_df = shared_train_val_split(df)

    print("[sensitivity] Training v1 (playtime percentile) on shared split...")
    v1_result = _train_and_evaluate(train_df, val_df, V1_FEATURE_COLS, V1_LABEL_COL)
    print(f"  v1 accuracy={v1_result['metrics']['accuracy']:.4f} "
          f"roc_auc={v1_result['metrics']['roc_auc']:.4f}")

    print("[sensitivity] Training v2 (library breadth) on shared split...")
    v2_result = _train_and_evaluate(train_df, val_df, V2_FEATURE_COLS, V2_LABEL_COL)
    print(f"  v2 accuracy={v2_result['metrics']['accuracy']:.4f} "
          f"roc_auc={v2_result['metrics']['roc_auc']:.4f}")

    print("[sensitivity] Comparing label definitions and segments...")
    label_agreement = compare_label_definitions(df)
    segment_comparison = compare_segment_churn_rates(df)

    common_importance_v1 = {
        k: v for k, v in v1_result["permutation_importance"]["mean"].items()
        if k in COMMON_FEATURE_COLS
    }
    common_importance_v2 = {
        k: v for k, v in v2_result["permutation_importance"]["mean"].items()
        if k in COMMON_FEATURE_COLS
    }
    rank_v1 = sorted(common_importance_v1, key=common_importance_v1.get, reverse=True)
    rank_v2 = sorted(common_importance_v2, key=common_importance_v2.get, reverse=True)

    payload: Dict[str, Any] = {
        "random_state": RANDOM_STATE,
        "split": {
            "method": "shared_train_val_split (unstratified, same rows for v1 and v2)",
            "test_size": 0.2,
        },
        "note": (
            "v1 here is retrained on a shared, unstratified split for a fair "
            "comparison with v2. This is NOT the same model/split as the original "
            "churn_model.pkl documented in Issue 1's baseline_metrics.json (that one "
            "used a stratified split). Both v1 numbers are legitimate; they answer "
            "different questions."
        ),
        "v1": v1_result,
        "v2": v2_result,
        "label_agreement": label_agreement,
        "segment_comparison": segment_comparison,
        "common_feature_importance_ranking": {
            "features_compared": COMMON_FEATURE_COLS,
            "v1_rank": rank_v1,
            "v2_rank": rank_v2,
            "top_feature_matches": rank_v1[0] == rank_v2[0] if rank_v1 and rank_v2 else None,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / RESULTS_FILE
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[sensitivity] Done. Saved to {out_path}")
    return out_path


if __name__ == "__main__":
    run_sensitivity_analysis()
