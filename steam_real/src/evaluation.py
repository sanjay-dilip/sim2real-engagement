"""
Shared evaluation utilities for the Steam churn sensitivity analysis.

Used identically by the v1 (playtime-percentile) and v2 (library-breadth)
churn models, and the DummyClassifier baseline, so that reported metrics are
directly comparable: same split logic, same metric functions, same
permutation-importance method.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2


def shared_train_val_split(df: pd.DataFrame, random_state: int = RANDOM_STATE):
    """
    Split by row/user_id only -- NOT stratified by any label column -- so
    the exact same users land in train vs. validation regardless of which
    churn definition (v1 or v2) is being modeled. This is what makes the
    v1/v2 metric comparison a fair, same-population comparison rather than
    two differently-sampled experiments.

    Returns (train_df, val_df) as full DataFrames; callers pull whichever
    feature/label columns they need from these.
    """
    train_df, val_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=random_state
    )
    return train_df, val_df


def evaluate_classifier(model: Any, X_val, y_val) -> Dict[str, Any]:
    """
    Compute the full metric set used throughout this sensitivity analysis:
    accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and the confusion
    matrix. PR-AUC is reported explicitly (not just ROC-AUC) since v1 and v2
    have different class prevalence.
    """
    y_pred = model.predict(X_val)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)[:, 1]
    else:
        y_proba = y_pred.astype(float)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "prevalence_val": float(np.mean(y_val)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_val, y_proba))
    except ValueError:
        metrics["roc_auc"] = None
    try:
        metrics["pr_auc"] = float(average_precision_score(y_val, y_proba))
    except ValueError:
        metrics["pr_auc"] = None

    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
    return metrics


def compute_permutation_importance(
    model: Any,
    X_val,
    y_val,
    feature_cols: Sequence[str],
    n_repeats: int = 5,
    random_state: int = RANDOM_STATE,
    scoring: str = "roc_auc",
) -> Dict[str, Dict[str, float]]:
    """
    Permutation importance on the validation set, using the same method,
    scoring, and repeat count for every model in this analysis (v1, v2, and
    -- if ever compared -- the dummy baseline). Do not compare this against
    a model's .feature_importances_ (Gini) from elsewhere in the repo.
    """
    result = permutation_importance(
        model, X_val, y_val, n_repeats=n_repeats, random_state=random_state, scoring=scoring
    )
    return {
        "mean": dict(zip(feature_cols, result.importances_mean.tolist())),
        "std": dict(zip(feature_cols, result.importances_std.tolist())),
    }
