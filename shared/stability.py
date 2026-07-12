"""
Generic repeated-cross-validation stability utility, shared by the anime
continuation model and both Steam churn models (v1, v2).

Deliberately has no dependency on anime_simulated/ or steam_real/'s package
structure (both use different import conventions -- see their respective
READMEs) -- callers pass in already-loaded X/y and a zero-arg model factory,
and this module only depends on sklearn/numpy/pandas.

Every model in this repo is evaluated with RepeatedStratifiedKFold(n_splits=5,
n_repeats=10, random_state=42) unless a caller has a documented reason to
reduce n_repeats (e.g. an expensive model where 50 fold-fits is impractical).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold

DEFAULT_N_SPLITS = 5
DEFAULT_N_REPEATS = 10
DEFAULT_RANDOM_STATE = 42


def run_repeated_cv(
    model_factory: Callable[[], Any],
    X,
    y,
    feature_cols: Sequence[str],
    n_splits: int = DEFAULT_N_SPLITS,
    n_repeats: int = DEFAULT_N_REPEATS,
    random_state: int = DEFAULT_RANDOM_STATE,
    scoring: str = "roc_auc",
    permutation_n_repeats: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fit a fresh model (via model_factory()) on each of n_splits * n_repeats
    stratified folds, evaluate on the held-out fold, and compute permutation
    importance on that same held-out fold.

    Returns {"fold_metrics": [...], "fold_importances": [...]}, one entry
    per fold, in the order RepeatedStratifiedKFold produces them.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    fold_metrics: List[Dict[str, Any]] = []
    fold_importances: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X_arr, y_arr)):
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        model = model_factory()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)[:, 1]
        else:
            y_proba = y_pred.astype(float)

        metrics: Dict[str, Any] = {
            "fold": fold_idx,
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        }
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_val, y_proba))
        except ValueError:
            metrics["roc_auc"] = None
        try:
            metrics["pr_auc"] = float(average_precision_score(y_val, y_proba))
        except ValueError:
            metrics["pr_auc"] = None
        fold_metrics.append(metrics)

        result = permutation_importance(
            model,
            X_val,
            y_val,
            n_repeats=permutation_n_repeats,
            random_state=random_state,
            scoring=scoring,
        )
        fold_importances.append(dict(zip(feature_cols, result.importances_mean.tolist())))

    return {"fold_metrics": fold_metrics, "fold_importances": fold_importances}


def summarize_metrics(fold_metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Mean, std, and a 95% percentile interval (2.5th/97.5th percentile across
    folds) for each metric.
    """
    metric_names = [k for k in fold_metrics[0].keys() if k != "fold"]
    summary: Dict[str, Dict[str, float]] = {}
    for name in metric_names:
        values = np.array([m[name] for m in fold_metrics if m[name] is not None], dtype=float)
        if len(values) == 0:
            summary[name] = {"mean": None, "std": None, "ci_low": None, "ci_high": None}
            continue
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)),
            "n_folds": int(len(values)),
        }
    return summary


def summarize_importance_stability(
    fold_importances: List[Dict[str, float]],
    feature_cols: Sequence[str],
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    For each feature: how often it appears in the top-k most important
    features across folds (top_k_frequency, in [0, 1]), and its rank
    variability (mean/std of its importance rank across folds, rank 1 =
    most important). High std_rank means the feature's importance ranking
    is not stable across folds/splits.
    """
    n_folds = len(fold_importances)
    ranks_by_feature: Dict[str, List[int]] = {f: [] for f in feature_cols}
    top_k_hits: Dict[str, int] = {f: 0 for f in feature_cols}

    for importances in fold_importances:
        ranked = sorted(feature_cols, key=lambda f: importances.get(f, 0.0), reverse=True)
        for rank, feature in enumerate(ranked, start=1):
            ranks_by_feature[feature].append(rank)
            if rank <= top_k:
                top_k_hits[feature] += 1

    result: Dict[str, Any] = {"top_k": top_k, "n_folds": n_folds, "features": {}}
    for feature in feature_cols:
        ranks = np.array(ranks_by_feature[feature], dtype=float)
        result["features"][feature] = {
            "top_k_frequency": top_k_hits[feature] / n_folds if n_folds else None,
            "mean_rank": float(ranks.mean()) if len(ranks) else None,
            "std_rank": float(ranks.std(ddof=1)) if len(ranks) > 1 else 0.0,
        }
    return result
