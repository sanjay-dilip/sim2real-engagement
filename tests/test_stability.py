"""
Tests for the shared repeated-cross-validation stability utility
(shared/stability.py), used by both the anime and Steam stability scripts.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from stability import (
    run_repeated_cv,
    summarize_importance_stability,
    summarize_metrics,
)


def _synthetic_classification_data(n_rows: int = 100, n_features: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features))
    # Make the first feature genuinely predictive so permutation importance
    # has something non-trivial to find, rather than pure noise.
    y = (X[:, 0] + rng.normal(scale=0.5, size=n_rows) > 0).astype(int)
    feature_cols = [f"f{i}" for i in range(n_features)]
    return X, y, feature_cols


def test_run_repeated_cv_produces_expected_fold_count():
    X, y, feature_cols = _synthetic_classification_data()
    n_splits, n_repeats = 4, 3

    result = run_repeated_cv(
        lambda: LogisticRegression(),
        X,
        y,
        feature_cols,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=0,
    )

    expected_folds = n_splits * n_repeats
    assert len(result["fold_metrics"]) == expected_folds
    assert len(result["fold_importances"]) == expected_folds
    # Every fold's metric dict should have the same keys.
    metric_keys = set(result["fold_metrics"][0].keys())
    assert all(set(m.keys()) == metric_keys for m in result["fold_metrics"])
    # Every fold's importance dict should cover every feature.
    assert all(set(imp.keys()) == set(feature_cols) for imp in result["fold_importances"])


def test_run_repeated_cv_is_reproducible_given_same_seed():
    X, y, feature_cols = _synthetic_classification_data()

    result_a = run_repeated_cv(
        lambda: LogisticRegression(), X, y, feature_cols, n_splits=3, n_repeats=2, random_state=7
    )
    result_b = run_repeated_cv(
        lambda: LogisticRegression(), X, y, feature_cols, n_splits=3, n_repeats=2, random_state=7
    )

    for m_a, m_b in zip(result_a["fold_metrics"], result_b["fold_metrics"]):
        assert m_a == m_b


def test_summarize_metrics_known_values():
    fold_metrics = [
        {"fold": 0, "accuracy": 0.8, "roc_auc": 0.9},
        {"fold": 1, "accuracy": 0.6, "roc_auc": 0.7},
        {"fold": 2, "accuracy": 1.0, "roc_auc": 1.0},
        {"fold": 3, "accuracy": 0.8, "roc_auc": 0.8},
    ]

    summary = summarize_metrics(fold_metrics)

    assert np.isclose(summary["accuracy"]["mean"], 0.8)
    assert summary["accuracy"]["n_folds"] == 4
    assert np.isclose(summary["roc_auc"]["mean"], 0.85)
    # std should be the sample std (ddof=1) of [0.8, 0.6, 1.0, 0.8]
    expected_std = np.std([0.8, 0.6, 1.0, 0.8], ddof=1)
    assert np.isclose(summary["accuracy"]["std"], expected_std)


def test_summarize_metrics_zero_variance_case():
    """
    Mirrors the Steam v1 case: every fold reports an identical metric value
    -- std must be exactly 0.0, not NaN or a tiny numerical artifact, since
    the README explicitly relies on "std == 0.0" as the leakage-confirmation
    signal.
    """
    fold_metrics = [{"fold": i, "accuracy": 1.0} for i in range(10)]
    summary = summarize_metrics(fold_metrics)
    assert summary["accuracy"]["mean"] == 1.0
    assert summary["accuracy"]["std"] == 0.0


def test_summarize_importance_stability_top_k_frequency():
    """
    Hand-constructed case: feature "a" is always the top feature (appears in
    top-1 every fold), "b" alternates, "c" is never in top-1.
    """
    fold_importances = [
        {"a": 0.9, "b": 0.5, "c": 0.1},
        {"a": 0.8, "b": 0.6, "c": 0.05},
        {"a": 0.7, "b": 0.65, "c": 0.2},
        {"a": 0.6, "b": 0.55, "c": 0.15},
    ]
    result = summarize_importance_stability(fold_importances, ["a", "b", "c"], top_k=1)

    assert result["features"]["a"]["top_k_frequency"] == 1.0
    assert result["features"]["a"]["mean_rank"] == 1.0
    assert result["features"]["a"]["std_rank"] == 0.0
    assert result["features"]["c"]["top_k_frequency"] == 0.0
    # b is always rank 2 (a is always rank 1, c is always rank 3), so b's
    # rank should also be perfectly stable even though it's never top-1.
    assert result["features"]["b"]["mean_rank"] == 2.0
