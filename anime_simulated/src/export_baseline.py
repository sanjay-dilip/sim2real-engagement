"""
Export a reproducible baseline metrics artifact for the anime continuation model.

Reuses the existing loading/split/evaluation functions from .models -- does not
retrain the model or change its predictions. Reads the model already saved to
models/next_episode_model.pkl and evaluates it on a validation split reproduced
with the same random_state used at training time.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict

from sklearn.inspection import permutation_importance

from .config import MODELS_DIR, MODEL_FILE, RNG_SEED
from .models import (
    load_ml_dataset,
    get_features_and_target,
    train_val_split,
    evaluate_model,
)

RESULTS_DIR = MODELS_DIR.parent / "results"
RESULTS_FILE = "baseline_metrics.json"

LEAKAGE_NOTE = (
    "anime_mean_p_continue is a direct aggregate of p_continue, the same "
    "probability used to draw label_next_episode during simulation "
    "(src/simulation_pipeline.py). Its importance should be read as a "
    "labeled leakage signal, not a genuine behavioral driver. See README.md."
)
METHOD_NOTE = (
    "Metrics and permutation importance are computed on the held-out validation "
    "split (20%%, random_state=%d), using the model already saved to "
    "models/%s. This script does not retrain the model." % (RNG_SEED, MODEL_FILE)
)


def load_trained_model() -> tuple[Any, list[str]]:
    """Load the already-trained model and its feature columns from disk."""
    with open(MODELS_DIR / MODEL_FILE, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["feature_cols"]


def export_baseline_metrics() -> Path:
    """
    Build baseline_metrics.json: validation metrics, Gini importance, and
    permutation importance for the currently-saved anime continuation model.
    """
    df = load_ml_dataset()
    X, y, feature_cols = get_features_and_target(df)
    X_train, X_val, y_train, y_val = train_val_split(X, y)

    model, saved_feature_cols = load_trained_model()
    if saved_feature_cols != feature_cols:
        raise ValueError(
            "Saved model feature columns do not match the current feature "
            "pipeline output -- rerun the training pipeline before exporting."
        )

    metrics = {k: float(v) for k, v in evaluate_model(model, X_val, y_val).items()}
    gini_importance = dict(zip(feature_cols, model.feature_importances_.tolist()))

    perm_result = permutation_importance(
        model, X_val, y_val, n_repeats=5, random_state=RNG_SEED, scoring="roc_auc"
    )
    permutation_importance_out = {
        "mean": dict(zip(feature_cols, perm_result.importances_mean.tolist())),
        "std": dict(zip(feature_cols, perm_result.importances_std.tolist())),
    }

    payload: Dict[str, Any] = {
        "model_type": type(model).__name__,
        "random_state": RNG_SEED,
        "split": {
            "method": "train_test_split",
            "test_size": 0.2,
            "stratify_on": "label_next_episode",
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
        },
        "feature_cols": feature_cols,
        "metrics": metrics,
        "gini_importance": gini_importance,
        "permutation_importance": permutation_importance_out,
        "notes": [LEAKAGE_NOTE, METHOD_NOTE],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / RESULTS_FILE
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    return out_path


if __name__ == "__main__":
    path = export_baseline_metrics()
    print(f"Baseline metrics written to: {path}")
