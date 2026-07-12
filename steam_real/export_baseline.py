"""
Export a reproducible baseline metrics artifact for the Steam churn model.

Reuses FEATURE_COLS and load_ml_dataset from src.modeling -- does not retrain
the model or change its predictions. Reads the model already saved to
models/churn_model.pkl (which also carries the metrics computed at training
time) and computes permutation importance on a validation split reproduced
with the same split parameters used at training time.

Run from within steam_real/ (matches run_steam_pipeline.py's import style):
    cd steam_real && python export_baseline.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.config import MODELS_DIR
from src.modeling import FEATURE_COLS, load_ml_dataset

RESULTS_DIR = MODELS_DIR.parent / "results"
RESULTS_FILE = "baseline_metrics.json"
MODEL_FILE = "churn_model.pkl"
RANDOM_STATE = 42

CIRCULARITY_NOTE = (
    "churned is defined as the bottom 20th percentile of total_playtime_value "
    "(src/features.py). total_playtime_value, avg_session_length, and "
    "playtime_per_game are all directly derived from that same column -- this "
    "is documented label circularity, not a genuine behavioral signal. "
    "See README.md."
)
METHOD_NOTE = (
    "metrics are read from the saved model artifact (models/%s), computed at "
    "training time on its held-out validation split. This script does not "
    "retrain the model; permutation importance is computed on a validation "
    "split reproduced with the same split parameters (test_size=0.2, "
    "random_state=%d, stratify=churned) used at training time."
    % (MODEL_FILE, RANDOM_STATE)
)


def load_trained_model() -> tuple[Any, Dict[str, Any]]:
    """Load the already-trained model and its saved metrics from disk."""
    with open(MODELS_DIR / MODEL_FILE, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["metrics"]


def export_baseline_metrics() -> Path:
    """
    Build baseline_metrics.json: saved training-time metrics, Gini importance,
    and permutation importance for the currently-saved Steam churn model.
    """
    df = load_ml_dataset()
    df = df.dropna(subset=["churned"])
    X = df[FEATURE_COLS]
    y = df["churned"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model, saved_metrics = load_trained_model()
    metrics = {
        k: (float(v) if v is not None else None) for k, v in saved_metrics.items()
    }
    gini_importance = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))

    perm_result = permutation_importance(
        model, X_val, y_val, n_repeats=5, random_state=RANDOM_STATE, scoring="roc_auc"
    )
    permutation_importance_out = {
        "mean": dict(zip(FEATURE_COLS, perm_result.importances_mean.tolist())),
        "std": dict(zip(FEATURE_COLS, perm_result.importances_std.tolist())),
    }

    payload: Dict[str, Any] = {
        "model_type": type(model).__name__,
        "random_state": RANDOM_STATE,
        "split": {
            "method": "train_test_split",
            "test_size": 0.2,
            "stratify_on": "churned",
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
        },
        "feature_cols": FEATURE_COLS,
        "metrics": metrics,
        "gini_importance": gini_importance,
        "permutation_importance": permutation_importance_out,
        "notes": [CIRCULARITY_NOTE, METHOD_NOTE],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / RESULTS_FILE
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    return out_path


if __name__ == "__main__":
    path = export_baseline_metrics()
    print(f"Baseline metrics written to: {path}")
