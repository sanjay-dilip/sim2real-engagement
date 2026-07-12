"""
Repeated-split stability analysis for both Steam churn models (v1 and v2).

Every metric elsewhere in this repo for these models comes from exactly one
fixed split (see modeling.py / sensitivity.py). This module runs
RepeatedStratifiedKFold(n_splits=5, n_repeats=10) for each definition
instead, and reports mean/std/CI plus feature-importance rank stability.

IMPORTANT -- read this before treating v1's stability as a positive result:
v1's near-perfect single-split accuracy/ROC-AUC is expected to repeat as
near-perfect with near-zero variance across all 50 folds too, because the
label (`churned`) is a deterministic function of a feature the model can
see (`total_playtime_value`). Low variance here confirms the leakage is
structural, not sensitive to which rows land in which fold -- it is NOT
evidence the model is good. See steam_real/README.md and Issue #1/#3.

Run from within steam_real/ (matches this project's existing import style):
    cd steam_real && python stability.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "shared") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "shared"))
from stability import (  # noqa: E402
    DEFAULT_N_REPEATS,
    DEFAULT_N_SPLITS,
    DEFAULT_RANDOM_STATE,
    run_repeated_cv,
    summarize_importance_stability,
    summarize_metrics,
)

from src.config import MODELS_DIR
from src.modeling import load_ml_dataset
from src.sensitivity import V1_FEATURE_COLS, V1_LABEL_COL, V2_FEATURE_COLS, V2_LABEL_COL

RESULTS_DIR = MODELS_DIR.parent / "results"
TOP_K = 3

V1_LEAKAGE_NOTE = (
    "Near-zero variance here CONFIRMS label circularity -- churned is a "
    "deterministic function of total_playtime_value, which the model can "
    "see directly. This is not evidence of a good model; it is evidence "
    "the leakage is structural and holds across every fold, not just the "
    "original single split. See steam_real/README.md."
)
V2_NOTE = (
    "v2's feature set excludes unique_games/owned_games/play_ratio (used to "
    "build churned_v2). Compare this file's ROC-AUC range against the "
    "DummyClassifier baseline reported in results/sensitivity_analysis.json "
    "before treating any single fold's performance as meaningful."
)


def _model_factory() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, random_state=DEFAULT_RANDOM_STATE, n_jobs=-1)


def _run_for_definition(
    df, feature_cols, label_col: str, note: str, results_file: str
) -> Path:
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[label_col].to_numpy(dtype=int)

    print(f"[stability] Running {DEFAULT_N_SPLITS}x{DEFAULT_N_REPEATS} repeated CV "
          f"for {label_col} ({len(y):,} rows)...")
    cv_result = run_repeated_cv(_model_factory, X, y, feature_cols, random_state=DEFAULT_RANDOM_STATE)
    metrics_summary = summarize_metrics(cv_result["fold_metrics"])
    importance_summary = summarize_importance_stability(
        cv_result["fold_importances"], feature_cols, top_k=TOP_K
    )

    print(f"[stability] {label_col} mean +/- std across folds:")
    for name, s in metrics_summary.items():
        print(f"  {name:10s}: {s['mean']:.4f} +/- {s['std']:.4f}")

    payload: Dict[str, Any] = {
        "label_col": label_col,
        "model_type": "RandomForestClassifier",
        "random_state": DEFAULT_RANDOM_STATE,
        "cv_method": "RepeatedStratifiedKFold",
        "n_splits": DEFAULT_N_SPLITS,
        "n_repeats": DEFAULT_N_REPEATS,
        "n_folds_total": DEFAULT_N_SPLITS * DEFAULT_N_REPEATS,
        "n_rows": int(len(y)),
        "prevalence": float(y.mean()),
        "feature_cols": feature_cols,
        "metrics_summary": metrics_summary,
        "feature_importance_stability": importance_summary,
        "notes": [note],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / results_file
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[stability] Done. Saved to {out_path}")
    return out_path


def run_steam_stability_analysis() -> Dict[str, Path]:
    df = load_ml_dataset()
    v1_path = _run_for_definition(
        df, V1_FEATURE_COLS, V1_LABEL_COL, V1_LEAKAGE_NOTE, "stability_steam_v1.json"
    )
    v2_path = _run_for_definition(
        df, V2_FEATURE_COLS, V2_LABEL_COL, V2_NOTE, "stability_steam_v2.json"
    )
    return {"v1": v1_path, "v2": v2_path}


if __name__ == "__main__":
    run_steam_stability_analysis()
