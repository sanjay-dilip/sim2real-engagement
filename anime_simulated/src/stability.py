"""
Repeated-split stability analysis for the anime continuation model.

Every metric elsewhere in this repo for this model comes from exactly one
80/20 split (see models.py / results/baseline_metrics.json). This module
runs RepeatedStratifiedKFold(n_splits=5, n_repeats=10) instead, and reports
mean/std/CI for each metric plus feature-importance rank stability, so those
single-split numbers can be read as a point estimate with a known range
rather than an unqualified fact.

Benchmarked before use: a single GradientBoostingClassifier fit + permutation
importance on this dataset takes ~9s, so 50 folds (~7.5 min) is the default
here -- no reduction in n_repeats was needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

from sklearn.ensemble import GradientBoostingClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
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

from .config import MODELS_DIR, RNG_SEED
from .models import get_features_and_target, load_ml_dataset

RESULTS_DIR = MODELS_DIR.parent / "results"
RESULTS_FILE = "stability_anime.json"
TOP_K = 3


def _model_factory() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(random_state=RNG_SEED)


def run_anime_stability_analysis() -> Path:
    df = load_ml_dataset()
    X, y, feature_cols = get_features_and_target(df)

    print(f"[stability] Running {DEFAULT_N_SPLITS}x{DEFAULT_N_REPEATS} repeated CV "
          f"for the anime continuation model ({len(y):,} rows)...")
    cv_result = run_repeated_cv(
        _model_factory, X, y, feature_cols, random_state=RNG_SEED
    )
    metrics_summary = summarize_metrics(cv_result["fold_metrics"])
    importance_summary = summarize_importance_stability(
        cv_result["fold_importances"], feature_cols, top_k=TOP_K
    )

    print("[stability] Mean +/- std across folds:")
    for name, s in metrics_summary.items():
        print(f"  {name:10s}: {s['mean']:.4f} +/- {s['std']:.4f}")

    payload: Dict[str, Any] = {
        "model_type": "GradientBoostingClassifier",
        "random_state": DEFAULT_RANDOM_STATE,
        "cv_method": "RepeatedStratifiedKFold",
        "n_splits": DEFAULT_N_SPLITS,
        "n_repeats": DEFAULT_N_REPEATS,
        "n_folds_total": DEFAULT_N_SPLITS * DEFAULT_N_REPEATS,
        "n_rows": int(len(y)),
        "feature_cols": feature_cols,
        "metrics_summary": metrics_summary,
        "feature_importance_stability": importance_summary,
        "notes": [
            "anime_mean_p_continue is a labeled leakage signal (see README.md); its "
            "importance-rank stability here should be read with that caveat, not as "
            "confirmation it is a genuine behavioral driver.",
            "These numbers come from a fresh RepeatedStratifiedKFold run, independent "
            "of the single 80/20 split used to train the persisted "
            "models/next_episode_model.pkl and documented in results/baseline_metrics.json. "
            "Both are legitimate; this file describes the *range*, the other describes "
            "one specific trained artifact.",
        ],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / RESULTS_FILE
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[stability] Done. Saved to {out_path}")
    return out_path


if __name__ == "__main__":
    run_anime_stability_analysis()
