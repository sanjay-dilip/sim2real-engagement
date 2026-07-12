"""
Reproducibility tests for the fixed-seed train/validation splits both
pipelines rely on. Uses small synthetic fixtures rather than the full
datasets -- these tests protect the "same seed -> same split" guarantee that
baseline_metrics.json (and any future sensitivity/stability analysis) depends
on, without the cost of a full pipeline rerun.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from anime_simulated.src.models import train_val_split as anime_train_val_split


def _synthetic_binary_dataset(n_rows: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, 4))
    y = (rng.random(n_rows) < 0.3).astype(int)
    return X, y


def test_anime_train_val_split_is_reproducible():
    X, y = _synthetic_binary_dataset()

    X_train_a, X_val_a, y_train_a, y_val_a = anime_train_val_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_b, X_val_b, y_train_b, y_val_b = anime_train_val_split(
        X, y, test_size=0.2, random_state=42
    )

    np.testing.assert_array_equal(X_train_a, X_train_b)
    np.testing.assert_array_equal(X_val_a, X_val_b)
    np.testing.assert_array_equal(y_train_a, y_train_b)
    np.testing.assert_array_equal(y_val_a, y_val_b)


def test_steam_style_train_test_split_is_reproducible():
    """
    steam_real/src/modeling.py calls sklearn's train_test_split directly
    (test_size=0.2, random_state=42, stratify=y) rather than through a
    wrapper -- this test pins that exact call signature's determinism so a
    future refactor (e.g. Issue 2's shared evaluation split) can rely on it.
    """
    df = pd.DataFrame(_synthetic_binary_dataset(n_rows=200, seed=1)[0], columns=list("abcd"))
    y = pd.Series(_synthetic_binary_dataset(n_rows=200, seed=1)[1])

    X_train_a, X_val_a, y_train_a, y_val_a = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    pd.testing.assert_frame_equal(X_train_a, X_train_b)
    pd.testing.assert_frame_equal(X_val_a, X_val_b)
    pd.testing.assert_series_equal(y_train_a, y_train_b)
    pd.testing.assert_series_equal(y_val_a, y_val_b)
