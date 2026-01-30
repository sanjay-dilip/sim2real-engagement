"""
Modeling pipeline for anime_simulated.
Takes:
    data/processed/ml_dataset.parquet
Trains:
    A simple classifier to predict label_next_episode
Saves:
    - Trained model to models/next_episode_model.pkl
    - Prints basic metrics
"""
from __future__ import annotations
import pickle
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from .config import PROCESSED_DIR, ML_DATASET_FILE, MODELS_DIR, MODEL_FILE, RNG_SEED
def load_ml_dataset() -> pd.DataFrame:
    """
    Load the ML dataset created by the feature pipeline.
    """
    path = PROCESSED_DIR / ML_DATASET_FILE
    df = pd.read_parquet(path)
    return df
def get_features_and_target(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Split the ML dataframe into X (features) and y (target).
    """
    target_col = "label_next_episode"
    feature_cols = [
        "episode_number",
        "watch_time_sec",
        "completed_fraction",
        "engagement_level",
        "anime_num_watch_events",
        "anime_num_users",
        "anime_mean_p_continue",
        "user_prev_episodes",
        "user_prev_episodes_this_anime",
        "user_prev_cont_rate",
    ]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=int)
    return X, y, feature_cols
def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = RNG_SEED,
):
    """
    Simple train / validation split.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_val, y_train, y_val
def train_classifier(X_train: np.ndarray, y_train: np.ndarray) -> GradientBoostingClassifier:
    """
    Train a basic tree based classifier.
    You can swap this for LightGBM or XGBoost later if you want.
    """
    model = GradientBoostingClassifier(
        random_state=RNG_SEED,
    )
    model.fit(X_train, y_train)
    return model
def evaluate_model(
    model: GradientBoostingClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute basic evaluation metrics on the validation set.
    """
    y_pred = model.predict(X_val)
    # Some models have predict_proba, others do not. GradientBoosting does.
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)[:, 1]
    else:
        # fallback: use decision function and pass through a simple transform
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_val)
            y_proba = 1.0 / (1.0 + np.exp(-scores))
        else:
            # worst case, treat hard predictions as probabilities
            y_proba = y_pred.astype(float)
    metrics: Dict[str, Any] = {}
    metrics["accuracy"] = accuracy_score(y_val, y_pred)
    metrics["precision"] = precision_score(y_val, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_val, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_val, y_pred, zero_division=0)
    try:
        metrics["roc_auc"] = roc_auc_score(y_val, y_proba)
    except ValueError:
        metrics["roc_auc"] = np.nan
    return metrics
def save_model(model, feature_cols):
    out_path = MODELS_DIR / MODEL_FILE
    # Safety check: ensure folder exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_cols": feature_cols,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    return str(out_path)
def train_model() -> Tuple[GradientBoostingClassifier, Dict[str, Any]]:
    """
    High level training entrypoint.
    Steps:
      1. Load ML dataset
      2. Build X, y
      3. Train / validation split
      4. Train classifier
      5. Evaluate on validation set
      6. Save model to disk
    """
    df = load_ml_dataset()
    X, y, feature_cols = get_features_and_target(df)
    X_train, X_val, y_train, y_val = train_val_split(X, y)
    model = train_classifier(X_train, y_train)
    metrics = evaluate_model(model, X_val, y_val)
    model_path = save_model(model, feature_cols)
    print("Validation metrics:")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:0.4f}")
    print(f"\nModel saved to: {model_path}")
    return model, metrics
if __name__ == "__main__":
    train_model()