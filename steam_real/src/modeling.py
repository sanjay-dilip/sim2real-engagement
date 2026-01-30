from __future__ import annotations

from pathlib import Path
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from src.config import PROCESSED_DIR, MODELS_DIR
FEATURE_COLS = [
    "total_playtime_value",
    "sessions",
    "unique_games",
    "avg_session_length",
    "playtime_per_game",
]
def load_ml_dataset(path: Path | None = None) -> pd.DataFrame:
    """
    Load the ml_dataset parquet created by the feature pipeline.
    """
    if path is None:
        path = PROCESSED_DIR / "ml_dataset.parquet"
    df = pd.read_parquet(path)
    return df
def train_churn_model(df: pd.DataFrame):
    """
    Train a simple RandomForest churn model.
    Uses FEATURE_COLS as inputs and 'churned' as the label.
    """
    df = df.dropna(subset=["churned"])
    X = df[FEATURE_COLS]
    y = df["churned"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    report = classification_report(y_val, y_pred, output_dict=True)
    try:
        roc = roc_auc_score(y_val, y_proba)
    except ValueError:
        roc = None
    metrics = {
        "accuracy": report["accuracy"],
        "precision_0": report["0"]["precision"],
        "recall_0": report["0"]["recall"],
        "precision_1": report["1"]["precision"],
        "recall_1": report["1"]["recall"],
        "roc_auc": roc,
    }
    return model, metrics
def save_model(model, metrics: dict) -> Path:
    """
    Save the trained model and metrics to disk.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "churn_model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"model": model, "metrics": metrics}, f)
    return out_path
def run_modeling_pipeline(ml_dataset_path: Path | None = None) -> Path:
    """
    Full modeling pipeline:
      1. load ml_dataset
      2. train model
      3. print metrics
      4. save model
    """
    print("[modeling] Loading ML dataset...")
    if ml_dataset_path is not None:
        df = pd.read_parquet(ml_dataset_path)
    else:
        df = load_ml_dataset()
    print(f"[modeling] Rows in ML dataset: {len(df)}")
    print("[modeling] Training churn model...")
    model, metrics = train_churn_model(df)
    print("[modeling] Metrics:")
    for k, v in metrics.items():
        if v is None:
            print(f"  {k}: None")
        else:
            print(f"  {k}: {v:.4f}")
    print("[modeling] Saving model...")
    out_path = save_model(model, metrics)
    print(f"[modeling] Done. Saved to {out_path}")
    return out_path
if __name__ == "__main__":
    run_modeling_pipeline()