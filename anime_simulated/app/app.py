import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
# ---------------------------------------------------
# Setup paths and imports
# ---------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from anime_simulated.src.config import (
    PROCESSED_DIR,
    VIEWING_LOGS_FILE,
    ML_DATASET_FILE,
    MODELS_DIR,
    MODEL_FILE,
    RNG_SEED,
)
st.set_page_config(
    page_title="Simulated Anime Retention",
    layout="wide",
)
# ---------------------------------------------------
# Data loading helpers
# ---------------------------------------------------
@st.cache_data
def load_logs() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / VIEWING_LOGS_FILE)
    df["watch_start_time"] = pd.to_datetime(df["watch_start_time"])
    return df
@st.cache_data
def load_ml_dataset() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / ML_DATASET_FILE)
    return df
@st.cache_resource
def load_model():
    with open(MODELS_DIR / MODEL_FILE, "rb") as f:
        payload = pickle.load(f)
    model = payload["model"]
    feature_cols = payload["feature_cols"]
    return model, feature_cols
@st.cache_data
def compute_predictions(ml_df: pd.DataFrame):
    model, feature_cols = load_model()
    X = ml_df[feature_cols].to_numpy(dtype=float)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        proba = 1.0 / (1.0 + np.exp(-scores))
    else:
        proba = model.predict(X).astype(float)
    preds = (proba >= 0.5).astype(int)
    out = ml_df.copy()
    out["y_true"] = out["label_next_episode"].astype(int)
    out["y_proba"] = proba
    out["y_pred"] = preds
    return out
# ---------------------------------------------------
# Load data once
# ---------------------------------------------------
logs_df = load_logs()
ml_df = load_ml_dataset()
pred_df = compute_predictions(ml_df)
model, feature_cols = load_model()
# ---------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------
st.sidebar.title("Simulated Retention")
section = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Anime retention",
        "Top anime comparison",
        "Cohort analysis",
        "Engagement insights",
        "Model explainer",
        "Episode predictions",
    ],
)
# Some quick global numbers for reuse
n_users = logs_df["user_id"].nunique()
n_anime = logs_df["anime_row_id"].nunique()
n_events = len(logs_df)
# ---------------------------------------------------
# Section: Overview
# ---------------------------------------------------
if section == "Overview":
    st.title("Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Users", f"{n_users:,}")
    col2.metric("Anime titles", f"{n_anime:,}")
    col3.metric("Viewing events", f"{n_events:,}")
    st.markdown("### Global continuation by episode")
    retention = (
        logs_df.groupby("episode_number")["label_next_episode"]
        .mean()
        .reset_index()
        .sort_values("episode_number")
    )
    st.line_chart(retention, x="episode_number", y="label_next_episode")
    st.markdown("### Binge depth per user anime pair")
    depth = (
        logs_df.groupby(["user_id", "anime_row_id"])["episode_number"]
        .max()
        .rename("max_episode")
        .reset_index()
    )
    depth_counts = depth["max_episode"].value_counts().sort_index()
    st.bar_chart(depth_counts)
# ---------------------------------------------------
# Section: Anime retention
# ---------------------------------------------------
elif section == "Anime retention":
    st.title("Anime specific retention")
    anime_list = (
        logs_df[["anime_row_id", "anime_title"]]
        .drop_duplicates()
        .sort_values("anime_title")
    )
    title = st.selectbox(
        "Select an anime",
        anime_list["anime_title"],
    )
    anime_id = anime_list.loc[
        anime_list["anime_title"] == title, "anime_row_id"
    ].iloc[0]
    anime_logs = logs_df[logs_df["anime_row_id"] == anime_id]
    col_top, col_stats = st.columns([2, 1])
    with col_top:
        st.subheader(f"Retention curve for {title}")
        retention = (
            anime_logs.groupby("episode_number")["label_next_episode"]
            .mean()
            .reset_index()
            .sort_values("episode_number")
        )
        st.line_chart(retention, x="episode_number", y="label_next_episode")
    with col_stats:
        st.subheader("Quick stats")
        users_for_anime = anime_logs["user_id"].nunique()
        views_for_anime = len(anime_logs)
        ep_max = anime_logs["episode_number"].max()
        st.write(f"Users who watched: **{users_for_anime:,}**")
        st.write(f"Viewing events: **{views_for_anime:,}**")
        st.write(f"Episodes observed: **{ep_max}**")
    st.markdown("### Episode depth distribution")
    ep_depth = anime_logs.groupby("user_id")["episode_number"].max()
    st.bar_chart(ep_depth.value_counts().sort_index())
# ---------------------------------------------------
# Section: Top anime comparison
# ---------------------------------------------------
elif section == "Top anime comparison":
    st.title("Top anime comparison")
    st.markdown("Compare retention curves for the most watched anime.")
    top_n = st.slider("Number of top titles", 3, 15, 5)
    anime_counts = (
        logs_df.groupby(["anime_row_id", "anime_title"])["user_id"]
        .count()
        .rename("views")
        .reset_index()
        .sort_values("views", ascending=False)
    )
    top_anime = anime_counts.head(top_n)
    top_ids = top_anime["anime_row_id"].tolist()
    top_logs = logs_df[logs_df["anime_row_id"].isin(top_ids)]
    per_anime_retention = (
        top_logs.groupby(["anime_title", "episode_number"])["label_next_episode"]
        .mean()
        .reset_index()
        .sort_values(["anime_title", "episode_number"])
    )
    # Pivot for a wide chart
    pivot = per_anime_retention.pivot(
        index="episode_number",
        columns="anime_title",
        values="label_next_episode",
    )
    st.line_chart(pivot)
    st.markdown("Top titles by views")
    st.dataframe(top_anime.reset_index(drop=True))
# ---------------------------------------------------
# Section: Cohort analysis
# ---------------------------------------------------
elif section == "Cohort analysis":
    st.title("Cohort analysis")
    user_first = (
        logs_df.groupby("user_id")["watch_start_time"]
        .min()
        .rename("first_watch")
        .reset_index()
    )
    logs_cf = logs_df.merge(user_first, on="user_id", how="left")
    logs_cf["cohort_date"] = logs_cf["first_watch"].dt.date
    logs_cf["days_since_first"] = (
        logs_cf["watch_start_time"] - logs_cf["first_watch"]
    ).dt.days
    cohort_sizes = (
        logs_cf.groupby("cohort_date")["user_id"]
        .nunique()
        .rename("cohort_size")
        .reset_index()
    )
    active_users = (
        logs_cf.groupby(["cohort_date", "days_since_first"])["user_id"]
        .nunique()
        .rename("active_users")
        .reset_index()
    )
    cohort_curve = active_users.merge(cohort_sizes, on="cohort_date", how="left")
    cohort_curve["survival_rate"] = (
        cohort_curve["active_users"] / cohort_curve["cohort_size"]
    )
    cohort_dates = sorted(cohort_curve["cohort_date"].unique())
    selected_cohorts = st.multiselect(
        "Choose cohorts to plot",
        cohort_dates,
        default=cohort_dates[:3],
    )
    if selected_cohorts:
        plot_df = cohort_curve[
            cohort_curve["cohort_date"].isin(selected_cohorts)
        ]
        plt.figure()
        for cohort in selected_cohorts:
            grp = plot_df[plot_df["cohort_date"] == cohort]
            plt.plot(
                grp["days_since_first"],
                grp["survival_rate"],
                marker="o",
                label=str(cohort),
            )
        plt.xlabel("Days since first watch")
        plt.ylabel("Survival rate")
        plt.title("Cohort survival")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt.gcf())
    else:
        st.info("Select at least one cohort to see the curve.")
# ---------------------------------------------------
# Section: Engagement insights
# --------------------------------------------------
elif section == "Engagement insights":
    st.title("Engagement insights")
    st.markdown("### Engagement level distribution")
    st.histogram = st.bar_chart(ml_df["engagement_level"])
    st.markdown("### Engagement vs episodes watched")
    user_total = (
        logs_df.groupby("user_id")["episode_number"]
        .count()
        .rename("episodes_watched")
        .reset_index()
    )
    merged = ml_df[["user_id", "engagement_level"]].drop_duplicates()
    merged = merged.merge(user_total, on="user_id", how="left")
    st.scatter_chart(
        merged,
        x="engagement_level",
        y="episodes_watched",
    )
    st.markdown("Users with higher engagement_level should, on average, watch more episodes.")
# ---------------------------------------------------
# Section: Model explainer
# ---------------------------------------------------
elif section == "Model explainer":
    st.title("Model explainer")
    y_true = pred_df["y_true"].to_numpy()
    y_pred = pred_df["y_pred"].to_numpy()
    y_proba = pred_df["y_proba"].to_numpy()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Validation style metrics")
        for k, v in metrics.items():
            st.write(f"**{k}**: {v:0.4f}")
    with col2:
        st.subheader("Score distribution")
        st.histogram = st.bar_chart(pd.Series(y_proba).value_counts(bins=20).sort_index())
    st.markdown("### Feature importance (permutation)")
    # Use a sample for speed if needed
    sample_size = min(8000, len(pred_df))
    sample_idx = np.random.default_rng(RNG_SEED).choice(
        len(pred_df), size=sample_size, replace=False
    )
    X_sample = pred_df.iloc[sample_idx][feature_cols].to_numpy(dtype=float)
    y_sample = pred_df.iloc[sample_idx]["y_true"].to_numpy(dtype=int)
    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        n_repeats=5,
        random_state=RNG_SEED,
        scoring="roc_auc",
    )
    fi_df = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(fi_df)
    plt.figure()
    plt.barh(fi_df["feature"], fi_df["importance_mean"], xerr=fi_df["importance_std"])
    plt.gca().invert_yaxis()
    plt.xlabel("Drop in ROC AUC when shuffled")
    plt.title("Permutation feature importance")
    plt.grid(axis="x")
    st.pyplot(plt.gcf())
# ---------------------------------------------------
# Section: Episode predictions
# ---------------------------------------------------
elif section == "Episode predictions":
    st.title("Episode level predictions")
    st.markdown(
        "Pick an anime and episode to see predicted continuation probabilities for that context."
    )
    anime_list = (
        pred_df[["anime_row_id", "anime_title"]]
        .drop_duplicates()
        .sort_values("anime_title")
    )
    title = st.selectbox(
        "Select an anime",
        anime_list["anime_title"],
    )
    anime_id = anime_list.loc[
        anime_list["anime_title"] == title, "anime_row_id"
    ].iloc[0]
    subset = pred_df[pred_df["anime_row_id"] == anime_id].copy()
    ep_numbers = sorted(subset["episode_number"].unique())
    ep_chosen = st.selectbox("Episode number", ep_numbers)
    ep_rows = subset[subset["episode_number"] == ep_chosen].copy()
    if ep_rows.empty:
        st.info("No rows for this episode.")
    else:
        st.markdown("### Distribution of predicted continuation")
        st.histogram = st.bar_chart(
            pd.Series(ep_rows["y_proba"]).value_counts(bins=20).sort_index()
        )
        st.markdown("### Example rows")
        st.dataframe(
            ep_rows[
                [
                    "user_id",
                    "anime_title",
                    "episode_number",
                    "completed_fraction",
                    "engagement_level",
                    "y_true",
                    "y_proba",
                    "y_pred",
                ]
            ].head(20)
        )
        st.markdown(
            "Values close to 1.0 mean the model is very confident the user will watch the next episode."
        )