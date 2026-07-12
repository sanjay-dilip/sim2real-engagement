import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
sns.set(style="whitegrid")
# -----------------------
# Page config
# -----------------------
st.set_page_config(
    page_title="Steam Engagement & Proxy Churn",
    layout="wide",
)
st.title("🎮 Steam Engagement & Proxy Churn Dashboard")
st.markdown(
    """
This dashboard visualizes **real gameplay engagement** and **proxy churn**
using public Steam datasets.
⚠️ **Important**  
Churn shown here is **engagement-based (proxy)**, not time-based behavioral churn.
"""
)
# -----------------------
# Paths
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ML_DATASET_PATH = DATA_DIR / "ml_dataset.parquet"
MODEL_PATH = MODELS_DIR / "churn_model.pkl"
SENSITIVITY_PATH = PROJECT_ROOT / "results" / "sensitivity_analysis.json"
# -----------------------
# Load data
# -----------------------
@st.cache_data
def load_data():
    return pd.read_parquet(ML_DATASET_PATH)
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)["model"]
@st.cache_data
def load_sensitivity_analysis():
    with open(SENSITIVITY_PATH, "r") as f:
        return json.load(f)
ml_df = load_data()
model = load_model()
sensitivity = load_sensitivity_analysis()
# Add engagement tiers (same logic as notebook)
ml_df["playtime_tier"] = pd.qcut(
    ml_df["total_playtime_value"],
    q=5,
    labels=["Very Low", "Low", "Medium", "High", "Very High"]
)
# -----------------------
# Sidebar navigation
# -----------------------
section = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Engagement Tiers",
        "Churn & Model Insights",
        "Churn Definition Comparison (v1 vs v2)",
        "User-Level Exploration",
    ],
)
# =====================================================
# 1) OVERVIEW
# =====================================================
if section == "Overview":
    st.header("📊 Engagement Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Users", len(ml_df))
    col2.metric("Total Games (unique)", int(ml_df["unique_games"].sum()))
    col3.metric(
        "Avg Playtime per User",
        f"{ml_df['total_playtime_value'].mean():.1f}",
    )
    st.subheader("Total Playtime Distribution (log scale)")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(ml_df["total_playtime_value"], bins=50, ax=ax)
    ax.set_xscale("log")
    ax.set_xlabel("Total Playtime (log scale)")
    st.pyplot(fig)
    st.markdown(
        """
**Key takeaway:**  
Engagement is extremely heavy-tailed. Most users exhibit very low playtime,
while a small fraction accounts for a large share of total engagement.
"""
    )
# =====================================================
# 2) ENGAGEMENT TIERS
# =====================================================
elif section == "Engagement Tiers":
    st.header("🧩 Engagement Tiers")
    st.markdown(
        """
Users are segmented into **quantile-based engagement tiers** using total playtime.
This ensures balanced groups despite extreme skew.
"""
    )
    tier_order = ["Very Low", "Low", "Medium", "High", "Very High"]
    st.subheader("User Distribution by Tier")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=ml_df, x="playtime_tier", order=tier_order, ax=ax)
    ax.set_xlabel("Engagement Tier")
    st.pyplot(fig)
    st.subheader("Engagement Depth by Tier")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            data=ml_df,
            x="playtime_tier",
            y="total_playtime_value",
            order=tier_order,
            ax=ax,
        )
        ax.set_yscale("log")
        ax.set_xlabel("Tier")
        ax.set_ylabel("Total Playtime (log)")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            data=ml_df,
            x="playtime_tier",
            y="unique_games",
            order=tier_order,
            ax=ax,
        )
        ax.set_yscale("log")
        ax.set_xlabel("Tier")
        ax.set_ylabel("Unique Games (log)")
        st.pyplot(fig)
# =====================================================
# 3) CHURN & MODEL INSIGHTS
# =====================================================
elif section == "Churn & Model Insights":
    st.header("🤖 Churn & Model Insights")
    st.markdown(
        """
**Proxy churn definition:**  
Users in the bottom 20% of total playtime are labeled as churned.
This definition introduces circularity, which is explicitly acknowledged here.
"""
    )
    st.subheader("Proxy Churn Rate by Engagement Tier")
    churn_rate = (
        ml_df.groupby("playtime_tier")["churned"]
        .mean()
        .reindex(["Very Low", "Low", "Medium", "High", "Very High"])
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    churn_rate.plot(kind="bar", ax=ax)
    ax.set_ylabel("Churn Rate (proxy)")
    st.pyplot(fig)
    st.subheader("Feature Importance")
    feature_names = [
        "total_playtime_value",
        "sessions",
        "unique_games",
        "avg_session_length",
        "playtime_per_game",
    ]
    importances = model.feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    fi.plot(kind="bar", ax=ax)
    ax.set_ylabel("Importance")
    st.pyplot(fig)
    st.markdown(
        """
**Interpretation:**  
Total playtime dominates importance because churn is defined using playtime.
Secondary features capture engagement intensity and breadth.
"""
    )
# =====================================================
# 4) CHURN DEFINITION COMPARISON (v1 vs v2)
# =====================================================
elif section == "Churn Definition Comparison (v1 vs v2)":
    st.header("🔀 Churn Definition Comparison: Playtime vs. Library Breadth")
    st.markdown(
        """
**v1 (playtime percentile):** bottom 20% of `total_playtime_value`. Circular with
its own top features (see "Churn & Model Insights").

**v2 (library breadth):** bottom 20% of `play_ratio` = `unique_games / owned_games`
-- how much of a user's purchased library they actually played. Built from
purchase-event data, independent of playtime depth. Models trained against v2
exclude `unique_games`, `owned_games`, and `play_ratio` from their features, since
those directly construct the label.

All numbers below are loaded from `results/sensitivity_analysis.json`, generated by
`src/sensitivity.py` -- nothing on this page is hand-typed.
"""
    )
    agreement = sensitivity["label_agreement"]
    col1, col2, col3 = st.columns(3)
    col1.metric("v1 prevalence", f"{agreement['prevalence_v1']:.1%}")
    col2.metric("v2 prevalence", f"{agreement['prevalence_v2']:.1%}")
    col3.metric("Cohen's kappa", f"{agreement['cohens_kappa']:.3f}")
    st.markdown(
        f"""
**{agreement['pct_label_changed']}%** of users ({agreement['n_disagree']:,} of
{agreement['n_users']:,}) get a *different* churn label depending on which
definition is used. Cohen's kappa of **{agreement['cohens_kappa']:.3f}** indicates
the two definitions agree essentially no better than chance -- they are capturing
different populations, not the same "at-risk" users measured two ways.
- {agreement['newly_churned_under_v2']:,} users are newly flagged as churned under v2
  (low playtime-and-breadth engagement).
- {agreement['no_longer_churned_under_v2']:,} users flagged as churned under v1 are
  *not* flagged under v2 (they played little in total, but played most of what they own).
"""
    )
    st.subheader("Model performance: v1 vs. v2 (same split, same model family)")
    perf_rows = []
    for key, label in [("v1", "v1 (playtime)"), ("v2", "v2 (library breadth)")]:
        m = sensitivity[key]["metrics"]
        d = sensitivity[key]["dummy_baseline_metrics"]
        perf_rows.append(
            {
                "definition": label,
                "accuracy": m["accuracy"],
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "dummy_roc_auc": d["roc_auc"],
            }
        )
    perf_df = pd.DataFrame(perf_rows).set_index("definition")
    st.dataframe(perf_df.style.format("{:.3f}"))
    st.markdown(
        """
v1's ~1.0 accuracy/ROC-AUC reflects label circularity, not predictive skill (see
"Churn & Model Insights"). v2's ROC-AUC is close to its DummyClassifier baseline --
playtime-depth features carry little to no signal about library-breadth churn, which
is consistent with the near-zero Cohen's kappa above: these are largely different
constructs.
"""
    )
    st.subheader("Feature importance: v1 vs. v2 (permutation, same method, shared features only)")
    ranking = sensitivity["common_feature_importance_ranking"]
    rank_df = pd.DataFrame(
        {
            "v1 rank": ranking["v1_rank"],
            "v2 rank": ranking["v2_rank"],
        }
    )
    rank_df.index = rank_df.index + 1
    st.dataframe(rank_df)
    st.caption(
        "Restricted to the 4 features both models share -- unique_games/owned_games/"
        "play_ratio are excluded from this comparison since only v2 (via its label) "
        "and only the v1 model (via its feature set) can see them, respectively."
    )
    st.subheader("Churn rate by playtime tier: v1 vs. v2")
    seg_df = pd.DataFrame(sensitivity["segment_comparison"]["segments"]).set_index("playtime_tier")
    seg_df = seg_df.reindex(["Very Low", "Low", "Medium", "High", "Very High"])
    fig, ax = plt.subplots(figsize=(8, 5))
    seg_df[["churn_rate_v1", "churn_rate_v2"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("Churn Rate")
    ax.legend(["v1 (playtime)", "v2 (library breadth)"])
    st.pyplot(fig)
    st.markdown(
        """
v1's churn rate is trivially 100% in the lowest playtime tier and 0% everywhere else
-- by construction, since both are built from `total_playtime_value`. v2's churn
rate is roughly flat (~20-25%) across every playtime tier: library-breadth
disengagement is not concentrated among low-playtime users, another sign it is
measuring something genuinely different from playtime depth.
"""
    )
    high_risk = sensitivity["segment_comparison"]
    st.markdown(
        f"""
**High-risk user overlap:** of the {high_risk['high_risk_v1_count']:,} users flagged
churned under v1, only {high_risk['high_risk_overlap_count']:,}
({high_risk['high_risk_overlap_pct_of_v1']}%) are also flagged under v2. A retention
team acting on v1 alone would target a largely different set of users than one acting
on v2 alone.
"""
    )
# =====================================================
# 5) USER-LEVEL EXPLORATION
# =====================================================
elif section == "User-Level Exploration":
    st.header("👤 User-Level Exploration")
    user_id = st.selectbox(
        "Select a user",
        ml_df["user_id"].unique(),
    )
    user_row = ml_df[ml_df["user_id"] == user_id].iloc[0]
    st.subheader("Engagement Profile")
    st.dataframe(
        user_row[
            [
                "total_playtime_value",
                "sessions",
                "unique_games",
                "avg_session_length",
                "playtime_per_game",
                "playtime_tier",
            ]
        ].to_frame("value")
    )
    X_user = user_row[
        [
            "total_playtime_value",
            "sessions",
            "unique_games",
            "avg_session_length",
            "playtime_per_game",
        ]
    ].values.reshape(1, -1)
    churn_prob = model.predict_proba(X_user)[0, 1]
    st.subheader("Proxy Churn Probability")
    st.metric("P(Churn)", f"{churn_prob:.2f}")
    st.markdown(
        """
⚠️ This probability reflects **engagement risk**, not time-based churn.
"""
    )