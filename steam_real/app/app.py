import streamlit as st
import pandas as pd
import numpy as np
import pickle
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
ml_df = load_data()
model = load_model()
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
# 4) USER-LEVEL EXPLORATION
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