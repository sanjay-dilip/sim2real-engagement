import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
sns.set(style="whitegrid")
# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Sim2Real Engagement Comparison",
    layout="wide",
)
st.title("🔁 Engagement Comparison: Simulation vs Real Data")
st.markdown(
    """
This dashboard compares user engagement and churn-related behavior across two pipelines:

- A simulated anime viewing environment with event-level timestamps and explicit continuation labels
- A real-world Steam gameplay dataset where churn must be inferred from engagement depth

The purpose is to study how engagement structure and churn framing differ between simulated and real data,
and how these differences influence downstream model behavior.
"""
)
# -------------------------
# Paths
# -------------------------
ROOT = Path(__file__).resolve().parents[1]
ANIME_DATA = ROOT / "anime_simulated" / "data" / "processed" / "ml_dataset.parquet"
STEAM_DATA = ROOT / "steam_real" / "data" / "processed" / "ml_dataset.parquet"
ANIME_MODEL = ROOT / "anime_simulated" / "models" / "next_episode_model.pkl"
STEAM_MODEL = ROOT / "steam_real" / "models" / "churn_model.pkl"
# -------------------------
# Loaders
# -------------------------
@st.cache_data
def load_df(path):
    return pd.read_parquet(path)
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)["model"]
anime_df = load_df(ANIME_DATA)
steam_df = load_df(STEAM_DATA)
anime_retention_model = load_model(ANIME_MODEL)
steam_model = load_model(STEAM_MODEL)
# Add tiers for Steam
steam_df["playtime_tier"] = pd.qcut(
    steam_df["total_playtime_value"],
    q=5,
    labels=["Very Low", "Low", "Medium", "High", "Very High"]
)
# -------------------------
# Sidebar navigation
# -------------------------
section = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Engagement Structure",
        "Retention & Churn Framing",
        "Model Behavior",
    ],
)
# =====================================================
# OVERVIEW
# =====================================================
if section == "Overview":
    st.header("Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Simulated Anime Data")
        st.markdown(
            """
- Synthetic viewing logs  
- Explicit session timestamps  
- Time-based churn definition  
- Probabilistic model behavior
"""
        )
    with col2:
        st.subheader("Real Steam Data")
        st.markdown(
            """
- Real gameplay records  
- No timestamps or session boundaries  
- Engagement-depth proxy churn  
- Threshold-like model behavior
"""
        )
    st.markdown(
        """
**Takeaway:**  
Both pipelines study engagement and retention, but the signals available in each dataset lead to fundamentally different modeling choices and interpretations.
"""
    )
# =====================================================
# ENGAGEMENT STRUCTURE
# =====================================================
elif section == "Engagement Structure":
    st.header("Engagement Structure")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Anime Engagement Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(anime_df["watch_time_sec"], bins=50, ax=ax)
        ax.set_xscale("log")
        ax.set_xlabel("Watch Time (seconds, log scale)")
        st.pyplot(fig)
    with col2:
        st.subheader("Steam Engagement Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(steam_df["total_playtime_value"], bins=50, ax=ax)
        ax.set_xscale("log")
        ax.set_xlabel("Total Playtime (seconds, log scale)")
        st.pyplot(fig)
    st.markdown(
        """
**Observation:**  
Both environments show heavy-tailed engagement distributions.

- Anime engagement is smoother and more continuous because watch time is recorded per episode with timestamps.
- Steam engagement is more uneven, with sharp spikes and sparsity caused by aggregation over long time windows.

These differences reflect how engagement is *measured*, not necessarily how users behave.
"""
    )
# =====================================================
# RETENTION & CHURN
# =====================================================
elif section == "Retention & Churn Framing":
    st.header("Retention and Churn Framing")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Anime: Time-Based Retention")
        retention_rate = anime_df["label_next_episode"].mean()
        st.metric("Retention Rate", f"{retention_rate:.2f}")
        st.markdown(
            """
Retention is defined as whether a user continues to the next episode. This is a time-based behavioral signal.
"""
        )
    with col2:
        st.subheader("Steam: Engagement-Based Proxy Churn")
        churn_rate = steam_df["churned"].mean()
        st.metric("Proxy Churn Rate", f"{churn_rate:.2f}")
        st.markdown(
            """
Churn is defined as the bottom 20% of users by total playtime.  
This proxy is required because session timestamps and inactivity gaps are not available.
As a result, Steam churn behaves more like a threshold on engagement depth rather than a temporal exit.
"""
        )
    st.subheader("Proxy Churn Concentration (Steam)")
    churn_by_tier = (
        steam_df.groupby("playtime_tier")["churned"]
        .mean()
        .reindex(["Very Low", "Low", "Medium", "High", "Very High"])
    )
    st.markdown(
"""
**Interpretation:**  
All proxy churn mass concentrates in the *Very Low* engagement tier by construction.
This is expected behavior and confirms that the proxy churn definition is being applied correctly.
""")
    fig, ax = plt.subplots(figsize=(8, 4))
    churn_by_tier.plot(kind="bar", ax=ax)
    ax.set_ylabel("Churn Rate")
    st.pyplot(fig)
# =====================================================
# MODEL BEHAVIOR
# =====================================================
elif section == "Model Behavior":
    st.header("Model Behavior")
    anime_label = "label_next_episode"
    anime_non_features = ["user_id", "anime_row_id", "anime_title", "watch_start_time", anime_label,]
    anime_features = [col for col in anime_df.columns if col not in anime_non_features]
    steam_features = [
        "total_playtime_value",
        "sessions",
        "unique_games",
        "avg_session_length",
        "playtime_per_game",
    ]
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Anime Model Feature Importance")
        fi_anime = pd.Series(
            anime_retention_model.feature_importances_,
            index=anime_features,
        ).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        fi_anime.plot(kind="bar", ax=ax)
        st.pyplot(fig)
    with col2:
        st.subheader("Steam Model Feature Importance")
        fi_steam = pd.Series(
            steam_model.feature_importances_,
            index=steam_features,
        ).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        fi_steam.plot(kind="bar", ax=ax)
        st.pyplot(fig)
    st.markdown("""
**Interpretation:**  
The anime model distributes importance across multiple behavioral signals, reflecting richer temporal structure.

The Steam model is dominated by total playtime and related aggregates, which is expected given that churn is defined directly from engagement depth.

These differences highlight how model behavior is shaped by data availability rather than algorithm choice."""
    )