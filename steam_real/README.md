# Steam Real Engagement Pipeline

Real gameplay engagement modeling, proxy churn analysis, and sim2real comparison foundation

## Overview

This project builds the real-data engagement and churn pipeline using public Steam gameplay datasets.
Unlike the anime module, which relies on fully simulated viewing logs, this project works with real-world gameplay data and the limitations that come with it.

The goal is to understand how players engage with games over time, how engagement depth varies across users, and how churn can be approximated when true session timestamps are unavailable.

Instead of predicting episode continuation, this module focuses on:

- user-level engagement depth
- proxy churn modeling
- validating an end-to-end analytics and ML pipeline under real data constraints

This project sits inside the larger **sim2real-user-engagement** repo as the **real-data half**, designed to be compared directly with the simulated anime pipeline.

## Data

**Source:** Public Steam datasets (e.g. `steam-200k` and related metadata files)

**Important fields include:**

- User identifier
- Game identifier or name
- Total playtime (often in minutes)
- Implicit play events
- Game metadata (tags, genres, descriptions)

**Important limitation**

The dataset does not contain true timestamps or session boundaries.

As a result:

- time-based retention curves cannot be computed
- most user–game pairs have only one recorded event
- engagement must be analyzed through aggregated user-level features

All raw files live in:
`steam_real/data/raw/` and `steam_real/data/external/`

All processed files live in:
`steam_real/data/processed/`

## Data Pipeline

The pipeline mirrors the structure of the simulated project, adapted for real gameplay logs.

### 1. Ingestion pipeline

Loads and cleans raw Steam gameplay data:

- standardizes column names
- removes corrupted rows
- aggregates play events at the user–game level
- repares clean intermediate tables

### 2. Feature pipeline

Builds user-level engagement features, including:

- total playtime
- number of unique games played
- total session count (proxy)
- average session length
- playtime per game

Output: `ml_dataset.parquet`

These features represent engagement depth, which is the strongest signal available in the dataset.

### 3. Proxy churn definition

Because timestamps are unavailable, churn is defined as:

- Users in the bottom 20 percent of total playtime.

This is a proxy churn label, not true behavioral churn.

Its purpose is to:

- enable modeling
- validate the ML pipeline
- highlight real-world data limitations

### 4. Modeling pipeline

Trains a churn classifier using engagement features:

- tree-based model
- validation metrics (accuracy, precision, recall, ROC AUC)
- feature importance analysis

The trained model is saved to:
`models/churn_model.pkl`

Due to the churn definition, near-perfect performance is expected and explicitly discussed as label circularity.

## Notebooks
### 1. explore_raw_logs.ipynb

- EDA on raw Steam gameplay data
- playtime distribution analysis
- session sparsity diagnosis
- justification for user-level aggregation

### 2. retention_and_sessions.ipynb

- engagement tier construction using playtime quantiles
- treatment of engagement depth as a proxy for retention
- churn distribution across tiers
- explicit discussion of dataset limitations

### 3. modeling.ipynb

- evaluation of proxy churn model
- feature importance interpretation
- example user-level predictions
- analysis of circularity and modeling implications

## Proxy Engagement Model

The churn model uses:

- total playtime
- engagement intensity (playtime per game)
- average session length
- aggregate session counts
- game diversity

The model predicts: `P(user belongs to lowest engagement segment)`

Because churn is derived from playtime, performance reflects pipeline correctness, not real-world predictive accuracy.

## Engagement and Churn Analysis

The project includes:

- heavy-tailed engagement analysis
- engagement tier segmentation
- monotonic relationship between depth and churn
- feature importance diagnostics
- interpretable example predictions

These analyses mirror how real analytics teams reason about engagement when ideal data is unavailable.

## Streamlit Dashboard

This project includes a completed Streamlit dashboard for exploring real gameplay engagement and proxy churn in an interactive way.

The dashboard is designed to mirror how analytics and data science teams expose engagement insights to non-technical stakeholders, while remaining faithful to the limitations of the underlying data.

The app lives in:
`steam_real/app/app.py`

## Dashboard Views
### Global Engagement Overview

High-level views summarizing player engagement across the platform:

- distribution of total playtime
- engagement tier breakdown (Very Low → Very High)
- proportion of users in proxy churn vs non-churn groups
- heavy-tailed engagement patterns

These views provide immediate intuition about how engagement concentrates among a small subset of players.

### Engagement Tiers & Retention Proxy

Interactive exploration of engagement depth:

- users segmented into quantile-based engagement tiers
- proxy churn concentration in the lowest engagement tier
- comparisons of engagement intensity across tiers

This section makes explicit that churn is defined as an engagement-based proxy, not inactivity over time.

### Feature-Level Insights

Visualizations derived from the modeling pipeline:

- feature importance from the churn model
- comparison of playtime, session length, and game diversity
- how engagement depth translates into churn probability

This allows users to understand why the model behaves the way it does, rather than treating it as a black box.

### User-Level Exploration

Interactive inspection of individual users:

- engagement feature profiles
- predicted churn probability
- comparison against population averages

These views demonstrate how the model maps engagement depth to churn risk under the proxy definition.

## Sim2Real Context

The dashboard is intentionally designed to support sim2real comparison.

It highlights:

- how real gameplay engagement differs from simulated viewing behavior
- where time-based retention modeling breaks down in real datasets
- which engagement signals transfer cleanly between simulated and real domains

This positions the dashboard as both an analysis tool and a teaching artifact for sim2real modeling.

## Running the Dashboard

From the steam_real directory:
`streamlit run app/app.py`

The dashboard loads:

- processed engagement features
- trained churn model
- precomputed aggregates

No retraining is required to explore the data.

## Design Principles

The dashboard follows three core principles:

- **Honest visualization:**
All plots and metrics reflect dataset limitations and proxy definitions.
- **Interpretability first:**
Every model output is paired with feature-level explanation.
- **Consistency with the pipeline:**
Dashboard outputs match notebook results exactly.

## Summary

The Streamlit dashboard completes the real-data half of the sim2real engagement system by:

- exposing engagement patterns interactively
- visualizing proxy retention and churn
- interpreting model behavior transparently
- enabling direct comparison with simulated pipelines

It serves as the final layer connecting raw gameplay logs, modeling pipelines, and sim2real analysis into a single coherent system.