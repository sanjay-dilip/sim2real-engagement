# 🎮 Anime Simulated Engagement Pipeline

Full synthetic OTT-style retention modeling, continuation prediction, and dashboard

## 📌 Overview

This project builds a full engagement and retention pipeline using synthetic anime viewing logs.
The goal is to understand how people interact with episodic content and how to predict if a user will continue watching the next episode.

Instead of collecting real logs, this project creates a complete simulation on top of a real anime metadata file.
It learns what retention looks like, how to model continuation, and how to build dashboards that mirror what streaming platforms use.

The system includes:
- a metadata pipeline
- a viewing log simulator
- retention and cohort analysis
- a feature pipeline for ML
- a classifier to predict continuation
- a Streamlit dashboard
- an end-to-end runner

This project sits inside the larger **sim2real-user-engagement** repo as the simulated half.

## 🗂️ Data

**Source:** `anime-offline-database.json`

**Important fields:**

- Anime title
- Episodes
- Genres
- Popularity measures
- Related shows
- Tags and categories

The project builds two processed tables:

- anime_master.parquet
- episodes.parquet

Then it generates viewing logs with:

- user id
- anime id
- episode number
- watch start time
- watch time
- completion percent
- engagement level
- label: did the user watch the next episode or not

All clean files live in:
`anime_simulated/data/processed/`

## 🔧 Data Pipeline

The pipeline uses four steps:

### 1. Metadata pipeline

Loads the raw JSON and creates:

- a structured anime table
- an episode-level table
- cleaned fields
- consistent ids

### 2. Simulation pipeline

Creates synthetic viewing behavior:
- watch times based on completion
- engagement levels per user
- drop-off curves
- continuation vs churn labels
- different behavior for different anime

This produces a full synthetic event log.

### 3. Feature pipeline

Builds ML features from the logs:

- episode context
- user history (previous episodes, previous continuation rate)
- anime popularity stats
- watch completion stats

**Output:** `ml_dataset.parquet`

### 4. Modeling pipeline

Trains a next-episode classifier using:

- Gradient Boosting
- ROC AUC, precision, recall
- Feature importance (permutation)

The trained model is saved to:
`models/next_episode_model.pkl`

Everything is reproducible through the Python entrypoint.

## 🔎 Notebooks
### 1. explore_metadata.ipynb

- EDA on anime metadata
- episode counts
- genre exploration
- title consistency checks

### 2. simulate_viewing_logs.ipynb

- Prototype of the simulation logic
- inspection of continuation rates
- random user engagement patterns

### 3. retention_and_cohorts.ipynb

- global retention curves
- per-anime curves
- binge depth stats
- cohort survival plots

### 4. modeling.ipynb

- visual evaluation of the ML model
- ROC curve
- precision/recall
- feature importance
- performance by episode number

## 🤖 Simulated Engagement Model

The continuation model uses:

- user features (history, past continuation)
- anime features (popularity, average continuation)
- episode features
- watch behavior

The model predicts: `P(user continues to next episode)`

**Performance on synthetic validation:**

- Accuracy: ~0.81
- Recall: ~0.95
- ROC AUC: ~0.85

The scores make sense because the simulator has structure that the model can learn.

## 🧪 Cohort and Retention Analysis

The project includes:

- retention by episode number
- retention for the most watched anime
- user binge depth distribution
- cohort survival by first watch day

These are the same plots used in streaming product teams.

## 🌐 Streamlit Dashboard

The Streamlit app displays:

### Global views

- global retention curve
- binge depth distribution
- overall engagement patterns

### Anime views

- retention curve for a chosen anime
- episode depth distribution
- quick stats for that title

### Comparison

- compare top anime by views
- overlay retention curves

### Cohorts

- pick cohorts and see survival curves

### ML explainer

- metrics
- score distribution
- permutation feature importance

### Predictions

- pick an anime and episode
- see predicted continuation distribution
- explore sample episodes

Run with: `streamlit run anime_simulated/app/app.py`

## 🧩 Architecture Overview

```
                Anime Metadata (JSON)
                         |
                         v
               Metadata Cleaning Pipeline
                         |
                         v
                Synthetic Viewing Logs
       - user sessions
       - drop-off
       - continuation
       - engagement
                         |
                         v
                Feature Engineering
                         |
                         v
                Continuation Classifier
            - Gradient Boosting
            - ROC/AUC, precision
                         |
                         v
                  Streamlit Dashboard
```

## ▶️ How to Run Everything

From the project root: `python -m anime_simulated.run_anime_pipeline`

This runs:
- metadata
- simulation
- features
- model training

Outputs land in `data/processed/` and `models/`.

## Summary

This project builds a complete synthetic engagement system:

- OTT-style viewing logs
- drop-off and continuation behavior
- cohort and retention analysis
- ML prediction pipeline
- Streamlit dashboard

It serves as the simulated half of a sim to real workflow before moving to actual gameplay or streaming logs in the `steam_real` module.