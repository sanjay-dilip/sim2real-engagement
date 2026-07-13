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

The trained model is saved to:
`models/next_episode_model.pkl`

Everything is reproducible through the Python entrypoint.

**Note:** `run_anime_pipeline.py` itself does not compute feature importance —
it only trains, evaluates, and saves the model. Feature importance (both Gini
and permutation) is generated separately by `src/export_baseline.py`, written
to `results/baseline_metrics.json`, and also computed live in the Streamlit
dashboard's "Model explainer" page. See the leakage caveat below before
trusting any single feature's importance rank.

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
- feature importance (ad hoc, not persisted by this notebook — see
  `results/baseline_metrics.json` for the reproducible version)
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

**Label-construction caveat:** one feature, `anime_mean_p_continue`, is a direct
aggregate of `p_continue` — the same probability the simulator uses to draw
`label_next_episode` in the first place (`src/simulation_pipeline.py`). This is
a real, if secondary, form of label leakage: in `results/baseline_metrics.json`
it accounts for ~8% of Gini importance and ~4% of permutation importance,
noticeably behind `anime_num_watch_events` and `user_prev_episodes_this_anime`,
so it does not dominate the model the way Steam's playtime-derived features
dominate the Steam churn model (see `steam_real/README.md`), but it should be
read as a labeled leakage signal rather than a genuine behavioral driver when
interpreting feature-importance rankings.

**Stability and uncertainty:** the numbers above come from one 80/20 split.
`src/stability.py` runs `RepeatedStratifiedKFold(n_splits=5, n_repeats=10)` (50
folds) instead, reporting mean/std/95% interval per metric and feature-importance
rank stability. Results: `results/stability_anime.json`.

| metric | mean | std | 95% interval |
|---|---|---|---|
| accuracy | 0.8136 | 0.0030 | [0.8077, 0.8189] |
| recall | 0.9576 | 0.0024 | [0.9524, 0.9615] |
| roc_auc | 0.8569 | 0.0039 | [0.8505, 0.8640] |

These closely match the single-split numbers above with tight intervals — the
original split was not an outlier. Feature-importance stability across the 50
folds also sharpens the leakage-caveat picture: `anime_num_watch_events` and
`anime_num_users` are perfectly stable as the top-2 features (top-3 frequency
1.00, rank std 0.000 for both), while `anime_mean_p_continue` — the leakage
feature — never once appears in the top-3 across all 50 folds, holding a
perfectly stable 6th-place rank (out of 10 features, rank std 0.000). This is
further, independent evidence that the leakage is real but consistently
secondary, not an artifact of the one split reported above.

## ✅ Testing

`tests/test_anime_labels.py` covers this pipeline directly: the base
retention-probability curve, simulation reproducibility and seed-sensitivity,
and that `add_anime_level_features`/`add_user_level_features` aggregate
correctly and use only past events. `tests/test_split_reproducibility.py` and
`tests/test_stability.py` (shared) cover the train/validation split and the
repeated-CV utility behind the stability table above. Run with `pytest` from
the repo root.

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