# Engagement Comparison Dashboard

This part of the project brings together two engagement pipelines built on very different types of data and places them side by side for interpretation.

One pipeline is based on simulated anime viewing behavior, where user activity is generated with explicit timestamps, episode progression, and clear continuation signals. The other is built on real Steam gameplay data, where engagement must be inferred from aggregate playtime and session statistics rather than explicit time-based events.

The goal here is not to retrain models or force a unified definition of churn, but to understand how engagement and retention behave when the available signals differ.

## What this dashboard shows

The dashboard focuses on three aspects of engagement analysis:

### Engagement structure

It visualizes how activity is distributed in each dataset. Anime engagement is expressed through watch time per episode, while Steam engagement appears as total playtime across games. Both show heavy-tailed behavior, but the real data exhibits sharper spikes and greater sparsity.

### Retention and churn framing

Retention in the anime pipeline is defined by whether a user continues to the next episode, using explicit time-based signals. This label has its own caveat: it's a direct stochastic draw from a per-episode continuation probability, and one model feature (`anime_mean_p_continue`) is a direct aggregate of that same probability (see `anime_simulated/README.md`).
In contrast, Steam churn is defined using a proxy based on engagement depth, where users in the bottom segment of total playtime are treated as churned due to missing timestamps — a more severe circularity, since the same column is also a top model feature (see `steam_real/README.md`).

### Model behavior

Pretrained models from each pipeline are inspected to see which features drive their predictions. The anime model distributes importance across multiple behavioral signals, while the Steam model is dominated by engagement depth metrics due to how churn is constructed.

**Read this chart as illustrative, not a controlled comparison.** Both bars are
`.feature_importances_` (Gini/impurity importance) for implementation simplicity, but
this is not fully apples-to-apples:

- The two models are different algorithms (GradientBoosting for anime, RandomForest
  for Steam), so raw Gini values aren't directly comparable even in principle.
- Anime's own README/dashboard elsewhere present *permutation* importance as that
  model's primary explanation method — this chart uses Gini for both instead, which is
  consistent with Steam's own dashboard but not with anime's.
- The two feature sets share no feature names and describe different populations
  (synthetic anime viewers vs. real Steam players), so there is no meaningful
  feature-level overlap to compare, only a high-level "how concentrated is importance"
  shape comparison.
- Steam's concentration is not organic behavioral signal — it's label circularity (see
  `steam_real/README.md` and `steam_real/results/baseline_metrics.json`). Anime has a
  smaller, secondary analogue (`anime_mean_p_continue`, ~8% Gini importance — see
  `anime_simulated/README.md`).

## Why this comparison matters

Both pipelines aim to reason about engagement and retention, but they operate under different data constraints. The dashboard highlights how those constraints shape feature engineering, churn definitions, and model interpretation — though, per the caveats above, some of what the "Model behavior" chart shows also reflects model-family and importance-method choices, not data availability alone.

This makes it easier to reason about what a model is actually learning, and why similar modeling approaches can behave very differently when applied to simulated versus real-world data.

## How to run

From the project root:

`streamlit run shared/app.py`

The dashboard loads preprocessed datasets and trained models from the individual pipelines and is intended for exploration and interpretation rather than experimentation.