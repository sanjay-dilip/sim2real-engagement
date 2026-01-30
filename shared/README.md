# Engagement Comparison Dashboard

This part of the project brings together two engagement pipelines built on very different types of data and places them side by side for interpretation.

One pipeline is based on simulated anime viewing behavior, where user activity is generated with explicit timestamps, episode progression, and clear continuation signals. The other is built on real Steam gameplay data, where engagement must be inferred from aggregate playtime and session statistics rather than explicit time-based events.

The goal here is not to retrain models or force a unified definition of churn, but to understand how engagement and retention behave when the available signals differ.

## What this dashboard shows

The dashboard focuses on three aspects of engagement analysis:

### Engagement structure

It visualizes how activity is distributed in each dataset. Anime engagement is expressed through watch time per episode, while Steam engagement appears as total playtime across games. Both show heavy-tailed behavior, but the real data exhibits sharper spikes and greater sparsity.

### Retention and churn framing

Retention in the anime pipeline is defined by whether a user continues to the next episode, using explicit time-based signals.
In contrast, Steam churn is defined using a proxy based on engagement depth, where users in the bottom segment of total playtime are treated as churned due to missing timestamps.

### Model behavior

Pretrained models from each pipeline are inspected to see which features drive their predictions. The anime model distributes importance across multiple behavioral signals, while the Steam model is dominated by engagement depth metrics due to how churn is constructed.

## Why this comparison matters

Both pipelines aim to reason about engagement and retention, but they operate under different data constraints. The dashboard highlights how those constraints shape feature engineering, churn definitions, and model interpretation.

This makes it easier to reason about what a model is actually learning, and why similar modeling approaches can behave very differently when applied to simulated versus real-world data.

## How to run

From the project root:

`streamlit run shared/app.py`

The dashboard loads preprocessed datasets and trained models from the individual pipelines and is intended for exploration and interpretation rather than experimentation.