# Sim2Real User Engagement Analysis

This project explores how user engagement and churn modeling differ between simulated data and real-world data, using two parallel pipelines built on very different data realities.

The goal is not to build the most accurate model, but to understand how modeling assumptions, feature importance, and churn definitions shift when moving from simulation to real data.

## Project Overview

User engagement is often studied using simulated or synthetic data, where clean timestamps, session boundaries, and clear behavioral signals are available. Real-world data, however, is messy, incomplete, and frequently requires proxy definitions for core concepts like retention and churn.

This project compares:

- A simulated anime viewing pipeline with explicit episode progression and time-based retention
- A real Steam gameplay pipeline where churn must be inferred from engagement depth rather than timestamps
- A shared analysis layer brings both pipelines together to highlight structural differences in engagement signals and model behavior.

## Repository Structure

```
sim2real-engagement/
│
├── anime_simulated/
│   ├── notebooks/
│   ├── models/
│   ├── data/
│   └── README.md
│
├── steam_real/
│   ├── notebooks/
│   ├── models/
│   ├── data/
│   └── README.md
│
├── shared/
│   ├── app.py
│   └── README.md
│
└── requirements.txt
```

Each subdirectory contains its own README explaining the data, assumptions, and modeling choices used in that pipeline.

## Pipelines
### Simulated Anime Engagement

- Synthetic episode-level viewing logs
- Explicit session timestamps and watch duration
- Retention defined as continuation to the next episode
- Probabilistic, multi-signal model behavior

This pipeline represents a controlled environment where engagement signals are clean and well-defined.

### Real Steam Gameplay Engagement

- Aggregated gameplay statistics per user
- No session timestamps or episode-like structure
- Churn defined as a proxy using low engagement thresholds
- Model behavior dominated by engagement depth

This pipeline reflects the constraints of real-world data where churn must be inferred rather than observed.

### Shared Comparison Dashboard

The shared/ directory contains a Streamlit dashboard that places both pipelines side by side to compare:

- Engagement distributions
- Retention and churn framing
- Feature importance and model behavior

The dashboard is designed for interpretation, not retraining, and focuses on how data structure influences modeling outcomes.

## Key Takeaway

Both pipelines aim to model engagement and churn, but they operate under fundamentally different data constraints.
The project highlights how modeling decisions are shaped as much by data availability as by algorithm choice.

## How to Run

```
pip install -r requirements.txt
streamlit run shared/app.py
```
