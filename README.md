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

## Known Limitations

Both pipelines have label-construction caveats that should be read before trusting any
reported metric or feature-importance ranking. See each subproject's README for detail
(`anime_simulated/README.md`, `steam_real/README.md`), and `*/results/baseline_metrics.json` for the
underlying generated numbers.

- **Steam's original churn definition (v1) is circular with its own top features.**
  `churned` is defined as the bottom 20th percentile of `total_playtime_value`, and
  that same column (plus two features directly derived from it) is also fed into the
  model. This is why the Steam v1 model reports ~1.0 accuracy/ROC-AUC — it is not a
  genuine predictive result. A second, independent definition (v2 — library breadth:
  played vs. owned games) has since been added and compared against v1 on the same
  user population; the two agree no better than chance (Cohen's kappa ≈ -0.03). See
  `steam_real/README.md` and `steam_real/results/sensitivity_analysis.json` for the
  full sensitivity analysis.
- **Anime's continuation label has a smaller, analogous leakage issue.** One feature,
  `anime_mean_p_continue`, is a direct aggregate of `p_continue`, the same probability
  used to draw the `label_next_episode` label during simulation. Unlike Steam's
  circularity, this feature does not dominate the model (~8% Gini / ~4% permutation
  importance, well behind `anime_num_watch_events` and `user_prev_episodes_this_anime`),
  but it is a labeled leakage signal, not a genuine behavioral driver.
- **The two pipelines' feature spaces are fully disjoint and not comparable.** Anime and
  Steam share no feature names, model on different populations (synthetic vs. real),
  and use different label-construction processes. Any similarity or difference in their
  feature-importance charts is not evidence of anything about "data availability" in
  isolation — see the next point.
- **The two pipelines use different importance methods, and `shared/app.py`'s
  side-by-side comparison should be read as illustrative, not a controlled comparison.**
  `shared/app.py` plots `.feature_importances_` (Gini/impurity) for both models. This is
  consistent for Steam, but inconsistent with anime's own dashboard and README, which
  otherwise present permutation importance as the model's real explanation method.
  The two models are also different algorithms (GradientBoosting vs. RandomForest), so
  even same-method Gini values wouldn't be directly comparable across them. Differences
  shown in that chart reflect a mix of data, label construction, model family, and
  importance-method effects — not "data availability" alone.
- **All metrics above come from a single train/validation split** unless otherwise
  noted. `shared/stability.py` (used by both pipelines) runs
  `RepeatedStratifiedKFold(n_splits=5, n_repeats=10)` (50 folds) for the anime model
  and both Steam churn models, reporting mean/std/95% intervals instead. Steam v1's
  std is **exactly 0.0000** across all 50 folds for every metric — independent
  confirmation the circularity is structural, not a lucky split, and should never be
  read as evidence of a good model. See `anime_simulated/README.md`,
  `steam_real/README.md`, and `*/results/stability_*.json` for full results.

## How to Run

```
pip install -r requirements.txt
streamlit run shared/app.py
```
