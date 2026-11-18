# Overview

# Setup

# Data location

# How to reproduce experiments

From `pathsnap/`, run the real-data LightGBM tuning after candidate and label
tables are available:

```powershell
python -m src.models.tune --n-trials 20 --n-splits 3
```

Optuna minimizes mean candidate-row log loss across `GroupKFold` splits grouped
by `trip_id`. It only fits on training trips; student-test trips are used once
for the final tuned-versus-baseline comparison. The study database and trial
log are written under `results/optuna/`, the comparison is
`results/optuna/tuning_comparison.csv`, and the tuned bundle/config are written
under `artifacts/`.

# How to run the API

# How to run the UI

# Folder tour

# AI-assistance disclosure
