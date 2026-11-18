# Overview

PathSnap map-matches noisy GPS observations to directed road-network arcs.
The pipeline creates nearby candidates, scores each candidate, then uses
sequence decoding to select a route-consistent prediction.

# Setup

Create a virtual environment and install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run commands below from the `pathsnap/` directory unless noted otherwise.

# Data location

Raw trip data belongs under `data/` and is not committed. The evaluation map
command expects a data root containing `train/` and `student_test/`. Its
default is `data/Student folder/Student folder`; pass `--data-root` if your
local copy uses a different layout.

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

To compare the saved classifier with HMM/Viterbi on all 10 student-test trips,
reuse the trained bundle without retraining:

```powershell
python -m src.hmm.ml_evaluation
```

The command writes `results/hmm_ml_evaluation_report.json`,
`results/hmm_ml_predictions_student.parquet`, and one
`results/maps/hmm_trip_<trip_id>.html` map per student-test trip. It refuses to
replace existing outputs unless `--overwrite` is supplied. The `dp_trip_*.html`
maps are separate and are not changed by this command. If the raw track, nodes,
or route file for a trip is missing, its map is skipped and the missing inputs
are recorded in the JSON report; metrics and predictions are still produced
for all trips. A missing raw `.arcs` file can be read from the existing
`road_geometry` NPZ without modifying preprocessing outputs.

The report includes pointwise accuracy/precision/recall, unique-arc set
precision/recall, route continuity, edit distance, longest correct run,
candidate coverage, and per-trip plus aggregate results. Pointwise scores use
observations with non-null aligned labels; null predictions count as
incorrect. If no transition is possible, Viterbi now ends the current path and
restarts at that observation rather than nulling the whole trip segment.
Review the report's metric definitions before comparing it with an external
evaluator's score.

# How to run the API

# How to run the UI

# Folder tour

# AI-assistance disclosure
