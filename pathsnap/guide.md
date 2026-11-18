## PathSnap: Who Owns Which Files

*Notion-ready. Each person works only in their own folders and files. If you need something changed in someone else’s file, ask them.*

---

### 1. Ownership map (quick view)

| Folder / file | Owner | Others may |
| --- | --- | --- |
| `src/core/` (`io.py`, `geo.py`, `graph.py`) | **A** | import only |
| `src/features/candidates.py` | **A** | import only |
| `tools/build_candidate_table.py` | **A** | run |
| `splits.json`, `evaluator_trips.json` | **A** (create once) | read only, never edit |
| `src/labels/assign_labels.py` | **C** | import only |
| `src/hmm/` (`viterbi.py`, `transitions.py`, `metrics.py`) | **C** | import only |
| `src/viz/make_map.py` | **C** | import only |
| `src/models/` (`evaluate.py`, `cv.py`, `train_all.py`, `tune.py`) | **B** | import only |
| `tools/make_fake_table.py` | **B** | run |
| `artifacts/` (model bundle, `config.json`) | **B** | load only |
| `src/api/` (`main.py`, `pipeline.py`, `evaluation.py`) | **D** | run |
| `ui/app.py` | **D** | run |
| `notebooks/01_eda.ipynb` | **D** | read |
| `notebooks/02_experiments.ipynb` | **B** | read |
| `README.md`, `requirements.txt`, `.gitignore` | **D** | suggest changes |
| `src/config.py`, `CONTRACTS.md` | **Shared**: tell everyone before changing |  |
| `reports/` | **Each person writes their own file** |  |
| `results/plots`, `results/maps`, `results/optuna` | Whoever generates them |  |

---

### 2. Person A: Data, Candidates, Features

**Branch:** `feat/a-data`

| File | What to write | Needed by |
| --- | --- | --- |
| `src/core/io.py` | `Trip` dataclass and `load_trip(trip_id, with_route=False)`. Reads `.track` (lon, lat, t), `.nodes` (lon, lat), `.arcs` (u, v), `.route` (only if asked) | Everyone, **first priority (~10:30)** |
| `src/core/geo.py` | `to_xy(lon, lat)` using `pyproj` and the CRS that fits the data location | C, D, A |
| `src/core/graph.py` | `RoadGraph`: arc geometry (shapely lines in metres), `successors`, `hops`, `network_dist` | C, D |
| `src/features/candidates.py` | `build_candidates(trip, graph)` (STRtree search, radius 75 m, at most 10 arcs, features v1) and `FEATURE_COLS` | B, C, D |
| `tools/build_candidate_table.py` | Loops over train and student_test IDs from `splits.json`, saves `results/candidates_train.parquet` and `results/candidates_student.parquet` | B, C |
| `splits.json`, `evaluator_trips.json` | Generate once with seed 42 (80/10/10) | Everyone |
| `reports/section_2_features.md` | Candidate generation, miss-rate table, each feature with its reason | Report |

**Also:** after C pushes labels, compute the candidate miss-rate table for several radius and K settings and save it to `results/miss_rate.csv`.

**Don’t:** use `arc_id` as a feature, or read `.route` inside `build_candidates`.

---

### 3. Person B: Models and Training

**Branch:** `feat/b-models`

| File | What to write | Needed by |
| --- | --- | --- |
| `tools/make_fake_table.py` | Fake candidate table with the contract columns, to debug before real data arrives | B |
| `src/models/evaluate.py` | Macro F1, precision, recall, log loss, observation-level top-1, miss rate, confusion matrix | B, C |
| `src/models/cv.py` | `GroupKFold` by `trip_id`, pipelines, logging each run to `results/experiments.csv` | B |
| `src/models/train_all.py` | Defines and trains the 5 models (LogReg, GaussianNB, KNN subsampled, RF, LightGBM), builds the comparison table and plots | B |
| `src/models/tune.py` | Optuna (TPE) on the winning model, after 11:30 | B |
| `artifacts/model_bundle.joblib`, `artifacts/config.json` | Final model, scaler, feature list and order; `load_bundle` and `predict_proba(cands_df)` | D, C |
| `notebooks/02_experiments.ipynb` | Comparison table and plots | Report |
| `results/experiments.csv`, `results/plots/`, `results/optuna/` | Outputs of the above | Report |
| `reports/section_3_models.md`, `reports/section_4_bayesian.md` | Per-model answers, comparison, tuning results | Report |

**Early hand-off:** publish a stub bundle (softmax of negative distance, same `predict_proba` interface) so C and D can build against it.

**Inputs you wait for:** `results/candidates_*.parquet` (A) and `results/labels_*.parquet` (C). Merge them as `label = (arc_id == true_arc_id)`.

**Don’t:** use plain KFold, fit scalers outside the pipeline, or touch evaluator trips.

---

### 4. Person C: Labels, HMM, Maps

**Branch:** `feat/c-hmm`

| File | What to write | Needed by |
| --- | --- | --- |
| `src/labels/assign_labels.py` | `assign_labels(trip, graph)`: monotonic DP aligning GPS points to `.route` arcs. Returns `trip_id, obs_idx, true_arc_id` (null if far) | **B, blocked on this**; target ~10:50 |
| `tools/` (or a line in `assign_labels.py`) | Script that saves `results/labels_train.parquet` and `results/labels_student.parquet` | B, A |
| `src/hmm/metrics.py` | `route_continuity`, `sequence_metrics` (arc-set precision/recall, longest correct run, edit distance). **Share `route_continuity` with B early** | B |
| `src/hmm/transitions.py` | Transition rules: same arc, successor, 2–3 hops, epsilon for unreachable | C |
| `src/hmm/viterbi.py` | `decode(cands, probs, graph, params)` in log space, handling empty candidates, gaps, invalid transitions | D |
| `src/viz/make_map.py` | `make_map(trip_id, predicted=None, out_html=...)`: GPS, network, reference, predicted layers; legend; layer control | D, report |
| `results/maps/` | Folium HTML outputs | Report |
| `reports/section_5_hmm.md`, `reports/section_6_folium.md` | HMM answers, classifier-only vs HMM table, maps | Report |

**Until real inputs exist:** test Viterbi on a toy graph with stub probabilities. The HMM is not scored at 11:30, so keep it a rough draft in the morning.

**Don’t:** read `.route` anywhere except label creation and evaluation, or tune decoder parameters using evaluator trips.

---

### 5. Person D: Repo, EDA, API, UI

**Branch:** `feat/d-api`

| File | What to write | Needed by |
| --- | --- | --- |
| Repo skeleton, `.gitignore`, `requirements.txt`, `README.md` skeleton | Create first, push to `main` | Everyone, **first 15 minutes** |
| `notebooks/01_eda.ipynb` | Inventory, NaNs, `dt` histogram, extent, network size and components, `.route` checks, plots, a Folium map | Evaluation 1, report |
| `reports/data_issues_log.md` | Table of issue / evidence / decision | A, report |
| `reports/section_2_eda.md` | Preprocessing and exploration write-up | Report |
| `src/api/pipeline.py` | `run_pipeline(trip_id)`: `load_trip(with_route=False)` → `build_candidates` → `bundle.predict_proba` → `decode`. **Never opens `.route`** | API |
| `src/api/evaluation.py` | `evaluate(trip_id, predicted)`: the only API file that reads `.route`, called after prediction | API |
| `src/api/main.py` | FastAPI: `GET /health`, `GET /trips`, `POST /predict`; load model and graph once at startup; request logging; CORS | UI |
| `ui/app.py` | Streamlit page: trip dropdown, Predict button, map, metrics, error messages; API URL from an environment variable | Demo |
| `reports/section_7_api_ui.md` | Screenshots and deployment steps | Report |
| Final PDF and README | Merge all sections, check the structure | Submission |

**Morning scope:** repo, EDA and a skeleton API with stubs. Real pipeline wiring happens after 11:30.

**Don’t:** let the UI or API response expose reference geometry, or hardcode any trip.

---

### 6. Shared files (change only after telling everyone)

- `src/config.py`: seed, `RAW_DIR`, `CAND_RADIUS_M`, `CAND_MAX_K`, `GAP_THRESHOLD_S`
- `CONTRACTS.md`: function signatures and table schemas
- `src/__init__.py` and the other `__init__.py` files: empty

### 7. Data flow between files

```
io.py / geo.py / graph.py (A)
        │
        ├──► candidates.py (A) ──► results/candidates_*.parquet ──┐
        │                                                         ├──► train_all.py (B) ──► artifacts/model_bundle (B)
        └──► assign_labels.py (C) ──► results/labels_*.parquet ───┘               │
                                                                                   ▼
                              viterbi.py (C) ◄──── predict_proba ◄────────────────┘
                                   │
                                   ▼
                  src/api/pipeline.py (D) ──► main.py (D) ──► ui/app.py (D)
                                   │
                  evaluation.py (D) reads .route AFTER prediction
```

### 8. Rules for working in the repo

- Work on your own branch, commit small and often with clear messages (history is reviewed), and merge to `main` through short PRs or frequent pulls.
- Pull before you start each work block, to get the latest `core/` and tables.
- Only edit files in your own row of the ownership map. Ask the owner for changes elsewhere.
- Never commit `data/`, `data_eval/`, `.env`, or evaluator IDs.
- Load trips only through the IDs in `splits.json`, never by scanning the data folder.

### 9. Each person’s first file to push

| Person | First push | Why it matters |
| --- | --- | --- |
| D | Repo skeleton | Everyone clones it |
| A | `src/core/io.py` + `geo.py` + `graph.py` | Everyone imports them |
| C | `src/labels/assign_labels.py` | B can’t train on real data without labels |
| B | `src/models/evaluate.py` + `cv.py` | Used by C and B for comparison |

## PathSnap: Complete Guide Until Evaluation 1 (11:30 AM)

*Notion-ready. You are Member 2 = Role B, but read everything once so you know what the others are doing. Section 5 is the only place where “who needs what from whom” is listed, so check it whenever you get confused.*

---

### 1. What we must have by 11:30

Evaluation 1 scores two things, 5 marks each:

| Scored | What we show |
| --- | --- |
| **Data preprocessing and feature engineering** | We understand the files, cleaned the data, converted to metres, built candidates, labels and features |
| **Model selection and training** | 5 models from different families, group-aware cross-validation, a comparison table, a justified best model |

The HMM, Optuna, API and UI are **not** scored at 11:30. They come at 3:00 PM, so this morning they are only rough prep.

---

### 2. The idea in plain English

1. A car drives on roads, but its GPS dots are noisy and wobble sideways.
2. Near each dot there are a few road segments. We call these **candidates**.
3. For each (dot, candidate) pair we compute numbers called **features**. Example: how far the dot is from that segment.
4. We know the true route from `.route`, so each pair gets a **label**: 1 if it is the true segment, 0 if not.
5. We train 5 different classifiers to predict the label from the features, and compare them.

---

### 3. The four data files

| File | What it contains |
| --- | --- |
| `.track` | GPS points: `lon lat timestamp`, no header, about 1 point per second |
| `.nodes` | Road intersections: `lon lat`. The row number is the node ID (starts at 0) |
| `.arcs` | Road segments: `from_node to_node`. The row number is the **arc_id** |
| `.route` | The reference answer: an ordered list of arc_ids the car drove |

Three things to remember:

- One GPS row does **not** match one road row.
- `.route` is a list of arcs, **not** one label per GPS point. Person C must work out which arc each GPS point belongs to.
- Verify the real folder layout and file names. Don’t assume.

---

### 4. The one table everything revolves around

Each row is one (GPS point, candidate arc) pair:

| trip_id | obs_idx | arc_id | f_dist | f_rank | f_dist_gap | f_proj | f_bearing_diff | f_speed | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t01 | 0 | 512 | 4.2 | 1 | 0.0 | 0.4 | 3 | 8.1 | 1 |
| t01 | 0 | 733 | 31.0 | 2 | 26.8 | 0.9 | 88 | 8.1 | 0 |
- **A** builds everything on the left: `trip_id, obs_idx, arc_id` and the `f_*` features.
- **C** builds the true arc per point (`true_arc_id`), which gives the `label`.
- **B (you)** merges them and trains on it: `label = (arc_id == true_arc_id)`.
- **D** later builds the same table, without a label, for the API.

---

### 5. Who gives what to whom (the only dependency list)

| # | From | To | What | File | Target time | Blocks? |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | D | everyone | Repo skeleton | GitHub repo | ~10:00 | Yes |
| 2 | A | everyone | `splits.json` | `splits.json` | ~10:00 | Yes |
| 3 | A | B, C, D | Data loader, projection, road graph | `src/core/` | ~10:30 | Yes |
| 4 | A | B | Candidate and feature table plus `FEATURE_COLS` | `results/candidates_*.parquet` | ~11:00 | Yes (B’s real training) |
| 5 | C | B, A | Labels (`trip_id, obs_idx, true_arc_id`) | `results/labels_*.parquet` | ~10:50 | Yes (B’s real training) |
| 6 | C | B | `route_continuity` function | `src/hmm/metrics.py` | whenever ready | No, only fills one table column |
| 7 | B | C | A stub model (softmax of negative distance) | in `src/models/` | ~10:45 | No, C can write a stub |
| 8 | D | A | Data issues log (affects cleaning) | `reports/data_issues_log.md` | ~11:00 | No |

**Reading this simply:**

- A is the root. Everyone needs A first.
- B is blocked by A (table) and C (labels), so B builds everything on a **fake table** until they arrive.
- C is blocked only by A’s loader, not by B.
- D is blocked by nobody this morning.

---

### 6. Before anyone codes (9:45 to 10:15, all four together)

- [ ]  **Look at the real data folder together.** How are files named? One folder per trip? Are `.nodes` and `.arcs` per trip or shared? Write the answer in `CONTRACTS.md`.
- [ ]  **Data placement:** copy the dataset into `data/` on every laptop. It is gitignored, never pushed. Evaluator trips go in a separate `data_eval/` if you received them, and nobody opens them.
- [ ]  **D** creates the public repo, the folders, `.gitignore`, `requirements.txt`, and adds everyone as collaborators.
- [ ]  **Everyone** creates a virtual environment and runs `pip install -r requirements.txt`. Check `python -c "import shapely, pyproj, lightgbm, optuna, folium, fastapi"`. Agree on one Python version.
- [ ]  **A** creates `splits.json` (80 train, 10 student_test, 10 evaluator_test, shuffled with seed 42). Evaluator IDs go in a separate `evaluator_trips.json`.
- [ ]  **Agree and write in `CONTRACTS.md`:**
    - Raw coordinates are `(lon, lat)`. Distances only in metres after projecting.
    - `load_trip(trip_id, with_route=False)` returns `gps` (lon, lat, t), `nodes`, `arcs`, `route`.
    - `build_candidates(trip, graph)` returns `trip_id, obs_idx, arc_id, f_*`, sorted by `(trip_id, obs_idx, arc_id)`, **no label**.
    - `assign_labels(trip, graph)` returns `trip_id, obs_idx, true_arc_id`.
    - Candidate settings: radius 75 m, at most 10 arcs per point. Seed 42.
    - Features v1: `f_dist, f_rank, f_dist_gap, f_proj, f_bearing_diff, f_speed`.

**Folder structure**

```
pathsnap/
├── README.md  CONTRACTS.md  requirements.txt  .gitignore  splits.json
├── data/  (gitignored)
├── src/
│   ├── config.py
│   ├── core/       (A) io.py, geo.py, graph.py
│   ├── features/   (A) candidates.py
│   ├── labels/     (C) assign_labels.py
│   ├── models/     (B) evaluate.py, cv.py, train_all.py, tune.py
│   ├── hmm/        (C) viterbi.py, transitions.py, metrics.py
│   ├── viz/        (C) make_map.py
│   └── api/        (D) main.py, pipeline.py, evaluation.py
├── ui/             (D) app.py
├── tools/          build_candidate_table.py (A), make_fake_table.py (B)
├── notebooks/      01_eda.ipynb (D), 02_experiments.ipynb (B)
├── results/        plots/ maps/ optuna/
├── artifacts/      model bundle (B)
└── reports/        one file per person
```

Each person edits only their own files. Everyone works on their own branch (`feat/a-data`, `feat/b-models`, `feat/c-hmm`, `feat/d-api`), commits small and often, and pulls `main` before each work block.

---

### 7. Role A: Data, candidates, features (the root)

**Files:** `src/core/io.py`, `geo.py`, `graph.py`, `src/features/candidates.py`, `tools/build_candidate_table.py`, `splits.json`

**By ~10:30: `core/` (rough but working, push immediately)**

- `load_trip`: read each file with `pd.read_csv(path, sep=r"\s+", header=None, names=[...])`.
    - `.track` → `lon, lat, t`; `.nodes` → `lon, lat`; `.arcs` → `u, v`; `.route` only when `with_route=True`.
- `geo.to_xy`: convert (lon, lat) to metres with `pyproj` using the UTM zone or national grid that fits the data. Check the coordinates first. Note your choice and reason for the report.
- `RoadGraph`: each arc is a straight line from node `u` to node `v` in metres (shapely `LineString`). Add `successors(arc_id)`.

**By ~11:00: `build_candidates` with features v1**

- Build an `STRtree` over arc lines. For each GPS point take arcs within 75 m, keep the nearest 10. Count the points that have none.
- Features:

| Feature | Meaning | Why |
| --- | --- | --- |
| `f_dist` | Distance from point to arc (m) | Closer is more likely true |
| `f_rank` | Rank by distance among that point’s candidates | Relative closeness |
| `f_dist_gap` | `f_dist` minus the smallest `f_dist` for that point | Normalised per point |
| `f_proj` | Position of the projection along the arc (0 to 1) | Near an arc’s end, the next arc may be the truth |
| `f_bearing_diff` | Angle between car heading and arc direction | Separates parallel and opposite roads |
| `f_speed` | Speed from neighbouring points | Stops and gaps |
- First and last points lack neighbours: fill with a documented value (e.g. 0).
- **Never** use `arc_id` as a feature. **Never** read `.route` in this function (the API will call it too).
- `tools/build_candidate_table.py` loops over train and student IDs from `splits.json` and saves `results/candidates_train.parquet` and `candidates_student.parquet`.

**By ~11:20: miss-rate table.** After C’s labels arrive, compute the % of points where the true arc is not among the candidates, for radius 50/75/100 m and K 5/10/15. Save it to `results/miss_rate.csv`.

---

### 8. Role B (you): Models and training

**Files:** `tools/make_fake_table.py`, `src/models/evaluate.py`, `cv.py`, `train_all.py`, `notebooks/02_experiments.ipynb`, `results/experiments.csv`

**What you need:** A’s candidates (+ `FEATURE_COLS`) and C’s labels. Until they arrive, use a fake table.

**10:15 to 10:35: fake table.** `tools/make_fake_table.py` creates 30 fake trips with the same columns as the real table, where `label` is more likely 1 when `f_dist` is small. When the real files arrive you change only the file path.

**10:35 to 11:00: harness**

`evaluate.py` returns, for candidate rows:

| Metric | Meaning |
| --- | --- |
| Precision, Recall, F1, **Macro F1** | Over the binary label. Macro F1 stops the many label-0 rows from dominating |
| **Log loss** | Quality of the probabilities (the HMM needs them) |
| **Observation-level Top-1** | For each `(trip_id, obs_idx)`, is the highest-probability candidate the true arc? |
| Candidate miss rate | Points where no candidate is true. It caps top-1 |
| Confusion matrix | Optional diagnostic |

`cv.py`:

- `GroupKFold(n_splits=3)` with `groups = trip_id` (3 folds for speed this morning).
- Every model is a scikit-learn `Pipeline`, so scalers fit on training folds only.
- After CV, fit on all 80 train trips, then measure train score, CV score and student-test score. The gap between them is your **overfitting or underfitting evidence**.
- Every run appends one row to `results/experiments.csv`: model, family, config, seed, metrics, train time, inference time. Use a separate file such as `experiments_fake.csv` while testing on fake data.

**The 5 models (`train_all.py`)**

| # | Model | Family | Starting config |
| --- | --- | --- | --- |
| 1 | Logistic Regression | Linear | StandardScaler, C=1.0, max_iter=500 |
| 2 | Gaussian Naive Bayes | Probabilistic | StandardScaler, defaults |
| 3 | KNN | Instance-based | StandardScaler, k=15, distance-weighted, **train on a 150k-row subsample** |
| 4 | Random Forest | Ensemble | 100 trees, max_depth 12, min_samples_leaf 5 |
| 5 | LightGBM | Boosting | 300 trees, learning_rate 0.05, num_leaves 31 |

Skip SVM (too slow). Add an MLP only if time is left. Use only the `f_*` columns as features.

**11:00 to 11:25: real training**

1. Load A’s candidates and C’s labels. Merge: `label = (arc_id == true_arc_id)`. Drop rows where `true_arc_id` is null.
2. Train all 5 models with group CV on the 80 train trips, then evaluate on the student-test trips.
3. Produce:
    - A **comparison table**: model, family, key config, CV macro F1, student top-1, student macro F1, log loss, train time, inference time (and route continuity once C shares the function).
    - Three plots: CV macro F1 bars, student top-1 and macro F1 bars, log loss bars.
    - Loss curves only for LightGBM or an MLP, and only if you really log the training history.
4. **Pick the winner.** Primary metric: CV log loss (the HMM consumes probabilities), with student top-1 and macro F1 as tie-breakers. Also consider fold stability and inference speed. Never choose on training accuracy.

**If blocked**

- Labels late: ask C for the proxy label (nearest reference-route arc per point) and train on it temporarily.
- A model too slow: reduce the subsample or tree count. Don’t drop a family.

**Your first commit:** `tools/make_fake_table.py` and `src/models/evaluate.py` on branch `feat/b-models`, then `cv.py`. Test before committing.

---

### 9. Role C: Labels, then decoder prep

**Files:** `src/labels/assign_labels.py`, `src/hmm/metrics.py`, `viterbi.py`, `transitions.py`, `src/viz/make_map.py`

**By ~10:50: `assign_labels` (B is waiting on this)**

`.route` lists arcs in order, but there are many more GPS points than route arcs. Align them:

- For every point and route arc compute the point-to-arc distance in metres (using A’s projection and arc lines).
- Find a **non-decreasing** assignment: as time goes on, the route index can stay or move forward, never backward, minimising total distance. Use a small dynamic programme: `cost[t][j] = dist(p_t, r_j) + min over j' ≤ j of cost[t-1][j']`.
- If the best distance for a point is over about 100 m, set its label to null.
- Save `results/labels_train.parquet` and `labels_student.parquet` with `trip_id, obs_idx, true_arc_id`.
- **Check on a map** on 2 or 3 trips with the labelled arc highlighted. Count the nulls. Wrong labels make every model look broken.

Until A’s `core/` exists (~10:30), write the DP on a tiny hand-made example.

**11:00 to 11:25 (prep only, not scored)**

- `route_continuity(pred_arcs, graph)`: share of consecutive predicted arcs that are the same or connected. Share with B early.
- Draft Viterbi on a toy graph in log space with fake probabilities. Keep it rough.

---

### 10. Role D: Repo, EDA, API skeleton

**Files:** repo setup, `notebooks/01_eda.ipynb`, `reports/data_issues_log.md`, `src/api/*`, `ui/app.py`

**10:00:** create and push the repo skeleton first (Section 6).

**10:15 to 11:00: EDA on real data (train trips, loaded through `splits.json`)** in `notebooks/01_eda.ipynb`:

- [ ]  Inventory: trips per partition, files per trip, file sizes, rows per `.track`, missing files
- [ ]  Missing or invalid: NaNs, duplicates, out-of-range or swapped coordinates, backward timestamps
- [ ]  Time gaps: histogram of consecutive time differences (about 97% should be 1 s); list trips with big gaps
- [ ]  Extent: bounding box per trip (also tells A the UTM zone)
- [ ]  Road network: node and arc counts, connected components (`networkx`), isolated nodes, degree distribution, directed or reverse arcs
- [ ]  `.route`: length, valid arc ids, are consecutive arcs connected?
- [ ]  Plots: coordinate distributions, trajectories, gap histogram, network plot
- [ ]  One Folium map: GPS (blue), network (grey), reference route (green)
- [ ]  Data Issues and Decisions log: issue, evidence, decision

**11:00 to 11:25:** write a short preprocessing summary, build the API skeleton (`/health`, `/trips`, stub `/predict`), and open all evidence in browser tabs.

---

### 11. Timeline

| Time | A | B (you) | C | D |
| --- | --- | --- | --- | --- |
| 9:45 to 10:15 | Setup together, `splits.json` | Setup together | Setup together | Setup together, repo |
| 10:15 to 10:30 | `core/` loader, projection, graph | Fake table, `evaluate.py` | DP label function on a toy example | EDA: inventory, NaNs, gaps |
| 10:30 to 10:50 | Candidates and features v1 | `cv.py`, 5 model scripts on fake data | `assign_labels` on real data | EDA: network, `.route`, plots |
| 10:50 to 11:00 | Candidate tables for all trips | Finish scripts | Push labels, check on a map | Folium map, issues log |
| 11:00 to 11:20 | Miss-rate table, finalise features | **Train 5 models on real data** | `route_continuity`, Viterbi draft | Preprocessing summary, API skeleton |
| 11:20 to 11:30 | Prepare to present | Comparison table and plots | Help check results | Evidence in tabs |

**Checkpoints**

- **10:30:** is `core/` pushed? If not, C and D write a minimal local loader and swap later.
- **11:00:** are candidates and labels ready? If not, A shrinks scope (fewer trips, fewer features) and ships; B, C and D help rather than wait.

---

### 12. Rules everyone must follow

1. **Evaluator trips are never touched** in development. Always iterate over IDs from `splits.json`, never `os.listdir("data")`.
2. **`GroupKFold` by `trip_id`**, never plain KFold. Rows from one trip are highly similar, so random splits leak and inflate scores.
3. **Scalers fit on training data only** (inside the pipeline).
4. **`.route` is never read in the prediction path.** It is for labels and evaluation only.
5. **One `build_candidates` function** for training and the API.
6. **Never use `arc_id` as a feature.**
7. **Coordinates are (lon, lat).** Never measure distance in degrees.
8. **Every number in the report comes from a real run** saved in `results/`. No invented metrics or curves.
9. **No secrets in git. No sharing code, models or results with other teams.**
10. **Everyone must be able to explain everything.** Use AI for coding help if you like, but run your own experiments, write your own analysis, and disclose AI use as the policy requires.

---

### 13. What to present at 11:30

**Preprocessing and features (A and D)**

- What each file is, how they relate, what cleaning was done (NaNs, the ~3% non-1 s gaps, coordinate checks)
- Why project to metres instead of using degrees
- How candidates are generated, the miss-rate table, why radius and K
- How labels come from `.route` (monotonic alignment) and their limitations
- Each feature and why

**Model selection and training (B)**

- Comparison table of 5 models with real numbers
- Why GroupKFold, and why scalers sit inside the pipeline
- Primary selection metric and why
- Overfitting evidence (train vs CV vs student) for at least two models
- Why macro F1 is over binary candidate rows, and what the miss rate means for the ceiling
- Plan: Optuna on the winner after 11:30


PathSnap Team Playbook

Pastes into Notion as headings, tables and checkboxes. Everyone should read sections 1-3 first, then their own role in section 5.

1. The big picture in plain English

The problem: A car drives along a road, but its GPS dots are noisy. They wobble sideways, jump onto parallel streets, and sometimes disappear. We must work out which road segments the car actually used.

Our approach (3 stages):

Candidates: For every GPS point, list the road segments within ~50-100 m. These are the possible answers.
Classifier: A machine-learning model looks at each (GPS point, candidate segment) pair and outputs the probability that this segment is the correct one.
HMM + Viterbi: The classifier judges each point alone, so its answers may jump between roads that aren’t connected. Viterbi picks the best route overall, using the probabilities plus “a car can only move to a connected road”.

Then we wrap it: a FastAPI server on Laptop 1, a UI on Laptop 2, a GitHub repo, and a PDF report.

Glossary
Term	Meaning in this project
Trip	One drive. Has its own files (.track, .route, and network files .nodes and .arcs)
.track	GPS points: lon lat timestamp, no header
.nodes	Road intersections: lon lat. Row number = node ID (0-based)
.arcs	Road segments: from_node to_node. Row number = arc_id (0-based)
.route	Reference answer: the ordered list of arc_ids the car drove
Candidate	An arc that could plausibly be the one for a GPS point
Label	1 if the candidate is the true arc for that point, else 0
Emission probability	“How likely is this GPS point if the car is on this arc?” For us, the classifier’s probability
Transition probability	“How likely is moving from arc X to arc Y between consecutive points?” Based on road connectivity
Viterbi	Dynamic-programming algorithm that finds the single best chain of arcs
Candidate Miss Rate	% of points where the true arc isn’t in the candidate list. This is a hard ceiling on accuracy
Route continuity	% of consecutive predicted arcs that are the same or connected in the network

Key point: .route is a list of arcs, not one label per GPS point. You must derive per-point labels yourselves (see Phase 2). Everything downstream depends on getting this right.

2. Pipeline at a glance
.track + .nodes + .arcs
   │
   ▼ clean + project to metres
Candidate generation (nearest arcs per GPS point)
   │
   ▼ compute features per (point, arc) row
Classifier  →  P(arc is correct | features)
   │
   ▼ per-point probabilities
HMM / Viterbi (uses road connectivity)  →  predicted arc per GPS point
   │
   ▼
API response: predicted route + accuracy (accuracy computed by a SEPARATE server-side module using .route)

Training uses .route to create labels. Inference must never touch .route. Only the evaluation module reads it, after the prediction is already made.

3. Rules that cost marks (everyone read this)
[ ] Partitions: 80 train / 10 student-test / 10 evaluator-test. Evaluator-test is never touched during development (no training, no feature selection, no tuning, no looking at results).
[ ] Group-aware CV only: use GroupKFold with trip_id as the group. A plain KFold leaks and inflates scores.
[ ] Fit preprocessing on training data only (scalers, imputers, etc.).
[ ] At least 5 models from different families, all documented in the report, with a comparison table.
[ ] Bayesian optimisation (Optuna TPE) on the chosen model, with the full trial log saved.
[ ] Decoder parameters are tuned on train + student-test only.
[ ] No hardcoding of answers. The pipeline must work on any trip folder.
[ ] Ground truth stays server-side and is never exposed to the model or UI.
[ ] Every number in the PDF is reproducible from the repo. No invented metrics or curves. Loss curves only for models that really have training histories.
[ ] No secrets in git (no API keys or passwords).
[ ] Never share code, models or results with other teams. Everything is compared across teams.
[ ] Everyone must be able to explain everything (viva). Use AI for code help if you like, but do the experiments and write the analysis yourselves, and disclose AI assistance as the event policy requires.
[ ] Commit often with clear messages. Repo history is reviewed.
4. Team split
Person	Role	Owns	Report sections
A	Data & Features	Loading, cleaning, EDA, projection, candidates, labels, features	§2.2 (EDA)
B	Models & Tuning	5+ models, CV, metrics, Optuna, model export	§3 (models), §4 (Bayesian)
C	HMM & Maps	Viterbi decoder, route continuity, classifier-vs-HMM comparison, all Folium maps	§5 (HMM), §6 (Folium)
D	Integration	Repo, FastAPI, UI, logging, deployment, README, PDF assembly	§7 (API/UI), final merge

Buddy pairs (so no one is stuck or a single point of failure):

A ↔ C: labels and candidate geometry (both need the road graph).
B ↔ D: model artifact and inference interface.

The critical path: B and C can’t do real work until A produces the candidate table. So:

Agree on the data contract (section 6) in the first 30-45 minutes.
A ships a rough v1 table quickly (3-4 features), then improves it.
While waiting, B builds the CV and metrics harness on dummy data, C builds Viterbi on a toy graph, and D builds the API/UI with a stub predictor.
5. Detailed role checklists
👤 Person A: Data & Features

Phase 1: Exploration (use the doc’s checklist; run on the training partition unless stated)

[ ] Load all files with pandas.read_csv(sep=r"\s+", header=None). Name the columns yourself.
[ ] Inventory table: trips per partition, files per trip, file sizes, rows per .track, any missing files.
[ ] Verify whether .nodes and .arcs are shared across trips or per trip. Don’t assume.
[ ] Missing/invalid: NaNs, duplicates, lon/lat out of range, swapped coordinates, timestamps going backwards.
[ ] Time gaps: histogram of dt. About 97% should be 1 s, so find and characterise the other 3% (which trips have big gaps?).
[ ] Extent: bounding box per trip. Is it one city? Which UTM zone or national grid fits?
[ ] Network: number of nodes and arcs, connected components (networkx), isolated nodes, degree distribution. Are arcs directed? Do reverse arcs exist?
[ ] .route: length, valid arc_ids, consecutive arcs actually connected? (If not, note it as a reference-route limitation.)
[ ] Noise estimate: distance from each GPS point to the nearest reference-route arc (histogram). This tells you the radius R to use.
[ ] Plots: coordinate distributions, trajectory overview, sampling-interval histogram, network plot.
[ ] Keep a Data Issues & Decisions log (a Notion table: issue / evidence / decision / who).

Phase 2: Candidates, labels, features

Projection: convert (lon, lat) to metres with pyproj (UTM zone from the data centroid, or the national grid). Do all distances in metres. Build arc geometry as a straight line from the from-node to the to-node (check whether the dataset has intermediate shape points; it seems not).

Candidates:

[ ] Build a spatial index (shapely.STRtree or geopandas.sindex) over arcs.
[ ] For each GPS point: take arcs within radius R, capped at the K nearest.
[ ] Produce a Candidate Miss Rate table for several (R, K) values on training trips. Pick the smallest setting where the miss rate is very low. Put this table in the report.

Labels (the tricky part):

[ ] For each trip, take the reference route arc sequence r₁, r₂, …, rₘ.
[ ] Assign each GPS point to a route arc with a monotonic pointer: the route index can stay the same or move forward, never backward. Simple approach: a small DP minimising total point-to-arc distance with a non-decreasing index constraint.
[ ] Label = 1 for the candidate row whose arc_id equals the assigned route arc, else 0.
[ ] Sanity check: plot some trips with labels on a map (give C the data to render). Report label noise honestly. Reference routes aren’t perfect truth.

Features (document why each exists):

Feature	Why
Perpendicular distance point → arc (m)	Closer arc is more likely
Distance rank among candidates	Relative closeness
Distance minus best candidate’s distance	Same, normalised per point
Projection position along arc (0-1)	Near arc ends the point could belong to the next arc
Bearing difference: GPS heading vs arc direction	Separates parallel roads and the opposite direction
Speed, time gap to previous point	Detects gaps and stopping
Arc length, node degree at ends	Network context
Neighbour context: distance of prev/next points to this arc	Gives the classifier sequence context
Heading change over last few points	Turn detection
[ ] Do NOT use raw arc_id as a numeric feature, and nothing derived from .route at inference time.
[ ] Save candidates_{train|student}.parquet and a features.py with one function used by both training and the API (the same code path for both, or you’ll get train/serve skew).

A’s definition of done: v1 table delivered early, then v2 with miss-rate tuning; EDA plots; data-issue log; report §2.2 draft.

👤 Person B: Models & Tuning

Setup (before data exists)

[ ] Build evaluate.py: given labels, probabilities, trip_id and obs_idx, return all metrics (below). Test it on a dummy table.
[ ] Build cv.py: GroupKFold(n_splits=5) by trip, a fixed seed, and a results logger that appends to results/experiments.csv.

Metrics to compute for every model

Metric	Level	Notes
Precision, Recall, F1, Macro F1	Candidate rows	Heavy class imbalance: most candidates are label 0. Macro F1 treats both classes equally. State clearly which level you compute it at
Log Loss	Candidate rows	Measures probability quality, which the HMM needs
Observation-level Top-1 Accuracy	Per GPS point	Pick the highest-probability candidate and compare with the reference arc
Candidate Miss Rate	Per GPS point	Same for all models (comes from A’s candidates)
Route continuity	Per trip	Computed from top-1 picks (C gives you the function)
Training time, inference time	Per model	Use time.perf_counter
Confusion matrix	Candidate rows	Optional but cheap

Ask organisers (or decide and justify): whether “Macro F1” is over the binary candidate label (recommended) or multiclass over arcs (impractical with many arcs).

Models to train (min 5, different families)

#	Model	Family	Notes
1	Logistic Regression	Linear	Scale features; baseline
2	Gaussian Naive Bayes	Probabilistic	Fast; probabilities often poorly calibrated
3	KNN	Instance-based	Scale features; subsample training rows, or it’s very slow
4	Random Forest / Extra Trees	Ensemble	Limit depth/trees for inference speed
5	HistGradientBoosting / LightGBM / XGBoost	Boosting	Likely the strongest; has real training history
6	MLP (optional)	Neural	Has epoch-wise loss; plot only real curves

Skip full-dataset SVMs on millions of rows. If you include one, subsample and say so.

For each model, record the answers the report asks for:

[ ] Why it suits PathSnap (tie it to your features, not the textbook).
[ ] Strengths and limitations seen in your results.
[ ] Initial hyperparameters and the loss/objective.
[ ] Student-test: top-1, macro F1, log loss.
[ ] Over/underfitting evidence: train score vs CV score vs student-test score. State the gap.

Selection

[ ] Choose a primary selection metric and justify it (suggested: CV log loss or CV macro F1, with student-test top-1 and continuity as tie-breakers). Also weigh stability across folds, inference speed and compatibility with the decoder.
[ ] Never choose on training accuracy.

Bayesian optimisation (Optuna TPE)

[ ] Search space justified (e.g. for LightGBM: learning rate log-uniform, num_leaves, max_depth, min_child_samples, subsample, colsample, reg_lambda, n_estimators with early stopping).
[ ] Objective: mean GroupKFold score on the 80 training trips (say which one and why).
[ ] Record every trial (params, score, duration) to CSV and the study to SQLite. Report trial count and total compute time.
[ ] Refit with best params, evaluate on student-test, and compare tuned vs untuned on CV macro F1, student top-1, student macro F1, log loss and inference time (same partitions, same metrics).

Export: artifacts/model.joblib (or the native format), the scaler, the feature list/order, and a config.json. Hand D a predict_proba(df_features) → array function.

B’s definition of done: the comparison table, per-model sections, comparison plots (CV F1 bar chart, student top-1/F1, log loss), the Optuna log, and the exported artifacts.

👤 Person C: HMM, Viterbi & Maps

Start early on a toy graph (don’t wait for data).

Design

Hidden states at time t: the candidate arcs for GPS point t.
Emission: log P_clf(arc | point) from the classifier, normalised across candidates at each time step, optionally raised to a weight α (tunable). Optionally blend with a distance-based Gaussian.
Transition (arc i at t-1 → arc j at t):
same arc: high
j is a successor of i (head of i = tail of j): medium
2-3 hops apart: lower (needed when sampling skips arcs at speed)
not reachable within N hops: tiny ε, never exactly 0
Better version (Newson-Krumm): exp(-|network_distance(i,j) − gps_distance| / β). Precompute capped shortest paths with Dijkstra.
Viterbi (log space):
δ[0][j]   = log_emit[0][j]
δ[t][j]   = log_emit[t][j] + max_i ( δ[t-1][i] + log_trans[i][j] )
ψ[t][j]   = argmax_i ( ... )
backtrack from argmax_j δ[T-1][j]

Keep the candidate order identical between the classifier output and the decoder (the candidate-to-arc correspondence).

Edge cases (the report asks about all of these):

[ ] Empty candidate set at a point: widen the radius for that point, or skip it and restart the chain after it.
[ ] Large GPS gap (e.g. dt > threshold): split the trip into segments, decode each separately, or use a more permissive transition.
[ ] No valid transition (all probabilities ≈ 0): break and restart, and count how often it happens.
[ ] Return one predicted arc per relevant GPS point.

Metrics (write functions and share them with B)

Route continuity = share of consecutive predicted pairs (a_t, a_{t+1}) with a_t == a_{t+1} or connected (define your hop limit and say so).
Sequence-level: arc-set precision/recall vs the reference route, longest correct run, edit distance between the de-duplicated predicted route and the reference route.
Pointwise accuracy: classifier-only vs classifier+HMM.
[ ] Tune α, β, ε, hop limit and gap threshold on train + student-test only. Record the search.
[ ] Find and document cases where HMM did not help.

Folium maps (src/viz/make_map.py)

[ ] Function make_map(trip_id, predicted=None) with layers: raw GPS (blue dots), road network (grey), reference route (green), predicted route (red), plus a legend, layer control and auto-fit zoom to the trip bounds.
[ ] Save an HTML per trip in results/maps/. Screenshot at least one for the PDF.
[ ] Also provide an “EDA map” (GPS + network + reference) for A’s section.
[ ] Provide the map function to D for the UI. For the live demo, the map shows the GPS and the predicted route. See the ground-truth note under D.

C’s definition of done: a tested decoder, the classifier-only vs HMM comparison table, the maps, and report §5-6.

👤 Person D: Integration, API, UI, Repo

Day 1 hour 0:

[ ] Create the public GitHub repo, add .gitignore (raw data, large files, .env), the folder skeleton, requirements.txt and an empty README with headings. Add everyone as collaborators.
[ ] Agree on a branch strategy: everyone works on their own branch (feat/a-data, feat/b-models, …) and merges to main via small PRs, or at least pulls often.

API (Laptop 1, FastAPI)

GET  /health            → {"status":"ok"}
GET  /trips             → ["trip_01", ... 10 IDs]
POST /predict {"trip_id": "..."} →
     {
       "trip_id": "...",
       "predicted_arcs": [...],          # one per GPS point
       "route": [...],                   # de-duplicated arc sequence
       "gps_points": [[lon,lat],...],
       "predicted_geometry": [[lon,lat],...],
       "metrics": {"top1_acc":..., "route_continuity":..., "seq_f1":...},
       "timing_ms": ...
     }
[ ] Load model, scaler, network and HMM config once at startup, not per request.
[ ] Prediction path: reads only .track and the network files → candidates → features → classifier → HMM. It must never open .route.
[ ] A separate evaluation.py loads .route after prediction and computes metrics. Reference data stays server-side.
[ ] Run with uvicorn src.api.main:app --host 0.0.0.0 --port 8000; enable CORS; allow the port through the Windows/macOS firewall.
[ ] Log every request (timestamp, trip_id, duration, status) to the console. You must show these live to evaluators.
[ ] Test against all 10 trips (use the student-test trips during development via the same code path).

UI (Laptop 2): a simple Streamlit (or HTML/JS) page: dropdown of trips from GET /trips → button → POST /predict → show the Folium map, metrics cards, a loading state and clear error messages. The API URL comes from a config/env variable (the IP changes on the day).

Ground-truth caution: the rules say ground truth must “never be exposed to the UI” yet the UI shows “evaluation results.” The safest reading: the API returns the predicted route and the computed metrics, not the reference geometry. If you want a reference overlay in the live UI, ask an organiser first.

Network testing (do this early, not at the end)

[ ] Both laptops on the same Wi-Fi (or one phone hotspot); find Laptop 1’s IP (ipconfig / ifconfig).
[ ] From Laptop 2: curl http://<ip>:8000/health.
[ ] Take screenshots of the UI and of API responses in the terminal on the receiving laptop (the report requires this).
[ ] Rehearse the full evaluator flow twice.

Repo + report

[ ] README: install, run (API and UI), data location, how to reproduce experiments, folder tour.
[ ] Check that the code in the repo exactly matches what you demo.
[ ] Assemble the final single PDF; check every required section and screenshot is present.
6. Shared contracts (agree in the first 30 minutes)

Folder structure

data/                 (raw data, gitignored)
src/data/             loaders, cleaning
src/features/         candidates.py, labels.py, features.py
src/models/           train, cv, evaluate, tune
src/hmm/              viterbi.py, transitions.py, metrics.py
src/viz/              make_map.py
src/api/              main.py, pipeline.py, evaluation.py
ui/                   app.py
notebooks/            EDA, experiments
results/              csvs, plots, maps, optuna logs
artifacts/            model, scaler, config
reports/              report sections
README.md  requirements.txt

Candidate table (parquet), one row per (observation, candidate arc)

Column	Meaning
trip_id	string
obs_idx	index of GPS point in the trip (0-based)
arc_id	candidate arc (not a model feature)
f_*	feature columns
label	0/1 (training and eval only; absent at inference)

Fixed items

[ ] splits.json: {"train": [80 ids], "student_test": [10 ids]}. Nobody edits it.
[ ] SEED = 42 in a shared config.py.
[ ] Coordinates are always (lon, lat) in raw form. Projected metres use one shared function.
[ ] One features.py is used by both training and the API.
[ ] Shared function signatures, e.g.:
load_trip(trip_id) -> Trip
build_candidates(trip) -> DataFrame
model.predict_proba(X) -> ndarray
decode(candidates_df, probs, graph, params) -> list[arc_id]
metrics.route_continuity(pred_arcs, graph) -> float
7. Timeline

I don’t know how long the event is or what the pre-lunch evaluation covers. Confirm both with organisers now, then scale this plan. Blocks are relative.

Block	A	B	C	D
Kickoff (~45 min)	Read docs together; agree contracts; start loading data	Build the CV and metrics harness	Viterbi on a toy graph	Repo, skeleton, stub API + UI
Block 1	EDA + v1 candidates and labels → hand off	First 3 models on v1	Continuity/sequence metrics	Folium base, API endpoints
Block 2	Features v2, miss-rate table, EDA plots	Remaining models, comparison plots, start Optuna	Plug real probabilities into HMM, tune decoder	Wire the real pipeline into the API
Block 3	Report §2.2, issue log	Tuned model, export artifacts, report §3-4	Maps, ablation, report §5-6	Two-laptop test, all-10-trips test, README
Final block	Cross-check numbers	Cross-check numbers	Cross-check numbers	Merge PDF, final repo check, rehearsal

Sync points: a 5-minute stand-up every ~90 minutes. Say what you finished, what you need from someone, and what blocks you.
Freeze point: once the model and decoder are final, nobody changes them. Re-running the frozen pipeline on student-test for the report is fine.

8. Report outline and ownership
Report section	Content	Owner
2.2 EDA	Preprocessing, missing-value analysis, trajectory plots, network visuals, a Folium map, partition named for each analysis, features and why	A
3.1	Consolidated 5+ model comparison table (use actual numbers, or “N/A” with the reason)	B
3.2	One subsection per model answering the 9 compulsory questions	B
3.3	CV macro F1 bar chart, student top-1/F1, log loss, confusion matrices, real loss curves only	B
4	Bayesian optimisation answers + tuned vs untuned table	B
5	HMM answers (states, emission, transition, gaps, disconnected segments, did it help) + classifier-only vs HMM table	C
6	Folium maps: GPS, network, reference, predicted; legend; layer control	C
7	UI screenshots, API terminal screenshots, deployment steps	D
AI disclosure, final merge		D

Each person writes from their own results. Everyone reads the whole PDF before submission.

9. Common pitfalls
Swapping lon/lat, or computing distances in degrees.
Plain KFold instead of GroupKFold.
Fitting scalers on all data or training on student-test trips.
Bad label alignment: scores look absurdly low or perfect. Plot labels on a map to check.
Candidate radius too small, so a high miss rate caps every model.
Training KNN/SVM on millions of rows.
Training features computed one way and API features another way.
.route read anywhere in the prediction path.
Only testing on 1-2 trips. All ten must work.
Leaving two-laptop networking until the last hour.
Reporting numbers from an old run. Regenerate the final numbers from the final code.
10. Viva prep: everyone should be able to answer
What are .nodes, .arcs and .route, and how do they relate?
How did you create per-point labels from the route? What are its limitations?
Why group-aware CV? What would go wrong otherwise?
Why did you choose your final model and primary metric? What were the runner-ups?
What did Optuna search over, and how much did it help?
Define the HMM states, emissions, transitions and the Viterbi recurrence.
What happens when a GPS gap or an empty candidate set occurs?
Where did the HMM not help, and why?
Show where in the code the API avoids touching ground truth.
Why is the candidate miss rate a ceiling on accuracy?