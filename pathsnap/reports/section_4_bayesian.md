# 4. Bayesian optimization

## 4.1 Objective and search design

Optuna's TPE sampler tuned the LightGBM candidate classifier. The objective was
the mean candidate-row log loss across three `GroupKFold` folds grouped by
`trip_id`, using only the 80 training trips. Log loss was chosen because the
classifier probabilities are consumed by the HMM; the 10 student-test trips
were not used to select trial parameters and were evaluated only after
selection.

The search used seed 42 and the following ranges:

| Parameter | Search range | Rationale |
|---|---|---|
| `learning_rate` | 0.01–0.15, log scale | Search both smaller, steadier updates and faster learning rates. |
| `n_estimators` | 150–500, step 50 | Trade model capacity against fit and inference cost. |
| `max_depth` | 4–10 | Bound tree complexity. |
| `num_leaves` | 15 to `min(63, 2**max_depth)` | Control leaf capacity while respecting the selected depth. |
| `min_child_samples` | 10–100 | Regularize leaves with small sample counts. |
| `subsample` | 0.7–1.0 | Row subsampling; `subsample_freq=1` enables it. |
| `colsample_bytree` | 0.7–1.0 | Feature subsampling for regularization. |
| `reg_lambda` | 1e-4–10, log scale | Search a broad range of L2 regularization strengths. |

`n_estimators` was searched directly; early stopping was not used.

## 4.2 Search results

Ten trials completed. One additional trial was interrupted when the search was
shortened; it remains marked `RUNNING` in the study/log and is not included in
the completed-trial count or compute-time sum. The 10 completed trials took a
total of 446.1 seconds of trial compute time (sum of trial durations; about
7 minutes 26 seconds). The database and per-trial log are
[`lgbm_tuning.db`](../results/optuna/lgbm_tuning.db) and
[`lgbm_trials.csv`](../results/optuna/lgbm_trials.csv).

The best trial's mean fold log loss was 0.131850, with:

```text
n_estimators       = 450
learning_rate      = 0.0171725
max_depth          = 7
num_leaves         = 40
min_child_samples  = 63
subsample          = 0.713935
colsample_bytree   = 0.882263
reg_lambda         = 0.0007122
```

The final comparison below uses the same three CV folds and the same
student-test split for untuned and tuned LightGBM. CV log loss here is pooled
out-of-fold loss; it therefore differs slightly from the objective's mean of
the three fold losses.

| Metric | Untuned LightGBM | Tuned LightGBM |
|---|---:|---:|
| CV log loss | 0.137642 | **0.131832** |
| CV macro-F1 | 0.875557 | **0.876447** |
| CV top-1 | 79.869% | **80.062%** |
| Student log loss | 0.117321 | **0.114376** |
| Student macro-F1 | 0.893604 | **0.893722** |
| Student top-1 | 82.175% | **82.264%** |
| Final fit time | 11.43 s | 22.64 s |
| Student inference time | 0.358 s | 0.694 s |

Tuning reduced CV log loss by about 4.2% and student log loss by about 2.5%.
Student top-1 increased by 0.089 percentage points, a small gain. The tuned
model takes longer to fit and infer than the untuned LightGBM. Ten trials
provide a modest search, not a guarantee of a global optimum; the guide sets
no minimum trial count.

## 4.3 Export and handoff

The selected model and feature configuration are saved in
[`model_bundle.joblib`](../artifacts/model_bundle.joblib) and
[`config.json`](../artifacts/config.json). The LightGBM model does not require
a scaler. `LightGBMBundle.predict_proba(cands_df)` returns a one-dimensional
array in the input candidate-row order, normalized within each observation,
for downstream decoding.

The accuracy values in the comparison are classifier-only top-1 on the
retained, aligned, candidate-hit observations. They are not the final
evaluator score if it includes candidate misses or uses HMM-decoded arcs.
Update this section with the classifier-vs-HMM pointwise accuracy and route
continuity once C's real-data integration is available.

