# Final Student-Test Evaluation

## Summary

The saved LightGBM bundle was evaluated on all 10 `student_test` trips and was
not retrained. The evaluation compared per-observation classifier top-1
predictions with the same probabilities decoded by the HMM/Viterbi pipeline.
It reuses the project's `sequence_metrics` and route-continuity helpers.

The classifier achieved **82.14% pointwise accuracy** over all 19,143
observations with aligned, non-null labels. After fixing Viterbi's handling of
impossible transitions, HMM/Viterbi achieved **83.18%** on the same denominator.
It also improved route continuity and edit distance in this run.

| Aggregate metric | Classifier only | HMM/Viterbi |
|---|---:|---:|
| Pointwise accuracy | 82.14% | 83.18% |
| Pointwise precision | 82.14% | 83.18% |
| Pointwise recall | 82.14% | 83.18% |
| Unique-arc set precision | 73.85% | 81.78% |
| Unique-arc set recall | 96.06% | 98.03% |
| Route continuity | 96.34% | 99.72% |
| Total edit distance (lower is better) | 3,419 | 3,204 |
| Longest correct run (maximum across trips) | 971 | 1,204 |
| Null predictions | 0 | 0 |

HMM/Viterbi's pointwise accuracy is **1.04 percentage points higher** than the
classifier's in this run. Route continuity is still a separate metric: it is
calculated over neighboring predictions and should not be treated as a
substitute for accuracy.

## Evaluation details

- Model: saved `artifacts/model_bundle.joblib`; no training or tuning was run.
- Student-test trips: 10; GPS observations: 19,160.
- Non-null aligned labels: 19,143 (17 labels were unaligned).
- Candidate coverage: 19,114 / 19,143 valid labels (29 candidate misses;
  99.85% candidate recall).
- Candidate misses and null HMM predictions count as incorrect in pointwise
  accuracy. Unaligned labels are excluded.
- HMM transition scale: 20 m; directed route-continuity hop limit: 3;
  long-gap boundaries split decoding segments, and impossible transitions
  finalize the previous path and restart decoding at the current observation.
- Pointwise precision is exact-arc matches divided by non-null predictions on
  valid-label observations. Pointwise recall is exact-arc matches divided by
  valid labels; therefore it equals pointwise accuracy here.
- Arc-set precision/recall use the existing `sequence_metrics` implementation
  and scope aggregate arc IDs by trip. Edit distance is summed over trips;
  longest correct run is the maximum across trips.

The 82.14% figure uses all valid labels, including candidate misses. If those
29 misses are excluded, classifier-only top-1 is 82.26% (15,724 / 19,114),
matching the earlier conditional metric. Do not mix the two denominators.

## Per-trip comparison

| Trip | Valid labels | Classifier accuracy | HMM accuracy | HMM nulls | Classifier continuity | HMM continuity |
|---|---:|---:|---:|---:|---:|---:|
| 00000000 | 2,503 | 92.6% | 93.7% | 0 | 98.2% | 99.8% |
| 00000007 | 1,354 | 92.1% | 86.3% | 0 | 99.7% | 100.0% |
| 00000009 | 1,049 | 82.3% | 87.6% | 0 | 95.2% | 99.0% |
| 00000040 | 2,995 | 88.0% | 88.2% | 0 | 98.9% | 100.0% |
| 00000052 | 262 | 93.9% | 99.2% | 0 | 96.6% | 100.0% |
| 00000056 | 3,941 | 86.9% | 89.4% | 0 | 91.7% | 99.6% |
| 00000065 | 768 | 79.6% | 76.4% | 0 | 95.1% | 100.0% |
| 00000076 | 1,486 | 77.1% | 84.2% | 0 | 93.8% | 99.6% |
| 00000077 | 3,911 | 64.3% | 64.1% | 0 | 99.0% | 100.0% |
| 00000091 | 874 | 82.0% | 82.4% | 0 | 93.1% | 98.3% |

`00000000` has 2,503 consecutive observations and no long-gap boundaries. Its
candidate arc IDs are present in the graph, but some transition between
observations is unreachable. Before the fix, that made Viterbi return null for
the whole trip segment. The decoder now restarts at the observation with no
incoming transition; it returns predictions for the trip and records two
unreachable predicted transitions. This changes HMM accuracy from 70.93% to
83.18% overall without retraining the classifier.

## Map availability

Five HMM maps were generated:

- [00000056](../results/maps/hmm_trip_00000056.html)
- [00000065](../results/maps/hmm_trip_00000065.html)
- [00000076](../results/maps/hmm_trip_00000076.html)
- [00000077](../results/maps/hmm_trip_00000077.html)
- [00000091](../results/maps/hmm_trip_00000091.html)

Maps for `00000000`, `00000007`, `00000009`, `00000040`, and `00000052` could
not be generated because their raw `.track`, `.nodes`, and `.route` files are
absent from the local data directory. For `00000056`, the missing raw `.arcs`
table was read from its existing road-geometry NPZ; no preprocessing output
was changed. No `dp_trip_*.html` maps were overwritten.

## Output files

- Aggregate JSON, including metric definitions and map availability:
  [`hmm_ml_evaluation_report.json`](../results/hmm_ml_evaluation_report.json)
- Observation-aligned classifier and HMM predictions:
  [`hmm_ml_predictions_student.parquet`](../results/hmm_ml_predictions_student.parquet)
- Per-trip metrics:
  [`hmm_ml_per_trip_metrics.csv`](../results/hmm_ml_per_trip_metrics.csv)
- Aggregate metrics:
  [`hmm_ml_aggregate_metrics.csv`](../results/hmm_ml_aggregate_metrics.csv)

These are internal student-test results, not an official evaluator score.
