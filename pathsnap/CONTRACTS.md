# Contracts

- Raw coordinates are always (lon, lat); distances only in metres after projection.
- `load_trip(trip_id, with_route=False)` returns an object with `gps` (lon, lat, t), `nodes` (lon, lat), `arcs` (u, v), `route`.
- `build_candidates(trip, graph)` returns `trip_id, obs_idx, arc_id, f_*`, sorted by `(trip_id, obs_idx, arc_id)`, with no label.
- `assign_labels(trip, graph)` returns `trip_id, obs_idx, true_arc_id`.
- `FEATURE_COLS` list is defined in `features/candidates.py`.
- Candidate settings: radius 75 m, K at most 10; seed 42.
- Model bundle exposes `predict_proba(cands_df)`.
- HMM `DistanceEmissionModel` consumes candidate rows with `trip_id`, `obs_idx`,
  `arc_id`, and `distance_m`; it exposes row-aligned probabilities and log
  probabilities normalized per `(trip_id, obs_idx)`. This is a configurable
  distance-only baseline, not a trained ML emission model.
- `decode(cands, probs, graph, params)` consumes row-aligned emission
  probabilities and candidate features including `gps_distance_m` for
  transitions. `params` includes `transition_scale_m`; `long_gap` starts an
  independent decoding segment by default. It returns `obs_idx, pred_arc_id`
  and includes `trip_id` when present in the input.

