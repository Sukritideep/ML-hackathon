# Contracts

- Raw coordinates are always (lon, lat); distances only in metres after projection.
- `load_trip(trip_id, with_route=False)` returns an object with `gps` (lon, lat, t), `nodes` (lon, lat), `arcs` (u, v), `route`.
- `build_candidates(trip, graph)` returns `trip_id, obs_idx, arc_id, f_*`, sorted by `(trip_id, obs_idx, arc_id)`, with no label.
- `assign_labels(trip, graph)` returns `trip_id, obs_idx, true_arc_id`.
- `FEATURE_COLS` list is defined in `features/candidates.py`.
- Candidate settings: radius 75 m, K at most 10; seed 42.
- Model bundle exposes `predict_proba(cands_df)`.
- `decode(cands, probs, graph, params)` returns `obs_idx, pred_arc_id`.

