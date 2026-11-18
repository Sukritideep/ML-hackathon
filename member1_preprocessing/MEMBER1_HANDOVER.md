# Member 1 — Data Preprocessing and Feature Engineering

## Project

PathSnap — GPS trajectory reconstruction using road-network map matching.

## Completed work

1. Dataset inspection.
2. GPS and road-network data-quality audits.
3. Timestamp-gap and route-connectivity analysis.
4. Per-trip CRS audit.
5. Road-network geometry audit.
6. Projected road geometry generation.
7. GPS-to-road candidate generation.
8. Candidate validation.
9. Spatial, temporal, movement and road-network feature extraction.
10. Final structural validation.
11. Folium trajectory map generation.
12. Consolidated report generation.

## Dataset totals

- Trips processed: 90
- GPS observations: 227,577
- Road nodes: 2,024,174
- Road arcs: 4,061,945
- Route entries: 34,813
- Candidate rows: 1,615,773
- Average candidates per observation: 7.10
- Observations without a road within 75 metres: 13,604
- Fallback percentage: 5.98%
- Feature columns per candidate: 41

## Validation results

- Zero invalid road arc references.
- Zero duplicate candidate-observation pairs.
- Zero candidate rank errors.
- Zero invalid candidate distances.
- Zero invalid projection fractions.
- Zero negative GPS speeds.
- Candidate and feature row counts match.
- Average unique route-arc candidate overlap: 99.60%.

## Issues requiring attention

1. Trip 00000054 has an unusually high number of fallback observations.
2. Some trips have lower unique route-arc candidate overlap.
3. There are 1,260 GPS intervals greater than five seconds.
4. Fourteen route transitions were flagged as not directly connected.
5. There are 106 zero-length arcs and 24 self-loop arcs.

## Label alignment limitation

The reference .route files contain chronological road arc sequences.
They are not directly aligned one-to-one with GPS observations.

Unique route-arc candidate overlap is NOT GPS-level candidate recall.

A reliable chronological alignment between GPS observations and reference route arcs is still required before creating supervised training labels.

Do not treat the candidate rows as labelled training examples until this alignment is validated.

## Output folders

- outputs/road_geometry/
- outputs/candidates/
- outputs/features/
- outputs/reports/
- outputs/maps/

## Handover to Member 2

Member 2 can inspect the feature tables and prepare the model-training pipeline.

Supervised model training must wait until GPS-to-route label alignment is completed and observation-level candidate recall is measured.

## Handover to Member 3

Review route sequence continuity and help validate sequence alignment and transition handling.

## Reproducibility

All preprocessing scripts are stored in member1_preprocessing/src/.

The original raw dataset has not been modified.
