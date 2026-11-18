
from pathlib import Path
import json
import numpy as np
import pandas as pd

CANDIDATE_DIR = Path(
    "member1_preprocessing/outputs/candidates"
)

FEATURE_DIR = Path(
    "member1_preprocessing/outputs/features"
)

REPORT_DIR = Path(
    "member1_preprocessing/outputs/reports"
)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open("splits.json", "r") as f:
    splits = json.load(f)

partitions = {
    "train": splits["train"],
    "student_test": splits["student_test"]
}

reports = []

for partition, trip_ids in partitions.items():

    for trip_id in trip_ids:

        candidate_file = CANDIDATE_DIR / f"{trip_id}.csv"
        feature_file = FEATURE_DIR / f"{trip_id}.csv"

        candidates = pd.read_csv(candidate_file)
        features = pd.read_csv(feature_file)

        required_columns = [
            "obs_idx",
            "arc_id",
            "distance_m",
            "projection_fraction",
            "arc_length_m",
            "gps_speed_mps",
            "gps_heading_deg",
            "direction_difference_deg",
            "long_gap"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in features.columns
        ]

        # Verify candidate and feature row counts
        row_count_matches = (
            len(candidates) == len(features)
        )

        # Verify candidate identity/order
        identity_matches = (
            candidates[["obs_idx", "arc_id"]].equals(
                features[["obs_idx", "arc_id"]]
            )
        )

        # Verify one to ten candidates per observation
        counts = features.groupby("obs_idx").size()

        candidate_limit_violations = int(
            ((counts < 1) | (counts > 10)).sum()
        )

        # Feature value checks
        distance_invalid = int(
            (
                ~np.isfinite(features["distance_m"]) |
                (features["distance_m"] < 0)
            ).sum()
        )

        projection_invalid = int(
            (
                ~np.isfinite(features["projection_fraction"]) |
                (features["projection_fraction"] < 0) |
                (features["projection_fraction"] > 1)
            ).sum()
        )

        speed_invalid = int(
            (
                features["gps_speed_mps"].dropna() < 0
            ).sum()
        )

        # Fallback observations
        fallback_count = int(
            features.loc[
                features["fallback_used"].fillna(False),
                "obs_idx"
            ].nunique()
        )

        reports.append({
            "partition": partition,
            "trip_id": trip_id,
            "candidate_rows": len(candidates),
            "feature_rows": len(features),
            "feature_columns": len(features.columns),
            "row_count_matches": row_count_matches,
            "candidate_identity_matches": identity_matches,
            "missing_required_columns": ",".join(missing_columns),
            "candidate_limit_violations": candidate_limit_violations,
            "invalid_candidate_distances": distance_invalid,
            "invalid_projection_fractions": projection_invalid,
            "negative_gps_speeds": speed_invalid,
            "fallback_observations": fallback_count
        })

result = pd.DataFrame(reports)

output_file = REPORT_DIR / "final_preprocessing_validation.csv"
result.to_csv(output_file, index=False)

print("\nFINAL PREPROCESSING VALIDATION")
print("==============================")
print("Trips checked:", len(result))
print("Candidate rows:", result["candidate_rows"].sum())
print("Feature rows:", result["feature_rows"].sum())

print(
    "Row count mismatches:",
    int((~result["row_count_matches"]).sum())
)

print(
    "Candidate identity mismatches:",
    int((~result["candidate_identity_matches"]).sum())
)

print(
    "Missing required columns:",
    int((result["missing_required_columns"] != "").sum())
)

print(
    "Candidate limit violations:",
    result["candidate_limit_violations"].sum()
)

print(
    "Invalid candidate distances:",
    result["invalid_candidate_distances"].sum()
)

print(
    "Invalid projection fractions:",
    result["invalid_projection_fractions"].sum()
)

print(
    "Negative GPS speeds:",
    result["negative_gps_speeds"].sum()
)

print("\nTrips requiring attention:")
print(
    result[
        (result["fallback_observations"] > 0) |
        (~result["row_count_matches"]) |
        (~result["candidate_identity_matches"])
    ][
        ["trip_id", "fallback_observations",
         "row_count_matches", "candidate_identity_matches"]
    ].to_string(index=False)
)

print("\nReport saved to:", output_file)
