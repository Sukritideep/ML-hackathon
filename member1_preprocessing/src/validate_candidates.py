
from pathlib import Path
import json
import pandas as pd

CANDIDATE_DIR = Path(
    "member1_preprocessing/outputs/candidates"
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

        file = CANDIDATE_DIR / f"{trip_id}.csv"
        df = pd.read_csv(file)

        counts = df.groupby("obs_idx").size()

        # Check candidate rank sequence
        rank_errors = 0

        for _, group in df.groupby("obs_idx"):
            expected = list(range(1, len(group) + 1))
            actual = sorted(group["candidate_rank"].tolist())

            if actual != expected:
                rank_errors += 1

        # Check duplicate candidate arcs for an observation
        duplicate_pairs = int(
            df.duplicated(
                subset=["obs_idx", "arc_id"]
            ).sum()
        )

        # Check candidate radius and fallback logic
        fallback_rows = df["fallback_used"].fillna(False).astype(bool)
        outside_radius_rows = df["distance_m"] > 75

        fallback_observations = int(
            df.loc[fallback_rows, "obs_idx"].nunique()
        )

        observations = int(df["obs_idx"].nunique())

        reports.append({
            "partition": partition,
            "trip_id": trip_id,
            "observations_with_candidates": observations,
            "candidate_rows": len(df),
            "average_candidates": counts.mean(),
            "maximum_candidates": counts.max(),
            "observations_with_multiple_candidates": int(
                (counts > 1).sum()
            ),
            "fallback_observations": fallback_observations,
            "outside_75m_rows": int(outside_radius_rows.sum()),
            "duplicate_candidate_pairs": duplicate_pairs,
            "rank_errors": rank_errors,
            "missing_observation_indices": int(
                max(df["obs_idx"].max() + 1 - observations, 0)
            )
        })

result = pd.DataFrame(reports)

output_file = REPORT_DIR / "candidate_validation.csv"
result.to_csv(output_file, index=False)

print("\nCANDIDATE VALIDATION")
print("====================")
print("Trips checked:", len(result))
print("Observations with candidates:",
      result["observations_with_candidates"].sum())
print("Candidate rows:", result["candidate_rows"].sum())
print("Fallback observations:",
      result["fallback_observations"].sum())
print("Duplicate candidate pairs:",
      result["duplicate_candidate_pairs"].sum())
print("Rank errors:", result["rank_errors"].sum())
print("Maximum candidates for any observation:",
      result["maximum_candidates"].max())

print("\nTrips with most fallback observations:")
print(
    result.nlargest(10, "fallback_observations")[
        ["trip_id", "fallback_observations",
         "observations_with_candidates"]
    ].to_string(index=False)
)

print("\nReport saved to:", output_file)
