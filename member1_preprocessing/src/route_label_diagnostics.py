
from pathlib import Path
import json
import numpy as np
import pandas as pd

DATA_ROOT = Path(
    "/mnt/c/Users/sri07/Documents/Student folder/Student folder"
)

GEOMETRY_DIR = Path(
    "member1_preprocessing/outputs/road_geometry"
)

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

        trip_dir = DATA_ROOT / partition / trip_id

        route = pd.read_csv(
            trip_dir / f"{trip_id}.route",
            sep=r"\s+",
            header=None,
            names=["arc_id"]
        )

        route_ids = route["arc_id"].to_numpy(dtype=np.int64)

        candidates = pd.read_csv(
            CANDIDATE_DIR / f"{trip_id}.csv",
            usecols=["obs_idx", "arc_id", "distance_m"]
        )

        geometry = np.load(
            GEOMETRY_DIR / f"{trip_id}.npz"
        )

        geometry_ids = set(
            geometry["arc_id"].astype(np.int64).tolist()
        )

        candidate_ids = set(
            candidates["arc_id"].astype(np.int64).tolist()
        )

        route_id_set = set(route_ids.tolist())

        # Route entries represented in road geometry
        geometry_matches = np.isin(
            route_ids,
            list(geometry_ids)
        )

        # Unique route arc IDs represented in any candidate
        candidate_matches = np.isin(
            list(route_id_set),
            list(candidate_ids)
        )

        unique_route_ids = list(route_id_set)

        unique_route_covered = int(candidate_matches.sum())

        route_candidate_rows = candidates[
            candidates["arc_id"].isin(route_id_set)
        ]

        # Count GPS observations having at least one
        # candidate whose arc ID appears somewhere in route.
        observations_with_route_candidate = (
            route_candidate_rows["obs_idx"].nunique()
        )

        reports.append({
            "partition": partition,
            "trip_id": trip_id,
            "route_entries": len(route_ids),
            "unique_route_arc_ids": len(route_id_set),
            "route_entries_in_geometry_pct": (
                100 * geometry_matches.mean()
            ),
            "unique_route_arc_ids_in_candidates": (
                unique_route_covered
            ),
            "unique_route_arc_coverage_pct": (
                100 * unique_route_covered /
                len(unique_route_ids)
                if unique_route_ids else np.nan
            ),
            "observations_with_any_route_arc_candidate": (
                observations_with_route_candidate
            ),
            "total_candidate_observations": (
                candidates["obs_idx"].nunique()
            ),
            "candidate_rows_matching_any_route_arc": (
                len(route_candidate_rows)
            )
        })

result = pd.DataFrame(reports)

output_file = REPORT_DIR / "route_candidate_overlap.csv"
result.to_csv(output_file, index=False)

print("\nROUTE-CANDIDATE OVERLAP DIAGNOSTIC")
print("=================================")
print("Trips checked:", len(result))
print("Total route entries:", result["route_entries"].sum())

print(
    "Average unique route arc coverage:",
    round(result["unique_route_arc_coverage_pct"].mean(), 2),
    "%"
)

print(
    "Average route entries present in geometry:",
    round(result["route_entries_in_geometry_pct"].mean(), 2),
    "%"
)

print("\nLowest route arc candidate coverage:")
print(
    result.nsmallest(10, "unique_route_arc_coverage_pct")[
        ["trip_id", "unique_route_arc_coverage_pct",
         "observations_with_any_route_arc_candidate",
         "total_candidate_observations"]
    ].to_string(index=False)
)

print("\nReport saved to:", output_file)
