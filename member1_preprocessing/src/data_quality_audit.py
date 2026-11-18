
from pathlib import Path
import json
import numpy as np
import pandas as pd

# Dataset and split locations
DATA_ROOT = Path(
    "/mnt/c/Users/sri07/Documents/Student folder/Student folder"
)

with open("splits.json", "r") as f:
    splits = json.load(f)

partitions = {
    "train": splits["train"],
    "student_test": splits["student_test"]
}

audit_records = []

for partition, trip_ids in partitions.items():

    for trip_id in trip_ids:

        trip_dir = DATA_ROOT / partition / trip_id

        track_path = trip_dir / f"{trip_id}.track"
        nodes_path = trip_dir / f"{trip_id}.nodes"
        arcs_path = trip_dir / f"{trip_id}.arcs"
        route_path = trip_dir / f"{trip_id}.route"

        record = {
            "partition": partition,
            "trip_id": trip_id
        }

        # ---------------------------------
        # 1. GPS DATA QUALITY
        # ---------------------------------

        track = pd.read_csv(
            track_path, sep=r"\s+", header=None
        )

        lon = pd.to_numeric(track.iloc[:, 0], errors="coerce")
        lat = pd.to_numeric(track.iloc[:, 1], errors="coerce")
        timestamp = pd.to_numeric(track.iloc[:, 2], errors="coerce")

        gps_valid = (
            np.isfinite(lon) &
            np.isfinite(lat) &
            np.isfinite(timestamp)
        )

        valid_coordinates = (
            gps_valid &
            lon.between(-180, 180) &
            lat.between(-90, 90)
        )

        time_diffs = timestamp.diff()

        record["gps_rows"] = len(track)
        record["gps_non_numeric_or_missing"] = int((~gps_valid).sum())
        record["invalid_coordinates"] = int((~valid_coordinates).sum())
        record["duplicate_gps_rows"] = int(track.duplicated().sum())
        record["non_increasing_timestamps"] = int(
            (time_diffs <= 0).sum()
        )
        record["gaps_over_5_seconds"] = int(
            (time_diffs > 5).sum()
        )
        record["largest_time_gap_seconds"] = (
            float(time_diffs.max())
            if time_diffs.notna().any() else 0
        )

        # ---------------------------------
        # 2. ROAD NODE QUALITY
        # ---------------------------------

        nodes = pd.read_csv(
            nodes_path, sep=r"\s+", header=None
        )

        node_lon = pd.to_numeric(
            nodes.iloc[:, 0], errors="coerce"
        )
        node_lat = pd.to_numeric(
            nodes.iloc[:, 1], errors="coerce"
        )

        valid_nodes = (
            np.isfinite(node_lon) &
            np.isfinite(node_lat) &
            node_lon.between(-180, 180) &
            node_lat.between(-90, 90)
        )

        record["road_nodes"] = len(nodes)
        record["invalid_nodes"] = int((~valid_nodes).sum())

        # ---------------------------------
        # 3. ROAD ARC QUALITY
        # ---------------------------------

        arcs = pd.read_csv(
            arcs_path, sep=r"\s+", header=None
        )

        arc_values = arcs.apply(
            pd.to_numeric, errors="coerce"
        )

        valid_arc_values = np.isfinite(
            arc_values.to_numpy()
        ).all(axis=1)

        valid_arc_ids = (
            valid_arc_values &
            (arc_values[0] >= 0) &
            (arc_values[0] < len(nodes)) &
            (arc_values[1] >= 0) &
            (arc_values[1] < len(nodes)) &
            (arc_values[0] % 1 == 0) &
            (arc_values[1] % 1 == 0)
        )

        record["road_arcs"] = len(arcs)
        record["invalid_arcs"] = int((~valid_arc_ids).sum())

        # ---------------------------------
        # 4. REFERENCE ROUTE QUALITY
        # ---------------------------------

        route = pd.read_csv(
            route_path, sep=r"\s+", header=None
        ).iloc[:, 0]

        route_values = pd.to_numeric(
            route, errors="coerce"
        )

        valid_route_ids = (
            np.isfinite(route_values) &
            (route_values >= 0) &
            (route_values < len(arcs)) &
            (route_values % 1 == 0)
        )

        record["route_entries"] = len(route)
        record["invalid_route_arc_ids"] = int(
            (~valid_route_ids).sum()
        )

        # Check whether consecutive valid route arcs connect
        valid_route = route_values[valid_route_ids].astype(int).to_numpy()
        valid_arcs = arc_values.loc[valid_arc_ids].astype(int).to_numpy()

        connected_pairs = 0
        disconnected_pairs = 0

        for previous_id, next_id in zip(
            valid_route[:-1], valid_route[1:]
        ):
            previous_end = valid_arcs[previous_id, 1]
            next_start = valid_arcs[next_id, 0]

            if previous_end == next_start:
                connected_pairs += 1
            else:
                disconnected_pairs += 1

        record["connected_route_pairs"] = connected_pairs
        record["disconnected_route_pairs"] = disconnected_pairs

        audit_records.append(record)

# ---------------------------------
# 5. SAVE AUDIT REPORT
# ---------------------------------

audit_df = pd.DataFrame(audit_records)

output_dir = Path(
    "member1_preprocessing/outputs/reports"
)
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "data_quality_audit.csv"

audit_df.to_csv(output_file, index=False)

# ---------------------------------
# 6. DISPLAY SUMMARY
# ---------------------------------

print("\nDATA QUALITY AUDIT")
print("===================")

columns_to_sum = [
    "gps_non_numeric_or_missing",
    "invalid_coordinates",
    "duplicate_gps_rows",
    "non_increasing_timestamps",
    "gaps_over_5_seconds",
    "invalid_nodes",
    "invalid_arcs",
    "invalid_route_arc_ids",
    "disconnected_route_pairs"
]

for column in columns_to_sum:
    print(f"{column}: {audit_df[column].sum()}")

print(
    "\nLargest GPS time gap:",
    audit_df["largest_time_gap_seconds"].max(),
    "seconds"
)

print("\nAudit saved to:", output_file)
