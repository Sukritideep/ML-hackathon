
from pathlib import Path
import json
import pandas as pd

DATA_ROOT = Path(
    "/mnt/c/Users/sri07/Documents/Student folder/Student folder"
)

with open("splits.json", "r") as f:
    splits = json.load(f)

partitions = {
    "train": splits["train"],
    "student_test": splits["student_test"]
}

gap_records = []
route_records = []

for partition, trip_ids in partitions.items():

    for trip_id in trip_ids:

        trip_dir = DATA_ROOT / partition / trip_id

        # --------------------------------
        # 1. Inspect GPS timestamp gaps
        # --------------------------------

        track_path = trip_dir / f"{trip_id}.track"

        track = pd.read_csv(
            track_path, sep=r"\s+", header=None,
            names=["longitude", "latitude", "timestamp"]
        )

        track["time_gap"] = track["timestamp"].diff()

        large_gaps = track.index[track["time_gap"] > 5]

        for idx in large_gaps:

            previous = track.iloc[idx - 1]
            current = track.iloc[idx]

            gap_records.append({
                "partition": partition,
                "trip_id": trip_id,
                "previous_observation": idx - 1,
                "current_observation": idx,
                "previous_timestamp": previous["timestamp"],
                "current_timestamp": current["timestamp"],
                "gap_seconds": current["time_gap"],
                "previous_longitude": previous["longitude"],
                "previous_latitude": previous["latitude"],
                "current_longitude": current["longitude"],
                "current_latitude": current["latitude"]
            })

        # --------------------------------
        # 2. Inspect disconnected routes
        # --------------------------------

        arcs_path = trip_dir / f"{trip_id}.arcs"
        route_path = trip_dir / f"{trip_id}.route"

        arcs = pd.read_csv(
            arcs_path, sep=r"\s+", header=None
        ).to_numpy(dtype=int)

        route = pd.read_csv(
            route_path, sep=r"\s+", header=None
        ).iloc[:, 0].to_numpy(dtype=int)

        for i in range(len(route) - 1):

            previous_arc_id = route[i]
            next_arc_id = route[i + 1]

            previous_end = arcs[previous_arc_id, 1]
            next_start = arcs[next_arc_id, 0]

            if previous_end != next_start:

                route_records.append({
                    "partition": partition,
                    "trip_id": trip_id,
                    "previous_route_position": i,
                    "next_route_position": i + 1,
                    "previous_arc_id": previous_arc_id,
                    "next_arc_id": next_arc_id,
                    "previous_end_node": previous_end,
                    "next_start_node": next_start
                })

# --------------------------------
# 3. Save detailed reports
# --------------------------------

output_dir = Path(
    "member1_preprocessing/outputs/reports"
)
output_dir.mkdir(parents=True, exist_ok=True)

gaps_df = pd.DataFrame(gap_records)
routes_df = pd.DataFrame(route_records)

gaps_df.to_csv(
    output_dir / "large_timestamp_gaps.csv",
    index=False
)

routes_df.to_csv(
    output_dir / "disconnected_route_pairs.csv",
    index=False
)

# --------------------------------
# 4. Print summary
# --------------------------------

print("\nLARGE TIMESTAMP GAPS")
print("====================")
print("Total:", len(gaps_df))

print("\nTop 15 largest gaps:")
print(
    gaps_df.nlargest(15, "gap_seconds")[
        ["trip_id", "gap_seconds",
         "previous_timestamp", "current_timestamp"]
    ].to_string(index=False)
)

print("\nDISCONNECTED ROUTE PAIRS")
print("========================")
print("Total:", len(routes_df))

print("\nAffected trips:")
if len(routes_df):
    print(
        routes_df.groupby("trip_id").size().to_string()
    )
else:
    print("None")

print("\nDetailed reports saved in:", output_dir)
