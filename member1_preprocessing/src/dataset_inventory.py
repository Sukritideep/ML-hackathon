
from pathlib import Path
import json
import pandas as pd

# 1. Dataset location
DATA_ROOT = Path(
    "/mnt/c/Users/sri07/Documents/Student folder/Student folder"
)

# 2. Official splits file
SPLITS_FILE = Path("splits.json")

with open(SPLITS_FILE, "r") as file:
    splits = json.load(file)

# 3. Read only permitted training and student-test IDs
partitions = {
    "train": splits["train"],
    "student_test": splits["student_test"]
}

records = []

# 4. Inspect every permitted trip
for partition, trip_ids in partitions.items():

    for trip_id in trip_ids:

        trip_dir = DATA_ROOT / partition / trip_id

        track_file = trip_dir / f"{trip_id}.track"
        nodes_file = trip_dir / f"{trip_id}.nodes"
        arcs_file = trip_dir / f"{trip_id}.arcs"
        route_file = trip_dir / f"{trip_id}.route"

        record = {
            "partition": partition,
            "trip_id": trip_id,
            "gps_observations": None,
            "road_nodes": None,
            "road_arcs": None,
            "route_length": None,
            "missing_files": []
        }

        files = {
            "track": track_file,
            "nodes": nodes_file,
            "arcs": arcs_file,
            "route": route_file
        }

        for name, path in files.items():
            if not path.exists():
                record["missing_files"].append(name)

        # Count rows in each available file
        if track_file.exists():
            record["gps_observations"] = len(
                pd.read_csv(track_file, sep=r"\s+", header=None)
            )

        if nodes_file.exists():
            record["road_nodes"] = len(
                pd.read_csv(nodes_file, sep=r"\s+", header=None)
            )

        if arcs_file.exists():
            record["road_arcs"] = len(
                pd.read_csv(arcs_file, sep=r"\s+", header=None)
            )

        if route_file.exists():
            record["route_length"] = len(
                pd.read_csv(route_file, sep=r"\s+", header=None)
            )

        records.append(record)

# 5. Create inventory table
df = pd.DataFrame(records)

# 6. Display summary
print("\nDATASET INVENTORY")
print("=================")

print("Total trips inspected:", len(df))

print("\nPartition counts:")
print(df["partition"].value_counts())

print("\nTotal GPS observations:",
      df["gps_observations"].sum())

print("Total road nodes:",
      df["road_nodes"].sum())

print("Total road arcs:",
      df["road_arcs"].sum())

print("Total route entries:",
      df["route_length"].sum())

# 7. Save report
output_dir = Path("member1_preprocessing/outputs/reports")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "dataset_inventory.csv"
df.to_csv(output_file, index=False)

print("\nInventory saved to:", output_file)
