
from pathlib import Path
import json

import numpy as np
import pandas as pd
from pyproj import Transformer

DATA_ROOT = Path(
    "/mnt/c/Users/sri07/Documents/Student folder/Student folder"
)

with open("splits.json", "r") as f:
    splits = json.load(f)

partitions = {
    "train": splits["train"],
    "student_test": splits["student_test"]
}

records = []

for partition, trip_ids in partitions.items():

    for trip_id in trip_ids:

        trip_dir = DATA_ROOT / partition / trip_id

        # Load road nodes
        nodes = pd.read_csv(
            trip_dir / f"{trip_id}.nodes",
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude"]
        )

        # Load directed road arcs
        arcs = pd.read_csv(
            trip_dir / f"{trip_id}.arcs",
            sep=r"\s+",
            header=None,
            names=["from_node", "to_node"]
        )

        # Choose a local UTM projection using GPS centre
        track = pd.read_csv(
            trip_dir / f"{trip_id}.track",
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude", "timestamp"]
        )

        center_lon = track["longitude"].median()
        center_lat = track["latitude"].median()

        zone = int((center_lon + 180) // 6) + 1

        if center_lat >= 0:
            epsg = 32600 + zone
        else:
            epsg = 32700 + zone

        transformer = Transformer.from_crs(
            "EPSG:4326",
            f"EPSG:{epsg}",
            always_xy=True
        )

        # Project node coordinates into metres
        x, y = transformer.transform(
            nodes["longitude"].to_numpy(),
            nodes["latitude"].to_numpy()
        )

        x = np.asarray(x)
        y = np.asarray(y)

        from_ids = arcs["from_node"].to_numpy(dtype=np.int64)
        to_ids = arcs["to_node"].to_numpy(dtype=np.int64)

        # Validate arc endpoint references
        valid = (
            (from_ids >= 0) &
            (to_ids >= 0) &
            (from_ids < len(nodes)) &
            (to_ids < len(nodes))
        )

        valid_from = from_ids[valid]
        valid_to = to_ids[valid]

        # Calculate projected arc lengths
        dx = x[valid_to] - x[valid_from]
        dy = y[valid_to] - y[valid_from]

        lengths = np.hypot(dx, dy)

        # Calculate directed arc bearings
        bearings = (
            np.degrees(np.arctan2(dx, dy)) + 360
        ) % 360

        # Calculate incoming and outgoing node degrees
        out_degree = np.bincount(
            valid_from,
            minlength=len(nodes)
        )

        in_degree = np.bincount(
            valid_to,
            minlength=len(nodes)
        )

        total_degree = in_degree + out_degree

        records.append({
            "partition": partition,
            "trip_id": trip_id,
            "suggested_epsg": epsg,
            "node_count": len(nodes),
            "arc_count": len(arcs),
            "valid_arc_count": int(valid.sum()),
            "invalid_arc_count": int((~valid).sum()),
            "self_loop_count": int(
                np.sum(valid_from == valid_to)
            ),
            "zero_length_arc_count": int(
                np.sum(lengths == 0)
            ),
            "isolated_node_count": int(
                np.sum(total_degree == 0)
            ),
            "average_arc_length_m": float(
                np.mean(lengths)
            ) if len(lengths) else np.nan,
            "median_arc_length_m": float(
                np.median(lengths)
            ) if len(lengths) else np.nan,
            "p95_arc_length_m": float(
                np.percentile(lengths, 95)
            ) if len(lengths) else np.nan,
            "maximum_arc_length_m": float(
                np.max(lengths)
            ) if len(lengths) else np.nan,
            "average_node_degree": float(
                np.mean(total_degree)
            )
        })

df = pd.DataFrame(records)

output_dir = Path(
    "member1_preprocessing/outputs/reports"
)
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "road_network_audit.csv"
df.to_csv(output_file, index=False)

print("\nROAD NETWORK AUDIT")
print("==================")
print("Trips inspected:", len(df))
print("Total nodes:", df["node_count"].sum())
print("Total arcs:", df["arc_count"].sum())
print("Invalid arc references:", df["invalid_arc_count"].sum())
print("Self-loop arcs:", df["self_loop_count"].sum())
print("Zero-length arcs:", df["zero_length_arc_count"].sum())

print("\nArc length statistics (metres):")
print(df[
    [
        "average_arc_length_m",
        "median_arc_length_m",
        "p95_arc_length_m",
        "maximum_arc_length_m"
    ]
].describe().round(2).to_string())

print("\nTrips with largest maximum arc lengths:")
print(
    df.nlargest(10, "maximum_arc_length_m")[
        ["trip_id", "maximum_arc_length_m"]
    ].to_string(index=False)
)

print("\nReport saved to:", output_file)
