
from pathlib import Path
import json

import numpy as np
import pandas as pd
from pyproj import Transformer

DATA_ROOT = Path(
    "/mnt/c/Users/sri07/Documents/Student folder/Student folder"
)

OUTPUT_DIR = Path(
    "member1_preprocessing/outputs/road_geometry"
)
REPORT_DIR = Path(
    "member1_preprocessing/outputs/reports"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open("splits.json", "r") as f:
    splits = json.load(f)

partitions = {
    "train": splits["train"],
    "student_test": splits["student_test"]
}

records = []

for partition, trip_ids in partitions.items():

    for trip_id in trip_ids:

        print(f"Processing {partition}/{trip_id}...")

        trip_dir = DATA_ROOT / partition / trip_id

        # Load GPS observations to determine the local projection
        track = pd.read_csv(
            trip_dir / f"{trip_id}.track",
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude", "timestamp"]
        )

        center_lon = track["longitude"].median()
        center_lat = track["latitude"].median()

        zone = int((center_lon + 180) // 6) + 1
        zone = min(max(zone, 1), 60)

        if center_lat >= 0:
            epsg = 32600 + zone
        else:
            epsg = 32700 + zone

        transformer = Transformer.from_crs(
            "EPSG:4326",
            f"EPSG:{epsg}",
            always_xy=True
        )

        # Load road nodes
        nodes = pd.read_csv(
            trip_dir / f"{trip_id}.nodes",
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude"]
        )

        # Project all nodes into metres
        node_x, node_y = transformer.transform(
            nodes["longitude"].to_numpy(),
            nodes["latitude"].to_numpy()
        )

        node_x = np.asarray(node_x, dtype=np.float64)
        node_y = np.asarray(node_y, dtype=np.float64)

        # Load directed road arcs
        arcs = pd.read_csv(
            trip_dir / f"{trip_id}.arcs",
            sep=r"\s+",
            header=None,
            names=["from_node", "to_node"]
        )

        from_all = arcs["from_node"].to_numpy(dtype=np.int64)
        to_all = arcs["to_node"].to_numpy(dtype=np.int64)

        # Retain only arcs whose endpoint references are valid
        valid = (
            (from_all >= 0) &
            (to_all >= 0) &
            (from_all < len(nodes)) &
            (to_all < len(nodes))
        )

        original_arc_ids = np.flatnonzero(valid).astype(np.int64)

        from_node = from_all[valid]
        to_node = to_all[valid]

        # Get projected endpoints for each arc
        x1 = node_x[from_node]
        y1 = node_y[from_node]
        x2 = node_x[to_node]
        y2 = node_y[to_node]

        # Calculate segment length in metres
        dx = x2 - x1
        dy = y2 - y1

        length_m = np.hypot(dx, dy)

        # Bearing measured clockwise from north
        bearing_deg = (
            np.degrees(np.arctan2(dx, dy)) + 360
        ) % 360

        # Calculate directed incoming/outgoing node degrees
        incoming_degree = np.bincount(
            to_node,
            minlength=len(nodes)
        )

        outgoing_degree = np.bincount(
            from_node,
            minlength=len(nodes)
        )

        # Save endpoint connectivity for every retained arc
        arrays = {
            "arc_id": original_arc_ids,
            "from_node": from_node,
            "to_node": to_node,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "length_m": length_m,
            "bearing_deg": bearing_deg,
            "from_in_degree": incoming_degree[from_node],
            "from_out_degree": outgoing_degree[from_node],
            "to_in_degree": incoming_degree[to_node],
            "to_out_degree": outgoing_degree[to_node]
        }

        output_file = OUTPUT_DIR / f"{trip_id}.npz"

        np.savez_compressed(
            output_file,
            **arrays
        )

        records.append({
            "partition": partition,
            "trip_id": trip_id,
            "epsg": epsg,
            "node_count": len(nodes),
            "original_arc_count": len(arcs),
            "saved_arc_count": len(original_arc_ids),
            "invalid_arc_count": int((~valid).sum()),
            "self_loop_count": int(
                np.sum(from_node == to_node)
            ),
            "zero_length_count": int(
                np.sum(length_m == 0)
            ),
            "median_length_m": float(
                np.median(length_m)
            ),
            "maximum_length_m": float(
                np.max(length_m)
            ),
            "geometry_file": str(output_file)
        })

# Save a manifest describing all generated files
manifest = pd.DataFrame(records)

manifest_file = REPORT_DIR / "road_geometry_manifest.csv"
manifest.to_csv(manifest_file, index=False)

print("\nROAD GEOMETRY BUILD COMPLETE")
print("============================")
print("Trips processed:", len(manifest))
print("Original arcs:", manifest["original_arc_count"].sum())
print("Saved valid arcs:", manifest["saved_arc_count"].sum())
print("Invalid arcs skipped:", manifest["invalid_arc_count"].sum())
print("Self-loop arcs retained:", manifest["self_loop_count"].sum())
print("Zero-length arcs retained:", manifest["zero_length_count"].sum())
print("\nManifest saved to:", manifest_file)
print("Geometry files saved in:", OUTPUT_DIR)
