
from pathlib import Path
import json

import numpy as np
import pandas as pd
import shapely
from shapely import LineString, Point, STRtree
from pyproj import Transformer

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_ROOT = Path(
    "/mnt/c/Users/sri07/Documents/Student folder/Student folder"
)

GEOMETRY_DIR = Path(
    "member1_preprocessing/outputs/road_geometry"
)

OUTPUT_DIR = Path(
    "member1_preprocessing/outputs/candidates"
)

REPORT_DIR = Path(
    "member1_preprocessing/outputs/reports"
)

RADIUS_M = 75
MAX_CANDIDATES = 10
GAP_THRESHOLD_S = 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

if int(shapely.__version__.split(".")[0]) < 2:
    raise RuntimeError(
        "This script requires Shapely 2.x. "
        "Upgrade using: pip install --upgrade shapely"
    )

with open("splits.json", "r") as f:
    splits = json.load(f)

partitions = {
    "train": splits["train"],
    "student_test": splits["student_test"]
}

trip_reports = []

# --------------------------------------------------
# PROCESS EACH TRIP
# --------------------------------------------------

for partition, trip_ids in partitions.items():

    for trip_id in trip_ids:

        print(f"Generating candidates: {partition}/{trip_id}")

        trip_dir = DATA_ROOT / partition / trip_id

        # Load GPS observations
        track = pd.read_csv(
            trip_dir / f"{trip_id}.track",
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude", "timestamp"]
        )

        # Load preprocessed road geometry
        geometry_file = GEOMETRY_DIR / f"{trip_id}.npz"

        road = np.load(geometry_file)

        arc_ids = road["arc_id"]
        from_nodes = road["from_node"]
        to_nodes = road["to_node"]

        x1 = road["x1"]
        y1 = road["y1"]
        x2 = road["x2"]
        y2 = road["y2"]

        lengths = road["length_m"]
        bearings = road["bearing_deg"]

        from_in = road["from_in_degree"]
        from_out = road["from_out_degree"]
        to_in = road["to_in_degree"]
        to_out = road["to_out_degree"]

        # Reconstruct the same per-trip UTM projection
        center_lon = track["longitude"].median()
        center_lat = track["latitude"].median()

        zone = int((center_lon + 180) // 6) + 1
        zone = min(max(zone, 1), 60)

        epsg = (
            32600 + zone if center_lat >= 0
            else 32700 + zone
        )

        transformer = Transformer.from_crs(
            "EPSG:4326",
            f"EPSG:{epsg}",
            always_xy=True
        )

        gps_x, gps_y = transformer.transform(
            track["longitude"].to_numpy(),
            track["latitude"].to_numpy()
        )

        gps_x = np.asarray(gps_x)
        gps_y = np.asarray(gps_y)

        # Build spatial index of road segments
        lines = np.empty(len(arc_ids), dtype=object)

        for i in range(len(arc_ids)):
            lines[i] = LineString([
                (x1[i], y1[i]),
                (x2[i], y2[i])
            ])

        tree = STRtree(lines)

        # Store candidate rows
        candidate_rows = []

        candidate_counts = []
        no_radius_match = 0

        timestamps = track["timestamp"].to_numpy()

        # Process every GPS observation
        for obs_idx in range(len(track)):

            point = Point(gps_x[obs_idx], gps_y[obs_idx])

            # Find road segments intersecting the 75m buffer
            nearby_indices = tree.query(
                point.buffer(RADIUS_M)
            )

            nearby_indices = np.asarray(
                nearby_indices,
                dtype=np.int64
            )

            # Calculate exact point-to-segment distances
            if len(nearby_indices) > 0:

                distances = np.array([
                    point.distance(lines[i])
                    for i in nearby_indices
                ])

                # Keep only candidates within the radius
                within = distances <= RADIUS_M

                nearby_indices = nearby_indices[within]
                distances = distances[within]

            # If no road is within 75m, retain nearest road
            # as a fallback and explicitly mark it.
            if len(nearby_indices) == 0:

                nearest_idx = int(tree.nearest(point))

                nearby_indices = np.array([nearest_idx])
                distances = np.array([
                    point.distance(lines[nearest_idx])
                ])

                no_radius_match += 1
                fallback_used = True

            else:
                fallback_used = False

            # Sort by exact distance and retain at most 10
            order = np.argsort(distances)[:MAX_CANDIDATES]

            nearby_indices = nearby_indices[order]
            distances = distances[order]

            candidate_counts.append(len(nearby_indices))

            # Calculate temporal gap from previous observation
            if obs_idx == 0:
                time_delta = np.nan
            else:
                time_delta = (
                    timestamps[obs_idx] -
                    timestamps[obs_idx - 1]
                )

            # Add one row for every candidate
            for rank, (road_idx, distance) in enumerate(
                zip(nearby_indices, distances), start=1
            ):

                line = lines[road_idx]

                # Fractional position along road segment
                projection_fraction = (
                    line.project(point) / line.length
                    if line.length > 0 else 0.0
                )

                projected_point = line.interpolate(
                    projection_fraction, normalized=True
                )

                candidate_rows.append({
                    "partition": partition,
                    "trip_id": trip_id,
                    "obs_idx": obs_idx,
                    "timestamp": timestamps[obs_idx],
                    "time_delta_s": time_delta,
                    "arc_id": int(arc_ids[road_idx]),
                    "from_node": int(from_nodes[road_idx]),
                    "to_node": int(to_nodes[road_idx]),
                    "candidate_rank": rank,
                    "distance_m": float(distance),
                    "within_radius": bool(distance <= RADIUS_M),
                    "fallback_used": fallback_used,
                    "projection_fraction": projection_fraction,
                    "projected_x": projected_point.x,
                    "projected_y": projected_point.y,
                    "arc_length_m": float(lengths[road_idx]),
                    "arc_bearing_deg": float(bearings[road_idx]),
                    "from_in_degree": int(from_in[road_idx]),
                    "from_out_degree": int(from_out[road_idx]),
                    "to_in_degree": int(to_in[road_idx]),
                    "to_out_degree": int(to_out[road_idx])
                })

        # Save this trip's candidate table
        candidates = pd.DataFrame(candidate_rows)

        output_file = OUTPUT_DIR / f"{trip_id}.csv"
        candidates.to_csv(output_file, index=False)

        trip_reports.append({
            "partition": partition,
            "trip_id": trip_id,
            "gps_observations": len(track),
            "candidate_rows": len(candidates),
            "observations_within_radius": (
                len(track) - no_radius_match
            ),
            "observations_without_radius_match": no_radius_match,
            "mean_candidates_per_observation": float(
                np.mean(candidate_counts)
            ),
            "maximum_candidates_per_observation": int(
                np.max(candidate_counts)
            ),
            "output_file": str(output_file)
        })

        # Release trip-specific objects before next trip
        del tree, lines, road, candidates

# --------------------------------------------------
# SAVE SUMMARY
# --------------------------------------------------

report = pd.DataFrame(trip_reports)

summary_file = REPORT_DIR / "candidate_generation_summary.csv"
report.to_csv(summary_file, index=False)

print("\nCANDIDATE GENERATION COMPLETE")
print("=============================")
print("Trips processed:", len(report))
print("GPS observations:", report["gps_observations"].sum())
print("Candidate rows:", report["candidate_rows"].sum())
print(
    "Observations with no road within radius:",
    report["observations_without_radius_match"].sum()
)
print(
    "Average candidates per observation:",
    round(
        report["candidate_rows"].sum() /
        report["gps_observations"].sum(), 2
    )
)
print("\nSummary saved to:", summary_file)
print("Candidate files saved in:", OUTPUT_DIR)
