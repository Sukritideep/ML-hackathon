
from pathlib import Path
import json

import numpy as np
import pandas as pd
from pyproj import Transformer

DATA_ROOT = Path(
    "/mnt/c/Users/sri07/Documents/Student folder/Student folder"
)

GEOMETRY_DIR = Path(
    "member1_preprocessing/outputs/road_geometry"
)

CANDIDATE_DIR = Path(
    "member1_preprocessing/outputs/candidates"
)

OUTPUT_DIR = Path(
    "member1_preprocessing/outputs/features"
)

REPORT_DIR = Path(
    "member1_preprocessing/outputs/reports"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

GAP_THRESHOLD_S = 5

with open("splits.json", "r") as f:
    splits = json.load(f)

partitions = {
    "train": splits["train"],
    "student_test": splits["student_test"]
}

reports = []

for partition, trip_ids in partitions.items():

    for trip_id in trip_ids:

        print(f"Extracting features: {partition}/{trip_id}")

        trip_dir = DATA_ROOT / partition / trip_id

        # Load GPS observations
        track = pd.read_csv(
            trip_dir / f"{trip_id}.track",
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude", "timestamp"]
        )

        # Load candidate table
        candidates = pd.read_csv(
            CANDIDATE_DIR / f"{trip_id}.csv"
        )

        # Load road geometry
        road = np.load(
            GEOMETRY_DIR / f"{trip_id}.npz"
        )

        # Recreate the per-trip UTM projection
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

        timestamps = track["timestamp"].to_numpy()
        n = len(track)

        # ------------------------------------------
        # TEMPORAL AND MOVEMENT FEATURES
        # ------------------------------------------

        time_delta = np.full(n, np.nan)
        gps_distance = np.full(n, np.nan)
        gps_speed = np.full(n, np.nan)
        gps_heading = np.full(n, np.nan)
        long_gap = np.zeros(n, dtype=bool)

        if n > 1:

            time_delta[1:] = np.diff(timestamps)

            dx = np.diff(gps_x)
            dy = np.diff(gps_y)

            distances = np.hypot(dx, dy)

            valid_interval = (
                (time_delta[1:] > 0) &
                (time_delta[1:] <= GAP_THRESHOLD_S)
            )

            gps_distance[1:] = np.where(
                valid_interval, distances, np.nan
            )

            gps_speed[1:] = np.where(
                valid_interval,
                distances / time_delta[1:],
                np.nan
            )

            headings = (
                np.degrees(np.arctan2(dx, dy)) + 360
            ) % 360

            gps_heading[1:] = np.where(
                valid_interval & (distances > 0),
                headings,
                np.nan
            )

            long_gap[1:] = (
                time_delta[1:] > GAP_THRESHOLD_S
            )

        # ------------------------------------------
        # ROAD ENDPOINT COORDINATES
        # ------------------------------------------

        road_table = pd.DataFrame({
            "arc_id": road["arc_id"],
            "x1": road["x1"],
            "y1": road["y1"],
            "x2": road["x2"],
            "y2": road["y2"]
        })

        # Attach road endpoint coordinates to candidates
        features = candidates.merge(
            road_table,
            on="arc_id",
            how="left",
            validate="many_to_one"
        )

        # Match each candidate row to its GPS observation
        obs = features["obs_idx"].to_numpy(dtype=int)

        px = gps_x[obs]
        py = gps_y[obs]

        # ------------------------------------------
        # SPATIAL FEATURES
        # ------------------------------------------

        dx_from = px - features["x1"].to_numpy()
        dy_from = py - features["y1"].to_numpy()

        dx_to = px - features["x2"].to_numpy()
        dy_to = py - features["y2"].to_numpy()

        distance_from_node = np.hypot(
            dx_from, dy_from
        )

        distance_to_node = np.hypot(
            dx_to, dy_to
        )

        features["distance_to_from_node_m"] = (
            distance_from_node
        )

        features["distance_to_to_node_m"] = (
            distance_to_node
        )

        features["distance_to_nearest_endpoint_m"] = (
            np.minimum(
                distance_from_node,
                distance_to_node
            )
        )

        features["candidate_progress_m"] = (
            features["projection_fraction"] *
            features["arc_length_m"]
        )

        # ------------------------------------------
        # MOVEMENT FEATURES
        # ------------------------------------------

        features["gps_distance_m"] = gps_distance[obs]
        features["gps_speed_mps"] = gps_speed[obs]
        features["gps_speed_kmh"] = (
            gps_speed[obs] * 3.6
        )

        features["gps_heading_deg"] = gps_heading[obs]
        features["time_delta_s"] = time_delta[obs]
        features["long_gap"] = long_gap[obs]

        # ------------------------------------------
        # DIRECTION FEATURES
        # ------------------------------------------

        angle_difference = (
            gps_heading[obs] -
            features["arc_bearing_deg"].to_numpy() +
            180
        ) % 360 - 180

        features["direction_difference_deg"] = (
            angle_difference
        )

        angle_rad = np.radians(angle_difference)

        features["direction_difference_sin"] = (
            np.sin(angle_rad)
        )

        features["direction_difference_cos"] = (
            np.cos(angle_rad)
        )

        # Sine/cosine representation of road bearing
        bearing_rad = np.radians(
            features["arc_bearing_deg"].to_numpy()
        )

        features["arc_bearing_sin"] = np.sin(bearing_rad)
        features["arc_bearing_cos"] = np.cos(bearing_rad)

        # Sine/cosine representation of GPS heading
        heading_rad = np.radians(
            features["gps_heading_deg"].to_numpy()
        )

        features["gps_heading_sin"] = np.sin(heading_rad)
        features["gps_heading_cos"] = np.cos(heading_rad)

        # ------------------------------------------
        # SAVE FEATURE TABLE
        # ------------------------------------------

        output_file = OUTPUT_DIR / f"{trip_id}.csv"
        features.to_csv(output_file, index=False)

        reports.append({
            "partition": partition,
            "trip_id": trip_id,
            "gps_observations": n,
            "candidate_rows": len(features),
            "feature_columns": len(features.columns),
            "missing_gps_speed_rows": int(
                features["gps_speed_mps"].isna().sum()
            ),
            "long_gap_observations": int(
                long_gap.sum()
            ),
            "output_file": str(output_file)
        })

        del road, candidates, features

# ----------------------------------------------
# SUMMARY REPORT
# ----------------------------------------------

report = pd.DataFrame(reports)

report_file = REPORT_DIR / "feature_extraction_summary.csv"
report.to_csv(report_file, index=False)

print("\nFEATURE EXTRACTION COMPLETE")
print("============================")
print("Trips processed:", len(report))
print("Total candidate rows:", report["candidate_rows"].sum())
print("Feature columns per trip:",
      report["feature_columns"].min(),
      "to",
      report["feature_columns"].max())
print("Long-gap observations:",
      report["long_gap_observations"].sum())

print("\nFeature files saved in:", OUTPUT_DIR)
print("Summary saved to:", report_file)
