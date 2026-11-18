from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from pyproj import Transformer

from shapely.geometry import Point, LineString

# ==================================================
# CONFIGURATION
# ==================================================

DATA_ROOT = Path(
    "/mnt/c/Users/sri07/Documents/Student folder/Student folder"
)

FEATURE_DIR = Path(
    "member1_preprocessing/outputs/features"
)

OUTPUT_ROOT = Path(
    "member1_preprocessing/outputs/labeled_candidates"
)

REPORT_DIR = Path(
    "member1_preprocessing/outputs/reports"
)

TRAIN_OUTPUT = OUTPUT_ROOT / "train"
TEST_OUTPUT = OUTPUT_ROOT / "student_test_reference"

TRAIN_OUTPUT.mkdir(parents=True, exist_ok=True)
TEST_OUTPUT.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Alignment settings
ALIGNMENT_WINDOW = 20
MAX_ALIGNMENT_DISTANCE_M = 100.0

with open("splits.json", "r") as f:
    splits = json.load(f)

partitions = {
    "train": splits["train"],
    "student_test": splits["student_test"]
}

trip_reports = []

# ==================================================
# PROCESS EACH TRIP
# ==================================================

for partition, trip_ids in partitions.items():

    for trip_id in trip_ids:

        print(f"Aligning labels: {partition}/{trip_id}")

        trip_dir = DATA_ROOT / partition / trip_id

        # --------------------------------------------------
        # 1. LOAD RAW DATA
        # --------------------------------------------------

        track = pd.read_csv(
            trip_dir / f"{trip_id}.track",
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude", "timestamp"]
        )

        nodes = pd.read_csv(
            trip_dir / f"{trip_id}.nodes",
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude"]
        )

        arcs = pd.read_csv(
            trip_dir / f"{trip_id}.arcs",
            sep=r"\s+",
            header=None,
            names=["from_node", "to_node"]
        )

        route = pd.read_csv(
            trip_dir / f"{trip_id}.route",
            sep=r"\s+",
            header=None,
            names=["arc_id"]
        )

        features = pd.read_csv(
            FEATURE_DIR / f"{trip_id}.csv"
        )

        # --------------------------------------------------
        # 2. PROJECT GPS AND ROAD NODES INTO METRES
        # --------------------------------------------------

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

        gps_x = np.asarray(gps_x, dtype=float)
        gps_y = np.asarray(gps_y, dtype=float)

        node_x, node_y = transformer.transform(
            nodes["longitude"].to_numpy(),
            nodes["latitude"].to_numpy()
        )

        node_x = np.asarray(node_x, dtype=float)
        node_y = np.asarray(node_y, dtype=float)

        # --------------------------------------------------
        # 3. BUILD REFERENCE ROUTE GEOMETRIES
        # --------------------------------------------------

        route_ids_original = route["arc_id"].to_numpy(
            dtype=np.int64
        )

        from_all = arcs["from_node"].to_numpy(
            dtype=np.int64
        )

        to_all = arcs["to_node"].to_numpy(
            dtype=np.int64
        )

        route_valid = (
            (route_ids_original >= 0) &
            (route_ids_original < len(arcs))
        )

        valid_route_ids = route_ids_original[route_valid]

        route_from = from_all[valid_route_ids]
        route_to = to_all[valid_route_ids]

        endpoint_valid = (
            (route_from >= 0) &
            (route_to >= 0) &
            (route_from < len(nodes)) &
            (route_to < len(nodes))
        )

        valid_route_ids = valid_route_ids[endpoint_valid]
        route_from = route_from[endpoint_valid]
        route_to = route_to[endpoint_valid]

        if len(valid_route_ids) == 0:
            raise ValueError(
                f"No valid route geometries for trip {trip_id}"
            )

        route_lines = []

        for i in range(len(valid_route_ids)):

            route_lines.append(
                LineString([
                    (node_x[route_from[i]], node_y[route_from[i]]),
                    (node_x[route_to[i]], node_y[route_to[i]])
                ])
            )

        route_lengths = np.array(
            [line.length for line in route_lines],
            dtype=float
        )

        # Cumulative route distance at the end of each arc
        route_cumulative = np.cumsum(route_lengths)

        total_route_distance = float(
            route_cumulative[-1]
        )

        # --------------------------------------------------
        # 4. CALCULATE GPS CUMULATIVE MOVEMENT DISTANCE
        # --------------------------------------------------

        n_observations = len(track)

        if n_observations < 1:
            raise ValueError(f"Empty GPS track: {trip_id}")

        if n_observations > 1:

            gps_step_distance = np.hypot(
                np.diff(gps_x),
                np.diff(gps_y)
            )

            gps_cumulative = np.concatenate([
                [0.0],
                np.cumsum(gps_step_distance)
            ])

        else:
            gps_cumulative = np.array([0.0])

        total_gps_distance = float(
            gps_cumulative[-1]
        )

        # Use distance progress when possible.
        # If GPS movement is zero, fall back to elapsed-time
        # progress; if time is also constant, use row progress.
        if total_gps_distance > 0.01:

            progress = (
                gps_cumulative / total_gps_distance
            )

        elif (
            track["timestamp"].iloc[-1] >
            track["timestamp"].iloc[0]
        ):

            progress = (
                (
                    track["timestamp"].to_numpy() -
                    track["timestamp"].iloc[0]
                ) /
                (
                    track["timestamp"].iloc[-1] -
                    track["timestamp"].iloc[0]
                )
            )

        else:

            progress = np.linspace(
                0.0, 1.0, n_observations
            )

        expected_route_distance = (
            progress * total_route_distance
        )

        # --------------------------------------------------
        # 5. ALIGN EACH GPS OBSERVATION TO ROUTE SEQUENCE
        # --------------------------------------------------

        aligned_route_index = np.full(
            n_observations, -1, dtype=int
        )

        aligned_arc_id = np.full(
            n_observations, -1, dtype=int
        )

        alignment_distance = np.full(
            n_observations, np.nan, dtype=float
        )

        previous_index = 0

        for obs_idx in range(n_observations):

            point = Point(
                gps_x[obs_idx],
                gps_y[obs_idx]
            )

            expected_distance = (
                expected_route_distance[obs_idx]
            )

            predicted_index = int(
                np.searchsorted(
                    route_cumulative,
                    expected_distance,
                    side="left"
                )
            )

            predicted_index = min(
                predicted_index,
                len(route_lines) - 1
            )

            # Search a local chronological route window
            lower = max(
                0,
                predicted_index - ALIGNMENT_WINDOW,
                previous_index - 2
            )

            upper = min(
                len(route_lines),
                predicted_index + ALIGNMENT_WINDOW + 1
            )

            # Safety if the lower bound passes the upper bound
            if lower >= upper:
                lower = max(
                    0,
                    min(predicted_index, len(route_lines) - 1)
                )
                upper = min(
                    len(route_lines),
                    lower + 1
                )

            best_index = -1
            best_cost = float("inf")
            best_distance = float("inf")

            for route_idx in range(lower, upper):

                distance = point.distance(
                    route_lines[route_idx]
                )

                # Tiny tie-breaker prefers expected progress
                cost = (
                    distance +
                    0.001 * abs(route_idx - predicted_index)
                )

                if cost < best_cost:
                    best_cost = cost
                    best_distance = distance
                    best_index = route_idx

            if best_index >= 0:

                aligned_route_index[obs_idx] = best_index

                aligned_arc_id[obs_idx] = int(
                    valid_route_ids[best_index]
                )

                alignment_distance[obs_idx] = best_distance

                previous_index = best_index

        # --------------------------------------------------
        # 6. CHECK WHETHER THE ALIGNED ARC IS A CANDIDATE
        # --------------------------------------------------

        aligned = pd.DataFrame({
            "obs_idx": np.arange(n_observations),
            "aligned_route_index": aligned_route_index,
            "aligned_route_arc_id": aligned_arc_id,
            "alignment_distance_m": alignment_distance,
            "route_progress_fraction": progress
        })

        aligned["alignment_within_100m"] = (
            aligned["alignment_distance_m"] <=
            MAX_ALIGNMENT_DISTANCE_M
        )

        candidate_sets = (
            features.groupby("obs_idx")["arc_id"]
            .apply(set)
            .to_dict()
        )

        aligned["candidate_hit"] = aligned.apply(
            lambda row: (
                int(row["aligned_route_arc_id"])
                in candidate_sets.get(int(row["obs_idx"]), set())
                if row["aligned_route_arc_id"] >= 0
                else False
            ),
            axis=1
        )

        aligned["candidate_miss"] = (
            aligned["alignment_within_100m"] &
            ~aligned["candidate_hit"]
        )

        # A label is usable only if:
        # 1. A route arc was aligned.
        # 2. Alignment distance is <= 100 metres.
        # 3. The aligned route arc exists in the candidate set.
        aligned["label_valid"] = (
            aligned["alignment_within_100m"] &
            aligned["candidate_hit"]
        )

        # --------------------------------------------------
        # 7. ATTACH LABELS TO CANDIDATE FEATURE ROWS
        # --------------------------------------------------

        labelled = features.merge(
            aligned[
                [
                    "obs_idx",
                    "aligned_route_index",
                    "aligned_route_arc_id",
                    "alignment_distance_m",
                    "route_progress_fraction",
                    "alignment_within_100m",
                    "candidate_hit",
                    "candidate_miss",
                    "label_valid"
                ]
            ],
            on="obs_idx",
            how="left",
            validate="many_to_one"
        )

        labelled["target_label"] = (
            (
                labelled["arc_id"] ==
                labelled["aligned_route_arc_id"]
            ) &
            labelled["label_valid"]
        ).astype("int8")

        # Preserve whether the candidate row is usable for training
        labelled["training_eligible"] = (
            labelled["label_valid"]
        )

        # --------------------------------------------------
        # 8. SAVE OUTPUTS SEPARATELY BY PARTITION
        # --------------------------------------------------

        if partition == "train":

            output_file = (
                TRAIN_OUTPUT / f"{trip_id}.csv"
            )

        else:

            output_file = (
                TEST_OUTPUT / f"{trip_id}.csv"
            )

        labelled.to_csv(
            output_file,
            index=False
        )

        # Save per-observation alignment report
        alignment_dir = OUTPUT_ROOT / "alignment"
        alignment_dir.mkdir(parents=True, exist_ok=True)

        aligned.to_csv(
            alignment_dir / f"{trip_id}.csv",
            index=False
        )

        # --------------------------------------------------
        # 9. CALCULATE ACTUAL CANDIDATE RECALL
        # --------------------------------------------------

        confident_count = int(
            aligned["alignment_within_100m"].sum()
        )

        candidate_hits = int(
            aligned["candidate_hit"].sum()
        )

        candidate_misses = int(
            aligned["candidate_miss"].sum()
        )

        valid_labels = int(
            aligned["label_valid"].sum()
        )

        candidate_recall = (
            100.0 * candidate_hits / confident_count
            if confident_count > 0
            else np.nan
        )

        trip_reports.append({
            "partition": partition,
            "trip_id": trip_id,
            "gps_observations": n_observations,
            "route_entries_original": len(route_ids_original),
            "valid_route_entries": len(valid_route_ids),
            "invalid_route_entries": (
                len(route_ids_original) - len(valid_route_ids)
            ),
            "aligned_observations_within_100m": confident_count,
            "observations_with_candidate_hit": candidate_hits,
            "candidate_misses": candidate_misses,
            "valid_label_observations": valid_labels,
            "alignment_coverage_pct": (
                100.0 * confident_count / n_observations
            ),
            "candidate_recall_pct": candidate_recall,
            "label_coverage_pct": (
                100.0 * valid_labels / n_observations
            ),
            "mean_alignment_distance_m": float(
                aligned["alignment_distance_m"].mean()
            ),
            "output_file": str(output_file)
        })

        print(
            f"  Label coverage: "
            f"{100.0 * valid_labels / n_observations:.2f}%"
        )

# ==================================================
# 10. SAVE FINAL REPORT
# ==================================================

summary = pd.DataFrame(trip_reports)

summary_file = (
    REPORT_DIR / "gps_route_alignment_and_recall.csv"
)

summary.to_csv(summary_file, index=False)

print("\nGPS-TO-ROUTE ALIGNMENT COMPLETE")
print("================================")

print("Trips processed:", len(summary))

print(
    "GPS observations:",
    summary["gps_observations"].sum()
)

print(
    "Observations aligned within 100m:",
    summary["aligned_observations_within_100m"].sum()
)

print(
    "Observations with candidate hit:",
    summary["observations_with_candidate_hit"].sum()
)

print(
    "Candidate misses:",
    summary["candidate_misses"].sum()
)

print(
    "Valid label observations:",
    summary["valid_label_observations"].sum()
)

print(
    "\nMean alignment coverage:",
    round(summary["alignment_coverage_pct"].mean(), 2),
    "%"
)

print(
    "Mean per-trip candidate recall:",
    round(summary["candidate_recall_pct"].mean(), 2),
    "%"
)

print(
    "Mean label coverage:",
    round(summary["label_coverage_pct"].mean(), 2),
    "%"
)

print("\nLowest candidate recall trips:")
print(
    summary.nsmallest(10, "candidate_recall_pct")[
        [
            "trip_id",
            "candidate_recall_pct",
            "alignment_coverage_pct",
            "label_coverage_pct"
        ]
    ].to_string(index=False)
)

print("\nFinal labelled training files:", TRAIN_OUTPUT)
print("Separate student-test reference files:", TEST_OUTPUT)
print("Per-observation alignment files:", OUTPUT_ROOT / "alignment")
print("Summary report:", summary_file)

print("\nIMPORTANT:")
print("Alignment is distance-guided and heuristic.")
print("Review alignment maps and low-coverage trips before training.")
print("Use only the train directory for supervised model fitting.")
print("Keep student-test reference labels separate for evaluation.")