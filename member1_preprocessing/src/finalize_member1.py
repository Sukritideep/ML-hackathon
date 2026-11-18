
from pathlib import Path
import json

import pandas as pd
import folium

# --------------------------------------------------
# PATHS
# --------------------------------------------------

DATA_ROOT = Path(
    "/mnt/c/Users/sri07/Documents/Student folder/Student folder"
)

BASE = Path("member1_preprocessing/outputs")

REPORT_DIR = BASE / "reports"
MAP_DIR = BASE / "maps"

MAP_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD TEAM SPLITS
# --------------------------------------------------

with open("splits.json", "r") as f:
    splits = json.load(f)

trip_partition = {}

for partition in ["train", "student_test"]:
    for trip_id in splits[partition]:
        trip_partition[str(trip_id).zfill(8)] = partition

# --------------------------------------------------
# LOAD EXISTING REPORTS
# --------------------------------------------------

report_files = [
    "dataset_inventory.csv",
    "per_trip_crs_audit.csv",
    "road_network_audit.csv",
    "road_geometry_manifest.csv",
    "candidate_generation_summary.csv",
    "candidate_validation.csv",
    "feature_extraction_summary.csv",
    "route_candidate_overlap.csv",
    "final_preprocessing_validation.csv"
]

reports = []

for filename in report_files:

    file_path = REPORT_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required report not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        dtype={"trip_id": str}
    )

    if "trip_id" not in df.columns:
        raise ValueError(
            f"trip_id column missing in {filename}"
        )

    df["trip_id"] = df["trip_id"].str.zfill(8)

    # Avoid duplicate columns when combining reports
    reports.append(df)

# --------------------------------------------------
# CREATE CONSOLIDATED REPORT
# --------------------------------------------------

combined = reports[0].copy()

for current in reports[1:]:

    # Remove columns already present, except trip_id
    new_columns = [
        col for col in current.columns
        if col == "trip_id" or col not in combined.columns
    ]

    current = current[new_columns]

    combined = combined.merge(
        current,
        on="trip_id",
        how="outer"
    )

# FIX: Replace existing partition column instead of inserting
combined["partition"] = combined["trip_id"].map(
    trip_partition
)

columns = ["partition"] + [
    col for col in combined.columns
    if col != "partition"
]

combined = combined[columns]

combined_file = (
    REPORT_DIR / "member1_consolidated_report.csv"
)

combined.to_csv(combined_file, index=False)

# --------------------------------------------------
# GENERATE FOLIUM MAPS
# --------------------------------------------------

example_trips = [
    "00000001",
    "00000054",
    "00000093"
]

for trip_id in example_trips:

    partition = trip_partition[trip_id]

    trip_dir = DATA_ROOT / partition / trip_id

    print(f"Generating map for {trip_id}...")

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

    center_lat = float(track["latitude"].median())
    center_lon = float(track["longitude"].median())

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="OpenStreetMap"
    )

    # GPS trajectory
    gps_coordinates = list(zip(
        track["latitude"],
        track["longitude"]
    ))

    folium.PolyLine(
        gps_coordinates,
        color="blue",
        weight=4,
        opacity=0.8,
        tooltip="Observed GPS trajectory"
    ).add_to(fmap)

    # Start marker
    folium.Marker(
        gps_coordinates[0],
        tooltip="GPS start",
        icon=folium.Icon(color="green")
    ).add_to(fmap)

    # End marker
    folium.Marker(
        gps_coordinates[-1],
        tooltip="GPS end",
        icon=folium.Icon(color="red")
    ).add_to(fmap)

    # Reference route
    route_ids = route["arc_id"].astype(int).tolist()

    route_segments = []
    current_segment = []

    for arc_id in route_ids:

        if arc_id < 0 or arc_id >= len(arcs):
            continue

        from_node = int(arcs.iloc[arc_id]["from_node"])
        to_node = int(arcs.iloc[arc_id]["to_node"])

        if (
            from_node < 0 or to_node < 0 or
            from_node >= len(nodes) or
            to_node >= len(nodes)
        ):
            continue

        start = [
            float(nodes.iloc[from_node]["latitude"]),
            float(nodes.iloc[from_node]["longitude"])
        ]

        end = [
            float(nodes.iloc[to_node]["latitude"]),
            float(nodes.iloc[to_node]["longitude"])
        ]

        if not current_segment:
            current_segment = [start, end]

        elif current_segment[-1] == start:
            current_segment.append(end)

        else:
            if len(current_segment) >= 2:
                route_segments.append(current_segment)

            current_segment = [start, end]

    if len(current_segment) >= 2:
        route_segments.append(current_segment)

    for segment in route_segments:

        folium.PolyLine(
            segment,
            color="orange",
            weight=4,
            opacity=0.8,
            tooltip="Reference route"
        ).add_to(fmap)

    folium.LayerControl().add_to(fmap)

    map_file = MAP_DIR / f"{trip_id}_trajectory_map.html"

    fmap.save(str(map_file))

    print("Saved:", map_file)

# --------------------------------------------------
# CREATE MEMBER 1 HANDOVER DOCUMENT
# --------------------------------------------------

candidates = pd.read_csv(
    REPORT_DIR / "candidate_generation_summary.csv"
)

route_overlap = pd.read_csv(
    REPORT_DIR / "route_candidate_overlap.csv"
)

total_observations = int(
    candidates["gps_observations"].sum()
)

total_candidate_rows = int(
    candidates["candidate_rows"].sum()
)

total_fallback = int(
    candidates[
        "observations_without_radius_match"
    ].sum()
)

average_candidates = (
    total_candidate_rows / total_observations
)

fallback_percentage = (
    100 * total_fallback / total_observations
)

average_unique_route_coverage = float(
    route_overlap[
        "unique_route_arc_coverage_pct"
    ].mean()
)

handover_text = f"""
# Member 1 — Data Preprocessing and Feature Engineering

## Project

PathSnap — GPS trajectory reconstruction using road-network map matching.

## Completed work

1. Dataset inspection.
2. GPS and road-network data-quality audits.
3. Timestamp-gap and route-connectivity analysis.
4. Per-trip CRS audit.
5. Road-network geometry audit.
6. Projected road geometry generation.
7. GPS-to-road candidate generation.
8. Candidate validation.
9. Spatial, temporal, movement and road-network feature extraction.
10. Final structural validation.
11. Folium trajectory map generation.
12. Consolidated report generation.

## Dataset totals

- Trips processed: 90
- GPS observations: {total_observations:,}
- Road nodes: 2,024,174
- Road arcs: 4,061,945
- Route entries: 34,813
- Candidate rows: {total_candidate_rows:,}
- Average candidates per observation: {average_candidates:.2f}
- Observations without a road within 75 metres: {total_fallback:,}
- Fallback percentage: {fallback_percentage:.2f}%
- Feature columns per candidate: 41

## Validation results

- Zero invalid road arc references.
- Zero duplicate candidate-observation pairs.
- Zero candidate rank errors.
- Zero invalid candidate distances.
- Zero invalid projection fractions.
- Zero negative GPS speeds.
- Candidate and feature row counts match.
- Average unique route-arc candidate overlap: {average_unique_route_coverage:.2f}%.

## Issues requiring attention

1. Trip 00000054 has an unusually high number of fallback observations.
2. Some trips have lower unique route-arc candidate overlap.
3. There are 1,260 GPS intervals greater than five seconds.
4. Fourteen route transitions were flagged as not directly connected.
5. There are 106 zero-length arcs and 24 self-loop arcs.

## Label alignment limitation

The reference .route files contain chronological road arc sequences.
They are not directly aligned one-to-one with GPS observations.

Unique route-arc candidate overlap is NOT GPS-level candidate recall.

A reliable chronological alignment between GPS observations and reference route arcs is still required before creating supervised training labels.

Do not treat the candidate rows as labelled training examples until this alignment is validated.

## Output folders

- outputs/road_geometry/
- outputs/candidates/
- outputs/features/
- outputs/reports/
- outputs/maps/

## Handover to Member 2

Member 2 can inspect the feature tables and prepare the model-training pipeline.

Supervised model training must wait until GPS-to-route label alignment is completed and observation-level candidate recall is measured.

## Handover to Member 3

Review route sequence continuity and help validate sequence alignment and transition handling.

## Reproducibility

All preprocessing scripts are stored in member1_preprocessing/src/.

The original raw dataset has not been modified.
"""

handover_file = Path(
    "member1_preprocessing/MEMBER1_HANDOVER.md"
)

handover_file.write_text(
    handover_text.strip() + "\n",
    encoding="utf-8"
)

# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\nMEMBER 1 HANDOVER PACKAGE CREATED")
print("=================================")

print("Trips included:", len(combined))
print("GPS observations:", total_observations)
print("Candidate rows:", total_candidate_rows)

print(
    "Average candidates per observation:",
    round(average_candidates, 2)
)

print("Fallback observations:", total_fallback)

print(
    "Fallback percentage:",
    round(fallback_percentage, 2),
    "%"
)

print(
    "Average unique route-arc overlap:",
    round(average_unique_route_coverage, 2),
    "%"
)

print("\nConsolidated report:", combined_file)
print("Handover document:", handover_file)
print("Maps directory:", MAP_DIR)

print("\nIMPORTANT:")
print("GPS-to-route label alignment is NOT yet validated.")
print("Unique route-arc overlap is NOT candidate recall.")
