"""Generate interactive GPS, reference-route, and DP-label maps."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any, Sequence

import folium
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_DATA_ROOT = Path(
    r"C:\Users\sneha sunil mithare\Downloads\Student folder_extracted\Student folder"
)
DEFAULT_TRIPS = ["00000054", "00000033", "00000002", "00000001"]


def _trip_id(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value)).zfill(8)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value)).zfill(8)
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(8)


def _read_whitespace(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing map input: {path}")
    try:
        return pd.read_csv(path, sep=r"\s+", header=None, names=columns)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def _connected_segments(
    arc_ids: Sequence[int], nodes: pd.DataFrame, arcs: pd.DataFrame
) -> list[list[list[float]]]:
    endpoints = arcs[["from_node", "to_node"]].to_numpy(dtype=np.int64)
    node_xy = nodes[["longitude", "latitude"]].to_numpy(dtype=float)
    lines: list[list[list[float]]] = []
    current: list[list[float]] = []
    previous_end: int | None = None

    for arc_id in arc_ids:
        if arc_id < 0 or arc_id >= len(endpoints):
            raise ValueError(f"Arc ID {arc_id} is outside the raw arc table")
        start_node, end_node = endpoints[arc_id]
        if (
            start_node < 0
            or end_node < 0
            or start_node >= len(node_xy)
            or end_node >= len(node_xy)
        ):
            raise ValueError(f"Arc ID {arc_id} references an invalid node")
        start_lon, start_lat = node_xy[start_node]
        end_lon, end_lat = node_xy[end_node]
        start = [float(start_lat), float(start_lon)]
        end = [float(end_lat), float(end_lon)]
        if current and previous_end == int(start_node):
            current.append(end)
        else:
            if len(current) > 1:
                lines.append(current)
            current = [start, end]
        previous_end = int(end_node)

    if len(current) > 1:
        lines.append(current)
    return lines


def _marker_indices(
    track_size: int,
    null_label_indices: Sequence[int],
    candidate_miss_indices: Sequence[int],
    max_markers: int,
) -> list[int]:
    flagged = list(dict.fromkeys([*null_label_indices, *candidate_miss_indices]))
    if len(flagged) > max_markers:
        stride = math.ceil(len(flagged) / max_markers)
        flagged = flagged[::stride]
    selected = set(flagged)
    room = max_markers - len(selected)
    if room > 0 and track_size:
        stride = max(1, math.ceil(track_size / room))
        selected.update(range(0, track_size, stride))
    return sorted(selected)


def generate_trip_map(
    trip_id: str,
    *,
    repo_root: str | Path = REPO_ROOT,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    output_dir: str | Path | None = None,
    max_markers: int = 1500,
    overwrite: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Create one split-approved map with DP labels and candidate diagnostics."""
    if isinstance(max_markers, bool) or max_markers < 1:
        raise ValueError("max_markers must be a positive integer")
    root = Path(repo_root)
    dataset_root = Path(data_root)
    trip_id = _trip_id(trip_id)
    split_file = root / "splits.json"
    splits = json.loads(split_file.read_text(encoding="utf-8"))
    partition_by_trip = {
        _trip_id(value): partition
        for partition in ("train", "student_test")
        for value in splits[partition]
    }
    if trip_id not in partition_by_trip:
        raise ValueError(f"Trip {trip_id} is not listed in root splits.json")
    partition = partition_by_trip[trip_id]

    project_results = root / "pathsnap" / "results"
    label_file = project_results / (
        "labels_train.parquet"
        if partition == "train"
        else "labels_student.parquet"
    )
    candidate_file = (
        root
        / "member1_preprocessing"
        / "outputs"
        / "candidates"
        / f"{trip_id}.csv"
    )
    raw_trip_dir = dataset_root / partition / trip_id
    track = _read_whitespace(
        raw_trip_dir / f"{trip_id}.track",
        ["longitude", "latitude", "timestamp"],
    )
    nodes = _read_whitespace(
        raw_trip_dir / f"{trip_id}.nodes", ["longitude", "latitude"]
    )
    arcs = _read_whitespace(
        raw_trip_dir / f"{trip_id}.arcs", ["from_node", "to_node"]
    )
    route = _read_whitespace(raw_trip_dir / f"{trip_id}.route", ["arc_id"])
    if track.empty:
        raise ValueError(f"Trip {trip_id} has no GPS observations")
    if not label_file.is_file():
        raise FileNotFoundError(f"Missing DP label file: {label_file}")
    if not candidate_file.is_file():
        raise FileNotFoundError(f"Missing candidate file: {candidate_file}")

    track[["longitude", "latitude"]] = track[
        ["longitude", "latitude"]
    ].apply(pd.to_numeric, errors="raise")
    nodes[["longitude", "latitude"]] = nodes[
        ["longitude", "latitude"]
    ].apply(pd.to_numeric, errors="raise")
    arcs[["from_node", "to_node"]] = arcs[
        ["from_node", "to_node"]
    ].apply(pd.to_numeric, errors="raise").astype(np.int64)
    route_ids = pd.to_numeric(route["arc_id"], errors="raise").astype(
        np.int64
    ).tolist()

    labels = pd.read_parquet(label_file)
    labels["trip_id"] = labels["trip_id"].map(_trip_id)
    trip_labels = labels[labels["trip_id"].eq(trip_id)].copy()
    if trip_labels.empty:
        raise ValueError(f"No DP labels found for trip {trip_id}")
    trip_labels = trip_labels.sort_values("obs_idx", kind="stable")
    if len(trip_labels) != len(track):
        raise ValueError(
            f"Trip {trip_id}: label rows ({len(trip_labels)}) do not match "
            f"GPS observations ({len(track)})"
        )
    if not trip_labels["obs_idx"].reset_index(drop=True).equals(
        pd.Series(np.arange(len(track)), name="obs_idx")
    ):
        raise ValueError(f"Trip {trip_id}: labels are not complete by obs_idx")

    candidates = pd.read_csv(
        candidate_file, usecols=["obs_idx", "arc_id"]
    )
    candidates["obs_idx"] = pd.to_numeric(
        candidates["obs_idx"], errors="raise"
    ).astype(np.int64)
    candidates["arc_id"] = pd.to_numeric(
        candidates["arc_id"], errors="raise"
    ).astype(np.int64)
    candidate_sets = candidates.groupby("obs_idx")["arc_id"].agg(set).to_dict()

    labeled_arc_ids = [
        int(value)
        for value in trip_labels["true_arc_id"].dropna().unique().tolist()
    ]
    reference_segments = _connected_segments(route_ids, nodes, arcs)
    dp_segments = _connected_segments(
        [
            int(value)
            for value in trip_labels["true_arc_id"].dropna().tolist()
        ],
        nodes,
        arcs,
    )
    if any(arc_id not in route_ids for arc_id in labeled_arc_ids):
        raise ValueError(
            f"Trip {trip_id}: DP labels contain arcs outside the reference route"
        )

    gps_coordinates = list(
        zip(
            track["latitude"].astype(float),
            track["longitude"].astype(float),
        )
    )
    center = [
        float(track["latitude"].median()),
        float(track["longitude"].median()),
    ]
    fmap = folium.Map(
        location=center,
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    folium.PolyLine(
        gps_coordinates,
        color="#1769aa",
        weight=3,
        opacity=0.8,
        tooltip="Observed GPS trajectory",
    ).add_to(folium.FeatureGroup(name="GPS trajectory").add_to(fmap))

    reference_group = folium.FeatureGroup(name="Reference route", show=False)
    for segment in reference_segments:
        folium.PolyLine(
            segment,
            color="#ff9800",
            weight=3,
            opacity=0.65,
            tooltip="Reference route",
        ).add_to(reference_group)
    reference_group.add_to(fmap)

    dp_group = folium.FeatureGroup(name="Monotonic-DP labeled arcs")
    for segment in dp_segments:
        folium.PolyLine(
            segment,
            color="#188038",
            weight=5,
            opacity=0.9,
            tooltip="DP-labeled road arcs",
        ).add_to(dp_group)
    dp_group.add_to(fmap)

    null_indices = trip_labels.loc[
        trip_labels["true_arc_id"].isna(), "obs_idx"
    ].astype(int).tolist()
    candidate_miss_indices = [
        int(row.obs_idx)
        for row in trip_labels.itertuples(index=False)
        if pd.notna(row.true_arc_id)
        and int(row.true_arc_id)
        not in candidate_sets.get(int(row.obs_idx), set())
    ]
    marker_group = folium.FeatureGroup(name="Null labels / candidate misses")
    for obs_idx in _marker_indices(
        len(track), null_indices, candidate_miss_indices, max_markers
    ):
        label = trip_labels.loc[trip_labels["obs_idx"].eq(obs_idx)].iloc[0][
            "true_arc_id"
        ]
        is_miss = (
            pd.notna(label)
            and int(label) not in candidate_sets.get(obs_idx, set())
        )
        status = (
            "Candidate miss"
            if is_miss
            else "DP null (>100 m)"
            if pd.isna(label)
            else "Sampled GPS observation"
        )
        latitude, longitude = gps_coordinates[obs_idx]
        color = "red" if is_miss else "orange" if pd.isna(label) else "blue"
        folium.CircleMarker(
            location=[latitude, longitude],
            radius=4 if status != "Sampled GPS observation" else 2,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(
                html.escape(
                    f"obs_idx={obs_idx}; true_arc_id={label}; status={status}"
                ),
                max_width=320,
            ),
        ).add_to(marker_group)
    marker_group.add_to(fmap)
    folium.Marker(gps_coordinates[0], tooltip="GPS start").add_to(fmap)
    folium.Marker(gps_coordinates[-1], tooltip="GPS end").add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)

    map_dir = Path(output_dir) if output_dir else project_results / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    output_path = map_dir / f"dp_trip_{trip_id}.html"
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Map already exists; pass overwrite=True to replace it: {output_path}"
        )
    fmap.save(str(output_path))

    valid_count = int(trip_labels["true_arc_id"].notna().sum())
    stats = {
        "trip_id": trip_id,
        "partition": partition,
        "gps_observations": len(track),
        "valid_dp_labels": valid_count,
        "label_coverage": valid_count / len(track),
        "candidate_misses": len(candidate_miss_indices),
        "null_labels": len(null_indices),
        "map_markers": len(
            _marker_indices(
                len(track), null_indices, candidate_miss_indices, max_markers
            )
        ),
        "output_path": str(output_path),
    }
    return output_path, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--trip-ids", nargs="+", default=DEFAULT_TRIPS)
    parser.add_argument("--max-markers", type=int, default=1500)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    results = [
        generate_trip_map(
            trip_id,
            repo_root=args.repo_root,
            data_root=args.data_root,
            output_dir=args.output_dir,
            max_markers=args.max_markers,
            overwrite=args.overwrite,
        )[1]
        for trip_id in args.trip_ids
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()