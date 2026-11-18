"""Generate global-DP GPS-to-route labels from the root split manifest.

Person 1's raw format is whitespace-delimited:
``<trip>.track`` = longitude, latitude, timestamp;
``<trip>.nodes`` = longitude, latitude;
``<trip>.arcs`` = from-node, to-node; and
``<trip>.route`` = ordered arc row IDs.

The script reproduces Person 1's median-GPS per-trip UTM selection, checks it
against the road-geometry manifest, and uses the projected endpoint arrays in
Person 1's NPZ files. Route labels come from the existing global monotonic DP,
not the older progress-window alignment. Only root ``splits.json`` train and
student_test IDs are processed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pyproj import Transformer

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.labels.assign_labels import monotonic_route_alignment


LABEL_COLUMNS = ["trip_id", "obs_idx", "true_arc_id"]
DEFAULT_DATA_ROOT = Path(
    r"C:\Users\sneha sunil mithare\Downloads\Student folder_extracted\Student folder"
)


class LabelBuildError(ValueError):
    """Raised when an input trip cannot be aligned without dropping data."""


def _projected_gps(track: pd.DataFrame) -> tuple[np.ndarray, int, Transformer]:
    if track.empty:
        raise LabelBuildError("GPS track contains no observations")
    longitude = track["longitude"].to_numpy(dtype=float)
    latitude = track["latitude"].to_numpy(dtype=float)
    if not np.isfinite(longitude).all() or not np.isfinite(latitude).all():
        raise LabelBuildError("GPS coordinates contain non-finite values")

    center_lon = float(np.median(longitude))
    center_lat = float(np.median(latitude))
    zone = int((center_lon + 180) // 6) + 1
    zone = min(max(zone, 1), 60)
    epsg = 32600 + zone if center_lat >= 0 else 32700 + zone
    transformer = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True
    )
    x, y = transformer.transform(longitude, latitude)
    gps_xy = np.column_stack((x, y)).astype(float, copy=False)
    if not np.isfinite(gps_xy).all():
        raise LabelBuildError("Projected GPS coordinates contain non-finite values")
    return gps_xy, epsg, transformer


def _read_whitespace(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=columns,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def _read_geometry(path: Path) -> dict[str, np.ndarray]:
    required = {
        "arc_id",
        "from_node",
        "to_node",
        "x1",
        "y1",
        "x2",
        "y2",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = required - set(archive.files)
            if missing:
                raise LabelBuildError(
                    f"{path} lacks required arrays: {sorted(missing)}"
                )
            arrays = {key: archive[key].copy() for key in required}
    except (OSError, ValueError) as exc:
        if isinstance(exc, LabelBuildError):
            raise
        raise LabelBuildError(f"Could not read geometry file {path}: {exc}") from exc

    lengths = {len(array) for array in arrays.values()}
    if len(lengths) != 1:
        raise LabelBuildError(f"Geometry arrays in {path} have inconsistent lengths")
    if len(np.unique(arrays["arc_id"])) != len(arrays["arc_id"]):
        raise LabelBuildError(f"Geometry file {path} has duplicate arc IDs")
    for key in ("x1", "y1", "x2", "y2"):
        if not np.isfinite(arrays[key]).all():
            raise LabelBuildError(f"Geometry array {key} in {path} is non-finite")
    return arrays


def _route_segments(
    trip_id: str,
    route_arc_ids: np.ndarray,
    raw_nodes: pd.DataFrame,
    raw_arcs: pd.DataFrame,
    geometry: dict[str, np.ndarray],
    transformer: Transformer,
) -> tuple[np.ndarray, int, list[int]]:
    arc_count = len(raw_arcs)
    invalid_route_ids = [
        int(arc_id)
        for arc_id in route_arc_ids
        if arc_id < 0 or arc_id >= arc_count
    ]
    if invalid_route_ids:
        raise LabelBuildError(
            f"Trip {trip_id}: route contains invalid arc row IDs "
            f"{invalid_route_ids[:20]}"
        )

    geometry_arc_ids = geometry["arc_id"].astype(np.int64, copy=False)
    geometry_index = {
        int(arc_id): index for index, arc_id in enumerate(geometry_arc_ids)
    }
    missing_geometry = [
        int(arc_id) for arc_id in route_arc_ids if int(arc_id) not in geometry_index
    ]
    if missing_geometry:
        raise LabelBuildError(
            f"Trip {trip_id}: route arcs lack saved geometry: "
            f"{missing_geometry[:20]}"
        )

    route_from = raw_arcs["from_node"].to_numpy(dtype=np.int64)[route_arc_ids]
    route_to = raw_arcs["to_node"].to_numpy(dtype=np.int64)[route_arc_ids]
    node_count = len(raw_nodes)
    invalid_nodes = (
        (route_from < 0)
        | (route_to < 0)
        | (route_from >= node_count)
        | (route_to >= node_count)
    )
    if invalid_nodes.any():
        bad_arc_ids = route_arc_ids[invalid_nodes].astype(int).tolist()
        raise LabelBuildError(
            f"Trip {trip_id}: route arcs reference invalid node rows "
            f"{bad_arc_ids[:20]}"
        )

    geometry_rows = np.fromiter(
        (geometry_index[int(arc_id)] for arc_id in route_arc_ids),
        dtype=np.intp,
        count=len(route_arc_ids),
    )
    if len(route_arc_ids):
        node_ids = np.unique(np.concatenate((route_from, route_to)))
        node_lon = raw_nodes["longitude"].to_numpy(dtype=float)[node_ids]
        node_lat = raw_nodes["latitude"].to_numpy(dtype=float)[node_ids]
        node_x, node_y = transformer.transform(node_lon, node_lat)
        projected_node_xy = {
            int(node_id): (float(x), float(y))
            for node_id, x, y in zip(node_ids, node_x, node_y)
        }
        expected = np.array(
            [
                [
                    *projected_node_xy[int(start_node)],
                    *projected_node_xy[int(end_node)],
                ]
                for start_node, end_node in zip(route_from, route_to)
            ],
            dtype=float,
        )
        observed = np.column_stack(
            [
                geometry[key][geometry_rows]
                for key in ("x1", "y1", "x2", "y2")
            ]
        )
        if not np.allclose(expected, observed, rtol=0.0, atol=1e-4):
            raise LabelBuildError(
                f"Trip {trip_id}: saved route geometry does not match raw nodes "
                "under Person 1's per-trip UTM projection"
            )
        route_segments = observed.reshape((-1, 2, 2))
    else:
        route_segments = np.empty((0, 2, 2), dtype=float)

    disconnected = int(np.count_nonzero(route_to[:-1] != route_from[1:]))
    return route_segments, disconnected, invalid_route_ids


def align_route_observations(
    gps_xy: np.ndarray,
    route_arc_ids: Sequence[int],
    route_segments: np.ndarray,
    max_distance: float = 100.0,
) -> tuple[list[int | None], np.ndarray, np.ndarray]:
    """Run the global DP and return original arc labels, indices, and distances."""
    route_arc_ids = list(route_arc_ids)
    route_segments = np.asarray(route_segments, dtype=float)
    if route_segments.shape != (len(route_arc_ids), 2, 2):
        raise LabelBuildError(
            "Route geometry must contain exactly one 2-endpoint segment per "
            "route arc ID"
        )
    selected, distances = monotonic_route_alignment(
        gps_xy, route_segments, max_distance=float("inf")
    )
    route_indices = np.array(
        [-1 if index is None else index for index in selected],
        dtype=np.int64,
    )
    labels = [
        int(route_arc_ids[index])
        if index is not None and distance <= max_distance
        else None
        for index, distance in zip(selected, distances)
    ]
    return labels, route_indices, distances


def _read_candidates(
    candidate_path: Path, trip_id: str, partition: str, observation_count: int
) -> dict[int, set[int]]:
    try:
        candidates = pd.read_csv(
            candidate_path,
            usecols=["partition", "trip_id", "obs_idx", "arc_id"],
            dtype={"partition": str, "trip_id": str},
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise LabelBuildError(
            f"Could not read candidates for {trip_id} from {candidate_path}: {exc}"
        ) from exc
    if not candidates["partition"].eq(partition).all():
        raise LabelBuildError(
            f"Candidate partition mismatch in {candidate_path}"
        )
    if not candidates["trip_id"].eq(trip_id).all():
        raise LabelBuildError(f"Candidate trip ID mismatch in {candidate_path}")

    candidates["obs_idx"] = pd.to_numeric(
        candidates["obs_idx"], errors="raise"
    ).astype(np.int64)
    candidates["arc_id"] = pd.to_numeric(
        candidates["arc_id"], errors="raise"
    ).astype(np.int64)
    if candidates["obs_idx"].lt(0).any() or candidates["obs_idx"].ge(
        observation_count
    ).any():
        raise LabelBuildError(
            f"Candidate observation indices are out of range for trip {trip_id}"
        )
    return {
        int(obs_idx): set(group["arc_id"].astype(int).tolist())
        for obs_idx, group in candidates.groupby("obs_idx", sort=False)
    }


def candidate_hit_miss_counts(
    true_arc_ids: Sequence[int | None],
    candidate_sets: dict[int, set[int]],
) -> tuple[int, int]:
    """Count candidate hits/misses only among valid DP labels."""
    hits = 0
    valid_count = 0
    for obs_idx, arc_id in enumerate(true_arc_ids):
        if arc_id is None:
            continue
        valid_count += 1
        if int(arc_id) in candidate_sets.get(obs_idx, set()):
            hits += 1
    return hits, valid_count - hits


def process_trip(
    trip_id: str,
    partition: str,
    data_root: Path,
    geometry_dir: Path,
    geometry_manifest: pd.DataFrame,
    candidate_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one split-approved trip, align it, and calculate diagnostics."""
    trip_dir = data_root / partition / trip_id
    track_path = trip_dir / f"{trip_id}.track"
    nodes_path = trip_dir / f"{trip_id}.nodes"
    arcs_path = trip_dir / f"{trip_id}.arcs"
    route_path = trip_dir / f"{trip_id}.route"
    required_paths = (track_path, nodes_path, arcs_path, route_path)
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise LabelBuildError(
            f"Trip {trip_id} is missing raw input files: {missing_paths}"
        )

    manifest_matches = geometry_manifest.loc[
        geometry_manifest["trip_id"].astype(str).eq(trip_id)
    ]
    if len(manifest_matches) != 1:
        raise LabelBuildError(
            f"Geometry manifest must contain exactly one entry for trip {trip_id}"
        )
    expected_epsg = int(manifest_matches.iloc[0]["epsg"])

    track = _read_whitespace(
        track_path, ["longitude", "latitude", "timestamp"]
    )
    raw_nodes = _read_whitespace(nodes_path, ["longitude", "latitude"])
    raw_arcs = _read_whitespace(arcs_path, ["from_node", "to_node"])
    route = _read_whitespace(route_path, ["arc_id"])
    if track.empty:
        raise LabelBuildError(f"Trip {trip_id} has no GPS observations")

    gps_xy, computed_epsg, transformer = _projected_gps(track)
    if computed_epsg != expected_epsg:
        raise LabelBuildError(
            f"Trip {trip_id}: computed EPSG:{computed_epsg} differs from "
            f"Person 1 manifest EPSG:{expected_epsg}"
        )
    if int(manifest_matches.iloc[0]["node_count"]) != len(raw_nodes):
        raise LabelBuildError(
            f"Trip {trip_id}: raw node count differs from geometry manifest"
        )
    if int(manifest_matches.iloc[0]["original_arc_count"]) != len(raw_arcs):
        raise LabelBuildError(
            f"Trip {trip_id}: raw arc count differs from geometry manifest"
        )

    try:
        raw_arcs["from_node"] = pd.to_numeric(
            raw_arcs["from_node"], errors="raise"
        ).astype(np.int64)
        raw_arcs["to_node"] = pd.to_numeric(
            raw_arcs["to_node"], errors="raise"
        ).astype(np.int64)
        route_arc_ids = pd.to_numeric(route["arc_id"], errors="raise").to_numpy(
            dtype=np.int64
        )
    except (ValueError, TypeError) as exc:
        raise LabelBuildError(
            f"Trip {trip_id} arc or route IDs are not valid integers"
        ) from exc

    geometry_path = geometry_dir / f"{trip_id}.npz"
    if not geometry_path.is_file():
        raise LabelBuildError(f"Missing Person 1 geometry file: {geometry_path}")
    geometry = _read_geometry(geometry_path)
    route_segments, disconnected, invalid_route_ids = _route_segments(
        trip_id,
        route_arc_ids,
        raw_nodes,
        raw_arcs,
        geometry,
        transformer,
    )

    true_arc_ids, route_index_array, distances = align_route_observations(
        gps_xy,
        route_arc_ids,
        route_segments,
    )
    if len(route_index_array) != len(track) or len(distances) != len(track):
        raise LabelBuildError(
            f"Trip {trip_id}: DP did not return one alignment per observation"
        )
    assigned_positions = route_index_array[route_index_array >= 0]
    monotonic_violations = int(
        np.count_nonzero(np.diff(assigned_positions) < 0)
    )
    if monotonic_violations:
        raise LabelBuildError(
            f"Trip {trip_id}: DP returned {monotonic_violations} decreasing "
            "route-index transitions"
        )

    labels = pd.DataFrame(
        {
            "trip_id": pd.Series([trip_id] * len(track), dtype="string"),
            "obs_idx": np.arange(len(track), dtype=np.int64),
            "true_arc_id": pd.array(true_arc_ids, dtype="Int64"),
        },
        columns=LABEL_COLUMNS,
    )
    candidate_sets = _read_candidates(
        candidate_dir / f"{trip_id}.csv",
        trip_id,
        partition,
        len(track),
    )
    candidate_hits, candidate_misses = candidate_hit_miss_counts(
        true_arc_ids, candidate_sets
    )
    valid_count = len(true_arc_ids) - sum(
        arc_id is None for arc_id in true_arc_ids
    )
    total_count = len(track)

    report = {
        "trip_id": trip_id,
        "partition": partition,
        "epsg": computed_epsg,
        "gps_observations": total_count,
        "valid_aligned_observations": valid_count,
        "label_coverage_pct": (
            100.0 * valid_count / total_count if total_count else 0.0
        ),
        "candidate_hits": candidate_hits,
        "candidate_misses": candidate_misses,
        "candidate_recall_pct": (
            100.0 * candidate_hits / valid_count if valid_count else None
        ),
        "observations_over_100m": total_count - valid_count,
        "monotonic_route_index_violations": monotonic_violations,
        "invalid_route_arc_ids": len(invalid_route_ids),
        "route_connectivity_issues": disconnected,
        "route_arc_count": len(route_arc_ids),
    }
    return labels, report


def _read_splits(path: Path) -> dict[str, list[str]]:
    try:
        splits = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabelBuildError(f"Cannot read authoritative splits {path}: {exc}") from exc
    if not isinstance(splits, dict):
        raise LabelBuildError("Root splits.json must contain a JSON object")
    result: dict[str, list[str]] = {}
    for partition in ("train", "student_test"):
        values = splits.get(partition)
        if not isinstance(values, list):
            raise LabelBuildError(f"Root splits.json needs a {partition} list")
        trip_ids = [str(value) for value in values]
        if any(len(value) != 8 or not value.isdigit() for value in trip_ids):
            raise LabelBuildError(
                f"Root splits.json contains invalid {partition} trip IDs"
            )
        if len(set(trip_ids)) != len(trip_ids):
            raise LabelBuildError(
                f"Root splits.json contains duplicate {partition} IDs"
            )
        result[partition] = trip_ids
    overlap = set(result["train"]) & set(result["student_test"])
    if overlap:
        raise LabelBuildError(
            f"Root train/student_test splits overlap: {sorted(overlap)}"
        )
    return result


def _validate_labels(
    labels: pd.DataFrame, expected_ids: Sequence[str], partition: str
) -> None:
    if list(labels.columns) != LABEL_COLUMNS:
        raise LabelBuildError(f"{partition} label output has an incorrect schema")
    if labels.duplicated(["trip_id", "obs_idx"]).any():
        raise LabelBuildError(
            f"{partition} labels contain duplicate (trip_id, obs_idx) rows"
        )
    if labels.duplicated(["trip_id", "obs_idx", "true_arc_id"]).any():
        raise LabelBuildError(
            f"{partition} labels contain duplicate triple rows"
        )
    found_ids = set(labels["trip_id"].dropna().astype(str))
    if found_ids != set(expected_ids):
        raise LabelBuildError(
            f"{partition} output IDs differ from root splits.json"
        )
    for trip_id, trip_labels in labels.groupby("trip_id", sort=False):
        observed = trip_labels["obs_idx"].to_numpy(dtype=np.int64)
        if not np.array_equal(observed, np.arange(len(observed), dtype=np.int64)):
            raise LabelBuildError(
                f"{partition}/{trip_id} does not contain every observation "
                "index exactly once"
            )


def _aggregate_report(reports: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(row["gps_observations"] for row in reports)
    valid = sum(row["valid_aligned_observations"] for row in reports)
    hits = sum(row["candidate_hits"] for row in reports)
    return {
        "trip_id": "__overall__",
        "partition": "overall",
        "epsg": None,
        "gps_observations": total,
        "valid_aligned_observations": valid,
        "label_coverage_pct": 100.0 * valid / total if total else 0.0,
        "candidate_hits": hits,
        "candidate_misses": sum(row["candidate_misses"] for row in reports),
        "candidate_recall_pct": 100.0 * hits / valid if valid else None,
        "observations_over_100m": sum(
            row["observations_over_100m"] for row in reports
        ),
        "monotonic_route_index_violations": sum(
            row["monotonic_route_index_violations"] for row in reports
        ),
        "invalid_route_arc_ids": sum(
            row["invalid_route_arc_ids"] for row in reports
        ),
        "route_connectivity_issues": sum(
            row["route_connectivity_issues"] for row in reports
        ),
        "route_arc_count": sum(row["route_arc_count"] for row in reports),
    }


def _write_outputs(
    output_dir: Path,
    train: pd.DataFrame,
    student: pd.DataFrame,
    validation: pd.DataFrame,
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    output_paths = (
        output_dir / "labels_train.parquet",
        output_dir / "labels_student.parquet",
        output_dir / "label_validation.csv",
    )
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise LabelBuildError(
            "Refusing to overwrite existing outputs without --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=".label-build-", dir=output_dir) as temp:
            temp_dir = Path(temp)
            temp_paths = [temp_dir / path.name for path in output_paths]
            train.to_parquet(temp_paths[0], index=False)
            student.to_parquet(temp_paths[1], index=False)
            validation.to_csv(temp_paths[2], index=False)
            for temp_path, output_path in zip(temp_paths, output_paths):
                temp_path.replace(output_path)
    except (ImportError, OSError, ValueError) as exc:
        raise LabelBuildError(f"Could not write label outputs: {exc}") from exc
    return output_paths


def _project_root() -> Path:
    return PROJECT_DIR.parent


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(
        description="Build split-isolated monotonic-DP route labels."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--splits", type=Path, default=root / "splits.json")
    parser.add_argument(
        "--geometry-dir",
        type=Path,
        default=root / "member1_preprocessing" / "outputs" / "road_geometry",
    )
    parser.add_argument(
        "--geometry-manifest",
        type=Path,
        default=root
        / "member1_preprocessing"
        / "outputs"
        / "reports"
        / "road_geometry_manifest.csv",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=root / "member1_preprocessing" / "outputs" / "candidates",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "pathsnap" / "results",
    )
    parser.add_argument(
        "--trip-id",
        help="Process one split-listed trip for a dry run; requires --dry-run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate and print validation without writing outputs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing only the requested label/report output files",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        splits = _read_splits(args.splits)
        all_ids = {
            trip_id: partition
            for partition in ("train", "student_test")
            for trip_id in splits[partition]
        }
        if args.trip_id:
            if args.trip_id not in all_ids:
                raise LabelBuildError(
                    f"Refusing non-split trip ID {args.trip_id!r}"
                )
            if not args.dry_run:
                raise LabelBuildError(
                    "--trip-id requires --dry-run; partial outputs are not written"
                )
            selected = [(args.trip_id, all_ids[args.trip_id])]
        else:
            selected = [
                (trip_id, partition)
                for partition in ("train", "student_test")
                for trip_id in splits[partition]
            ]

        if not args.data_root.is_dir():
            raise LabelBuildError(f"Raw data directory is missing: {args.data_root}")
        try:
            manifest = pd.read_csv(
                args.geometry_manifest, dtype={"trip_id": str}
            )
        except (OSError, pd.errors.ParserError) as exc:
            raise LabelBuildError(
                f"Cannot read geometry manifest {args.geometry_manifest}: {exc}"
            ) from exc
        required_manifest = {"trip_id", "partition", "epsg", "node_count", "original_arc_count"}
        missing_manifest_cols = required_manifest - set(manifest.columns)
        if missing_manifest_cols:
            raise LabelBuildError(
                f"Geometry manifest missing columns: {sorted(missing_manifest_cols)}"
            )

        label_frames: dict[str, list[pd.DataFrame]] = {
            "train": [],
            "student_test": [],
        }
        reports: list[dict[str, Any]] = []
        for trip_id, partition in selected:
            print(f"DP-labeling {partition}/{trip_id}")
            labels, report = process_trip(
                trip_id,
                partition,
                args.data_root,
                args.geometry_dir,
                manifest,
                args.candidate_dir,
            )
            label_frames[partition].append(labels)
            reports.append(report)

        report_frame = pd.DataFrame(reports)
        if args.trip_id:
            print(report_frame.to_string(index=False))
            return 0

        train_labels = pd.concat(label_frames["train"], ignore_index=True)
        student_labels = pd.concat(
            label_frames["student_test"], ignore_index=True
        )
        _validate_labels(train_labels, splits["train"], "train")
        _validate_labels(student_labels, splits["student_test"], "student_test")
        if set(train_labels["trip_id"]) & set(student_labels["trip_id"]):
            raise LabelBuildError("Train and student label outputs overlap")

        validation = pd.concat(
            [
                pd.DataFrame([_aggregate_report(reports)]),
                report_frame,
            ],
            ignore_index=True,
        )
        if not args.dry_run:
            output_paths = _write_outputs(
                args.output_dir,
                train_labels[LABEL_COLUMNS],
                student_labels[LABEL_COLUMNS],
                validation,
                args.overwrite,
            )
            for path in output_paths[:2]:
                labels = pd.read_parquet(path)
                expected = (
                    splits["train"]
                    if path.name == "labels_train.parquet"
                    else splits["student_test"]
                )
                _validate_labels(
                    labels[LABEL_COLUMNS], expected, path.stem.removeprefix("labels_")
                )
            print(f"Train labels: {output_paths[0]} ({len(train_labels)} rows)")
            print(
                f"Student labels: {output_paths[1]} ({len(student_labels)} rows)"
            )
            print(f"Validation report: {output_paths[2]}")
        print(validation.to_string(index=False))
        return 0
    except (LabelBuildError, OSError, ValueError, KeyError, ImportError) as exc:
        print(f"Label generation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
