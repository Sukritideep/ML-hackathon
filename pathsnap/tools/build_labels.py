"""Build true route-arc labels through an explicit projected-trip adapter.

The current core modules do not define a loader, graph, or projection API. This
script therefore does not guess those interfaces. Supply ``--adapter`` pointing
to a Python file that defines ``load_projected_trip(trip_id)`` and uses the same
projection agreed with Person A.

The adapter function must return a mapping with:

``trip_id``
    The requested trip ID.
``projection_crs``
    A pyproj-compatible CRS identifier for all projected coordinates.
``gps_xy_m``
    Finite GPS coordinates with shape (T, 2), in metres.
``node_xy_m``
    Mapping from original node IDs to projected (x, y) coordinates in metres.
``arc_endpoints``
    Mapping from each original arc ID to its directed ``(u, v)`` node IDs.
    Endpoints must follow the travel direction represented by each arc ID.
``route_arc_ids``
    Ordered original arc IDs from the reference route.

The script constructs route geometry from the endpoint node coordinates,
validates route IDs and directed connectivity, and preserves original arc IDs
in the output. The adapter is invoked only for IDs in train/student_test.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Hashable, Mapping, Sequence, TypedDict

import numpy as np
import pandas as pd

from src.labels.assign_labels import monotonic_route_alignment


class ProjectedTrip(TypedDict):
    trip_id: str
    projection_crs: str
    gps_xy_m: Any
    node_xy_m: Mapping[Hashable, Sequence[float]]
    arc_endpoints: Mapping[Hashable, Sequence[Hashable]]
    route_arc_ids: Sequence[Hashable]


class LabelBuildError(ValueError):
    """Raised when a trip cannot safely be converted to true labels."""


LABEL_COLUMNS = ["trip_id", "obs_idx", "true_arc_id"]


def _validate_crs(actual_crs: str, expected_crs: str) -> None:
    try:
        from pyproj import CRS
    except ImportError as exc:
        raise LabelBuildError(
            "pyproj is required to verify the adapter's projection; install "
            "the project requirements before building labels"
        ) from exc

    try:
        actual = CRS.from_user_input(actual_crs)
        expected = CRS.from_user_input(expected_crs)
    except Exception as exc:
        raise LabelBuildError(f"Invalid CRS declaration: {exc}") from exc

    if not expected.is_projected:
        raise LabelBuildError(
            f"Expected CRS {expected_crs!r} is not projected; use the CRS "
            "agreed with Person A"
        )
    if not actual.equals(expected):
        raise LabelBuildError(
            f"Adapter CRS {actual.to_string()!r} does not match the explicitly "
            f"selected CRS {expected.to_string()!r}"
        )
    if not expected.axis_info or any(
        not math.isclose(axis.unit_conversion_factor, 1.0, rel_tol=0.0, abs_tol=1e-12)
        for axis in expected.axis_info[:2]
    ):
        raise LabelBuildError(
            f"Expected CRS {expected.to_string()!r} does not use metres"
        )


def _finite_xy(value: Any, shape: tuple[int, ...], description: str) -> np.ndarray:
    try:
        coordinates = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise LabelBuildError(f"{description} must be numeric coordinates") from exc
    if coordinates.shape != shape:
        raise LabelBuildError(
            f"{description} must have shape {shape}, got {coordinates.shape}"
        )
    if not np.isfinite(coordinates).all():
        raise LabelBuildError(f"{description} contains non-finite coordinates")
    return coordinates


def build_trip_labels(
    trip_id: str,
    load_projected_trip: Callable[[str], Mapping[str, Any]],
    expected_crs: str,
    max_distance: float = 100.0,
) -> pd.DataFrame:
    """Validate one adapter payload and align its GPS to the ordered route."""
    try:
        trip = load_projected_trip(trip_id)
    except Exception as exc:
        raise LabelBuildError(f"Adapter failed to load trip {trip_id!r}: {exc}") from exc
    if not isinstance(trip, Mapping):
        raise LabelBuildError(
            f"Adapter result for {trip_id!r} must be a mapping"
        )

    required_fields = {
        "trip_id",
        "projection_crs",
        "gps_xy_m",
        "node_xy_m",
        "arc_endpoints",
        "route_arc_ids",
    }
    missing = sorted(required_fields - set(trip))
    if missing:
        raise LabelBuildError(
            f"Adapter result for {trip_id!r} is missing fields: {missing}"
        )
    if str(trip["trip_id"]) != trip_id:
        raise LabelBuildError(
            f"Adapter returned trip {trip['trip_id']!r} for request {trip_id!r}"
        )
    _validate_crs(str(trip["projection_crs"]), expected_crs)

    try:
        gps_xy = np.asarray(trip["gps_xy_m"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise LabelBuildError(
            f"GPS coordinates for {trip_id!r} must be numeric"
        ) from exc
    if gps_xy.ndim != 2 or gps_xy.shape[1:] != (2,):
        raise LabelBuildError(
            f"GPS coordinates for {trip_id!r} must have shape (T, 2)"
        )
    if not np.isfinite(gps_xy).all():
        raise LabelBuildError(
            f"GPS coordinates for {trip_id!r} contain non-finite values"
        )

    nodes = trip["node_xy_m"]
    arcs = trip["arc_endpoints"]
    if not isinstance(nodes, Mapping) or not isinstance(arcs, Mapping):
        raise LabelBuildError(
            f"node_xy_m and arc_endpoints for {trip_id!r} must be mappings"
        )
    projected_nodes = {
        node_id: _finite_xy(coordinate, (2,), f"Node {node_id!r}")
        for node_id, coordinate in nodes.items()
    }
    route_arc_ids = list(trip["route_arc_ids"])
    if not route_arc_ids:
        raise LabelBuildError(
            f"Trip {trip_id!r} has an empty reference route; refusing to build labels"
        )
    route_segments: list[np.ndarray] = []
    for route_position, arc_id in enumerate(route_arc_ids):
        try:
            arc_exists = arc_id in arcs
        except TypeError as exc:
            raise LabelBuildError(
                f"Route arc ID at position {route_position} is not hashable"
            ) from exc
        if not arc_exists:
            raise LabelBuildError(
                f"Trip {trip_id!r} route position {route_position} references "
                f"unknown original arc ID {arc_id!r}"
            )
        endpoints = arcs[arc_id]
        if not isinstance(endpoints, Sequence) or isinstance(endpoints, (str, bytes)) or len(endpoints) != 2:
            raise LabelBuildError(
                f"Arc {arc_id!r} in trip {trip_id!r} must have exactly two "
                "directed endpoint node IDs"
            )
        start_node, end_node = endpoints
        try:
            nodes_exist = (
                start_node in projected_nodes and end_node in projected_nodes
            )
        except TypeError as exc:
            raise LabelBuildError(
                f"Arc {arc_id!r} in trip {trip_id!r} has unhashable node IDs"
            ) from exc
        if not nodes_exist:
            raise LabelBuildError(
                f"Arc {arc_id!r} in trip {trip_id!r} references a node missing "
                "from node_xy_m"
            )
        if route_position and arcs[route_arc_ids[route_position - 1]][1] != start_node:
            previous_id = route_arc_ids[route_position - 1]
            raise LabelBuildError(
                f"Trip {trip_id!r} route is not directionally connected between "
                f"arc IDs {previous_id!r} and {arc_id!r}"
            )
        route_segments.append(
            np.stack((projected_nodes[start_node], projected_nodes[end_node]))
        )

    if route_segments:
        segment_xy = np.stack(route_segments)
    else:
        segment_xy = np.empty((0, 2, 2), dtype=float)

    assignments, _ = monotonic_route_alignment(
        gps_xy, segment_xy, max_distance=max_distance
    )
    true_arc_ids = [
        route_arc_ids[route_index] if route_index is not None else None
        for route_index in assignments
    ]
    return pd.DataFrame(
        {
            "trip_id": [trip_id] * len(assignments),
            "obs_idx": np.arange(len(assignments), dtype=np.int64),
            "true_arc_id": pd.Series(true_arc_ids, dtype=object),
        },
        columns=LABEL_COLUMNS,
    )


def _load_adapter(adapter_path: Path) -> Callable[[str], Mapping[str, Any]]:
    if not adapter_path.is_file():
        raise LabelBuildError(f"Adapter file does not exist: {adapter_path}")
    spec = importlib.util.spec_from_file_location(
        "_pathsnap_member3_label_adapter", adapter_path
    )
    if spec is None or spec.loader is None:
        raise LabelBuildError(f"Could not load adapter module: {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise LabelBuildError(f"Could not import adapter {adapter_path}: {exc}") from exc
    loader = getattr(module, "load_projected_trip", None)
    if not callable(loader):
        raise LabelBuildError(
            f"Adapter {adapter_path} must define load_projected_trip(trip_id)"
        )
    return loader


def _read_allowed_splits(path: Path) -> tuple[list[str], list[str]]:
    try:
        split_data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabelBuildError(f"Could not read splits file {path}: {exc}") from exc
    if not isinstance(split_data, dict):
        raise LabelBuildError("splits.json must contain a JSON object")

    result: list[list[str]] = []
    for split_name in ("train", "student_test"):
        ids = split_data.get(split_name, [])
        if not isinstance(ids, list) or not all(
            isinstance(trip_id, (str, int)) for trip_id in ids
        ):
            raise LabelBuildError(
                f"splits.json field {split_name!r} must be a list of IDs"
            )
        normalized = [str(trip_id) for trip_id in ids]
        if len(normalized) != len(set(normalized)):
            raise LabelBuildError(
                f"splits.json field {split_name!r} contains duplicate trip IDs"
            )
        result.append(normalized)
    if set(result[0]) & set(result[1]):
        raise LabelBuildError(
            "splits.json has trip IDs shared between train and student_test"
        )
    return result[0], result[1]


def _process_split(
    split_name: str,
    trip_ids: Sequence[str],
    loader: Callable[[str], Mapping[str, Any]],
    expected_crs: str,
    max_distance: float,
) -> pd.DataFrame:
    frames = [
        build_trip_labels(trip_id, loader, expected_crs, max_distance)
        for trip_id in trip_ids
    ]
    if not frames:
        return pd.DataFrame(columns=LABEL_COLUMNS)
    labels = pd.concat(frames, ignore_index=True)
    if labels.duplicated(["trip_id", "obs_idx"]).any():
        raise LabelBuildError(
            f"Duplicate (trip_id, obs_idx) rows found in {split_name} labels"
        )
    return labels[LABEL_COLUMNS]


def _print_summary(
    split_name: str, labels: pd.DataFrame, trip_count: int, output_path: Path
) -> None:
    total = len(labels)
    assigned = int(labels["true_arc_id"].notna().sum())
    null_count = total - assigned
    null_percentage = (100.0 * null_count / total) if total else 0.0
    print(f"\n{split_name}:")
    print(f"  Trips processed: {trip_count}")
    print(f"  Total GPS observations: {total}")
    print(f"  Successfully assigned labels: {assigned}")
    print(f"  Null labels: {null_count}")
    print(f"  Null-label percentage: {null_percentage:.2f}%")
    print(f"  Output: {output_path}")


def _write_outputs(
    results_dir: Path,
    train_labels: pd.DataFrame,
    student_labels: pd.DataFrame,
    overwrite: bool,
) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    output_paths = (
        results_dir / "labels_train.parquet",
        results_dir / "labels_student.parquet",
    )
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise LabelBuildError(
            "Refusing to overwrite existing label files without --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    with tempfile.TemporaryDirectory(prefix=".labels-", dir=results_dir) as temp_dir:
        temp_root = Path(temp_dir)
        temp_paths = (
            temp_root / output_paths[0].name,
            temp_root / output_paths[1].name,
        )
        try:
            train_labels.to_parquet(temp_paths[0], index=False)
            student_labels.to_parquet(temp_paths[1], index=False)
        except (ImportError, OSError, ValueError) as exc:
            raise LabelBuildError(
                f"Could not write parquet label files: {exc}"
            ) from exc
        for temp_path, output_path in zip(temp_paths, output_paths):
            temp_path.replace(output_path)
    return output_paths


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build labels for only train/student_test split IDs through the "
            "documented projected-trip adapter."
        )
    )
    project_dir = _project_root()
    parser.add_argument("--splits", type=Path, default=project_dir / "splits.json")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--expected-crs",
        help="Explicit projected CRS agreed with Person A, e.g. EPSG:32610",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=100.0,
        help="Maximum GPS-to-route distance in metres for a non-null label",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=project_dir / "results"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing label parquet files",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        train_ids, student_ids = _read_allowed_splits(args.splits)
        if not train_ids and not student_ids:
            raise LabelBuildError(
                f"No permitted IDs are listed in {args.splits} "
                "(train=0, student_test=0); no data was loaded and no labels "
                "were written"
            )
        if args.adapter is None:
            raise LabelBuildError(
                "No core data loader, graph representation, or agreed "
                "projection is implemented. Provide --adapter implementing "
                "load_projected_trip(trip_id) as documented in this script."
            )
        if not args.expected_crs:
            raise LabelBuildError(
                "An explicit --expected-crs agreed with Person A is required; "
                "the CRS will not be guessed"
            )
        if not math.isfinite(args.max_distance) or args.max_distance < 0:
            raise LabelBuildError("--max-distance must be finite and nonnegative")

        loader = _load_adapter(args.adapter)
        train_labels = _process_split(
            "train", train_ids, loader, args.expected_crs, args.max_distance
        )
        student_labels = _process_split(
            "student_test",
            student_ids,
            loader,
            args.expected_crs,
            args.max_distance,
        )
        output_paths = _write_outputs(
            args.results_dir, train_labels, student_labels, args.overwrite
        )
        _print_summary("train", train_labels, len(train_ids), output_paths[0])
        _print_summary(
            "student_test", student_labels, len(student_ids), output_paths[1]
        )
        return 0
    except LabelBuildError as exc:
        print(f"Label build error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
