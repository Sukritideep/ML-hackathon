"""Inspect split-approved PathSnap trip data without creating labels.

The script supports trip-specific ``<trip_id>.<ext>`` files and shared
``*.nodes``/``*.arcs`` files. Column positions can be overridden when the raw
dataset uses a different schema; all column options are zero-based.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


EXTENSIONS = (".track", ".nodes", ".arcs", ".route")
SHARED_STEMS = {
    ".nodes": {
        "nodes",
        "shared_nodes",
        "network_nodes",
        "map_nodes",
        "road_nodes",
        "roads_nodes",
        "network",
        "map",
        "road_network",
    },
    ".arcs": {
        "arcs",
        "shared_arcs",
        "network_arcs",
        "map_arcs",
        "road_arcs",
        "roads_arcs",
        "network",
        "map",
        "road_network",
    },
}


class InspectionError(ValueError):
    """Raised when the data layout or schema cannot be inspected safely."""


def _read_table(path: Path) -> pd.DataFrame:
    try:
        table = pd.read_csv(
            path,
            sep=r"[\s,;]+",
            engine="python",
            header=None,
            comment="#",
            dtype=str,
            skip_blank_lines=True,
        )
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise InspectionError(f"Could not read {path}: {exc}") from exc

    if table.empty:
        return table
    table = table.dropna(axis=0, how="all")
    return table.reset_index(drop=True)


def _numeric_column(table: pd.DataFrame, column: int, description: str) -> pd.Series:
    if column < 0 or column >= table.shape[1]:
        raise InspectionError(
            f"{description} column {column} is out of range for "
            f"a {table.shape[1]}-column file"
        )
    values = pd.to_numeric(table.iloc[:, column], errors="coerce")
    if values.isna().any():
        bad_rows = values.index[values.isna()].tolist()[:5]
        raise InspectionError(
            f"{description} column {column} contains non-numeric values "
            f"at rows {bad_rows}; specify the correct column"
        )
    return values


def _coordinate_columns(
    table: pd.DataFrame,
    x_column: int | None,
    y_column: int | None,
    description: str,
) -> tuple[int, int]:
    if (x_column is None) != (y_column is None):
        raise InspectionError(
            f"Specify both {description} x/lon and y/lat columns"
        )
    if x_column is not None and y_column is not None:
        _numeric_column(table, x_column, f"{description} x/lon")
        _numeric_column(table, y_column, f"{description} y/lat")
        return x_column, y_column

    geographic: list[tuple[int, int, int, float]] = []
    numeric = {
        column: pd.to_numeric(table.iloc[:, column], errors="coerce")
        for column in range(table.shape[1])
    }
    for x_idx in range(table.shape[1]):
        x_values = numeric[x_idx]
        if x_values.isna().any() or not x_values.between(-180, 180).all():
            continue
        for y_idx in range(table.shape[1]):
            if x_idx == y_idx:
                continue
            y_values = numeric[y_idx]
            if y_values.isna().any() or not y_values.between(-90, 90).all():
                continue
            # Prefer adjacent, rightmost columns: common files store optional
            # IDs/timestamps before coordinates. This is a schema heuristic;
            # bounds are printed and columns can be explicitly overridden.
            score = float(x_values.max() - x_values.min())
            geographic.append(
                (int(y_idx == x_idx + 1), x_idx, y_idx, score)
            )

    if not geographic:
        raise InspectionError(
            f"Could not identify geographic coordinate columns in {description}; "
            "supply both coordinate-column options"
        )
    _, x_idx, y_idx, _ = max(geographic, key=lambda item: item[:3])
    return x_idx, y_idx


def _resolve_file(
    data_dir: Path,
    trip_id: str,
    extension: str,
    required: bool = True,
) -> tuple[Path | None, str]:
    matches: list[Path] = []
    trip_id_lower = trip_id.lower()

    for path in data_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() != extension:
            continue
        if path.stem.lower() == trip_id_lower:
            matches.append(path)
        elif any(part.lower() == trip_id_lower for part in path.parts[:-1]):
            matches.append(path)

    matches = sorted(set(matches))
    if len(matches) > 1:
        raise InspectionError(
            f"Multiple {extension} files found for allowed trip {trip_id}: "
            + ", ".join(str(path) for path in matches)
        )
    if matches:
        return matches[0], "trip-specific"

    if extension in (".nodes", ".arcs"):
        shared = [
            path
            for path in data_dir.rglob(f"*{extension}")
            if path.is_file() and path.stem.lower() in SHARED_STEMS[extension]
        ]
        if len(shared) == 1:
            return shared[0], "shared"
        if len(shared) > 1:
            raise InspectionError(
                f"Multiple potential shared {extension} files found; "
                "select or rename the intended file: "
                + ", ".join(str(path) for path in sorted(shared))
            )

    if required:
        raise InspectionError(
            f"No {extension} file found for allowed trip {trip_id}"
        )
    return None, "missing"


def _choose_arc_columns(
    arcs: pd.DataFrame,
    arc_id_column: int | None,
    arc_u_column: int | None,
    arc_v_column: int | None,
) -> tuple[pd.Series, pd.Series, pd.Series, bool]:
    if arcs.shape[1] < 2:
        raise InspectionError(".arcs file must contain at least two endpoint columns")

    if arc_u_column is None and arc_v_column is None:
        if arcs.shape[1] >= 3:
            arc_id_column = 0 if arc_id_column is None else arc_id_column
            arc_u_column, arc_v_column = 1, 2
        else:
            arc_u_column, arc_v_column = 0, 1
    elif arc_u_column is None or arc_v_column is None:
        raise InspectionError("Specify both --arc-u-col and --arc-v-col")

    u = _numeric_column(arcs, arc_u_column, "arc start-node")
    v = _numeric_column(arcs, arc_v_column, "arc end-node")

    if arc_id_column is None:
        ids = pd.Series(np.arange(len(arcs)), index=arcs.index, dtype=np.int64)
        has_explicit_ids = False
    else:
        ids = _numeric_column(arcs, arc_id_column, "arc ID")
        has_explicit_ids = True
    return ids, u, v, has_explicit_ids


def _integer_id(value: float, description: str) -> int:
    if not math.isfinite(float(value)) or float(value) != int(value):
        raise InspectionError(f"{description} must be an integer ID, got {value!r}")
    return int(value)


def inspect_trip(
    trip_id: str,
    data_dir: Path,
    split_ids: set[str],
    options: argparse.Namespace,
) -> None:
    if trip_id not in split_ids:
        raise InspectionError(
            f"Refusing to inspect {trip_id!r}: it is not listed in train or "
            "student_test in splits.json"
        )

    files: dict[str, tuple[Path, str]] = {}
    for extension in EXTENSIONS:
        required = extension != ".nodes" and extension != ".arcs"
        path, scope = _resolve_file(data_dir, trip_id, extension, required)
        if path is not None:
            files[extension] = (path, scope)

    track = _read_table(files[".track"][0])
    route = _read_table(files[".route"][0])
    if track.empty:
        raise InspectionError(f"{files['.track'][0]} contains no GPS observations")

    gps_x_col, gps_y_col = _coordinate_columns(
        track, options.gps_lon_col, options.gps_lat_col, ".track"
    )
    lon = _numeric_column(track, gps_x_col, "GPS longitude")
    lat = _numeric_column(track, gps_y_col, "GPS latitude")
    if not lon.between(-180, 180).all() or not lat.between(-90, 90).all():
        raise InspectionError("GPS coordinate values are outside lon/lat bounds")

    route_arc_col = options.route_arc_col
    if route_arc_col is None:
        route_arc_col = 0
    route_arc_values = _numeric_column(route, route_arc_col, "route arc ID")
    route_arc_ids = [
        _integer_id(value, "Route arc ID") for value in route_arc_values.tolist()
    ]

    arcs_info = files.get(".arcs")
    nodes_info = files.get(".nodes")
    if arcs_info is None:
        raise InspectionError(
            "No .arcs file found, so route arc validity and connectivity "
            "cannot be checked"
        )
    arcs = _read_table(arcs_info[0])
    arc_ids, arc_u, arc_v, has_explicit_arc_ids = _choose_arc_columns(
        arcs, options.arc_id_col, options.arc_u_col, options.arc_v_col
    )
    arc_id_values = [
        _integer_id(value, "Arc ID") for value in arc_ids.tolist()
    ]
    if len(set(arc_id_values)) != len(arc_id_values):
        raise InspectionError(".arcs file contains duplicate arc IDs")
    arc_id_to_row = {arc_id: row for row, arc_id in enumerate(arc_id_values)}
    zero_based_row_ids = arc_id_values == list(range(len(arcs)))

    invalid_arc_ids = sorted(set(route_arc_ids) - set(arc_id_to_row))
    route_rows = [
        arc_id_to_row.get(arc_id) for arc_id in route_arc_ids
    ]
    route_pair_count = max(0, len(route_arc_ids) - 1)
    directed_links = 0
    undirected_links = 0
    for left, right in zip(route_rows, route_rows[1:]):
        if left is None or right is None:
            continue
        directed_links += int(arc_v.iloc[left] == arc_u.iloc[right])
        undirected_links += int(
            bool(
                {arc_u.iloc[left], arc_v.iloc[left]}
                & {arc_u.iloc[right], arc_v.iloc[right]}
            )
        )

    node_id_to_row: dict[int, int] = {}
    node_lon: pd.Series | None = None
    node_lat: pd.Series | None = None
    if nodes_info is not None:
        nodes = _read_table(nodes_info[0])
        if not nodes.empty:
            node_lon_col, node_lat_col = _coordinate_columns(
                nodes, options.node_lon_col, options.node_lat_col, ".nodes"
            )
            node_lon = _numeric_column(nodes, node_lon_col, "node longitude")
            node_lat = _numeric_column(nodes, node_lat_col, "node latitude")
            node_id_col = options.node_id_col
            if node_id_col is None:
                candidate_cols = [
                    column
                    for column in range(nodes.shape[1])
                    if column not in (node_lon_col, node_lat_col)
                ]
                node_id_col = candidate_cols[0] if candidate_cols else None
            if node_id_col is None:
                node_id_to_row = {row: row for row in range(len(nodes))}
            else:
                node_ids = _numeric_column(nodes, node_id_col, "node ID")
                node_id_to_row = {
                    _integer_id(value, "Node ID"): row
                    for row, value in enumerate(node_ids.tolist())
                }
            if len(node_id_to_row) != len(nodes):
                raise InspectionError(".nodes file contains duplicate node IDs")
    else:
        nodes = None

    all_lon = [lon.to_numpy()]
    all_lat = [lat.to_numpy()]
    if node_lon is not None and node_lat is not None:
        all_lon.append(node_lon.to_numpy())
        all_lat.append(node_lat.to_numpy())
    combined_lon = np.concatenate(all_lon)
    combined_lat = np.concatenate(all_lat)
    center_lon = float(np.mean(combined_lon))
    center_lat = float(np.mean(combined_lat))
    utm_zone = int((center_lon + 180.0) // 6.0) + 1
    utm_zone = min(60, max(1, utm_zone))
    if not -80.0 <= center_lat <= 84.0:
        raise InspectionError(
            f"Mean latitude {center_lat:.5f} is outside the standard UTM "
            "coverage (-80 to 84 degrees)"
        )
    epsg = (32600 if center_lat >= 0 else 32700) + utm_zone
    crs = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    gps_easting, gps_northing = transformer.transform(
        lon.to_numpy(), lat.to_numpy()
    )
    projected_eastings = [np.asarray(gps_easting)]
    projected_northings = [np.asarray(gps_northing)]
    projected_nodes: np.ndarray | None = None
    if node_lon is not None and node_lat is not None:
        node_easting, node_northing = transformer.transform(
            node_lon.to_numpy(), node_lat.to_numpy()
        )
        projected_nodes = np.column_stack((node_easting, node_northing))
        projected_eastings.append(np.asarray(node_easting))
        projected_northings.append(np.asarray(node_northing))
    projected_bounds = (
        float(np.min(np.concatenate(projected_eastings))),
        float(np.min(np.concatenate(projected_northings))),
        float(np.max(np.concatenate(projected_eastings))),
        float(np.max(np.concatenate(projected_northings))),
    )

    route_length_m: float | None = None
    route_arcs_without_geometry = 0
    if projected_nodes is not None and node_id_to_row:
        endpoint_values = [
            _integer_id(value, "Arc endpoint node ID")
            for value in pd.concat((arc_u, arc_v)).tolist()
        ]
        arc_endpoint_ids = list(zip(endpoint_values[: len(arcs)], endpoint_values[len(arcs) :]))
        route_length_m = 0.0
        for row in route_rows:
            if row is None:
                route_arcs_without_geometry += 1
                continue
            start_id, end_id = arc_endpoint_ids[row]
            if start_id not in node_id_to_row or end_id not in node_id_to_row:
                route_arcs_without_geometry += 1
                continue
            start_xy = projected_nodes[node_id_to_row[start_id]]
            end_xy = projected_nodes[node_id_to_row[end_id]]
            route_length_m += float(np.linalg.norm(end_xy - start_xy))
    else:
        route_length_m = None

    invalid_connectivity = route_pair_count - directed_links
    invalid_undirected_connectivity = route_pair_count - undirected_links

    print(f"\nTrip: {trip_id}")
    for extension in EXTENSIONS:
        if extension in files:
            path, scope = files[extension]
            print(f"  {extension}: {path} ({scope})")
    print(f"  GPS observations: {len(track)}")
    print(f"  Route length (arc references): {len(route_arc_ids)}")
    if route_length_m is None:
        print("  Projected route length: unavailable (node coordinates unavailable)")
    else:
        print(
            f"  Projected route length: {route_length_m:.2f} m "
            f"(arcs without geometry: {route_arcs_without_geometry})"
        )
    print(
        "  Geographic bounds (lon, lat): "
        f"({combined_lon.min():.7f}, {combined_lat.min():.7f}) .. "
        f"({combined_lon.max():.7f}, {combined_lat.max():.7f})"
    )
    print(f"  Selected projection: EPSG:{epsg} ({crs.name})")
    print(
        "  GPS projected bounds (easting, northing, metres): "
        f"({projected_bounds[0]:.2f}, {projected_bounds[1]:.2f}) .. "
        f"({projected_bounds[2]:.2f}, {projected_bounds[3]:.2f})"
    )
    print(f"  Arc ID column is explicit: {has_explicit_arc_ids}")
    print(f"  Arc IDs equal zero-based row indices: {zero_based_row_ids}")
    print(
        "  Nodes/arcs organization: "
        f"nodes={nodes_info[1] if nodes_info else 'missing'}, "
        f"arcs={arcs_info[1]}"
    )
    print(f"  Invalid route arc IDs ({len(invalid_arc_ids)}): {invalid_arc_ids[:20]}")
    print(
        "  Consecutive directed arc links: "
        f"{directed_links}/{route_pair_count}; broken={invalid_connectivity}"
    )
    print(
        "  Consecutive undirected endpoint links: "
        f"{undirected_links}/{route_pair_count}; "
        f"broken={invalid_undirected_connectivity}"
    )

    if nodes_info is not None:
        endpoint_ids = {
            _integer_id(value, "Arc endpoint node ID")
            for value in pd.concat((arc_u, arc_v)).tolist()
        }
        missing_node_ids = sorted(endpoint_ids - set(node_id_to_row))
        print(f"  Arc endpoints absent from nodes ({len(missing_node_ids)}): {missing_node_ids[:20]}")
    else:
        print("  Arc endpoint-to-node validation: unavailable (no .nodes file)")


def _split_ids(splits_path: Path) -> tuple[list[str], list[str]]:
    try:
        splits = json.loads(splits_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InspectionError(f"Could not read split file {splits_path}: {exc}") from exc
    if not isinstance(splits, dict):
        raise InspectionError("splits.json must contain an object")

    allowed: dict[str, list[str]] = {}
    for split_name in ("train", "student_test"):
        values = splits.get(split_name, [])
        if not isinstance(values, list) or not all(
            isinstance(value, (str, int)) for value in values
        ):
            raise InspectionError(f"splits.json {split_name!r} must be an ID list")
        allowed[split_name] = [str(value) for value in values]
    return allowed["train"], allowed["student_test"]


def _default_data_dir() -> Path:
    project_dir = Path(__file__).resolve().parents[1]
    return project_dir / "data"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect only trips listed in train/student_test splits; "
            "does not generate labels."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=_default_data_dir())
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "splits.json",
    )
    parser.add_argument(
        "--trip-id",
        help="Inspect one ID, which must be present in train or student_test",
    )
    parser.add_argument(
        "--split",
        choices=("train", "student_test"),
        default="train",
        help="Which allowed split to inspect when --trip-id is omitted",
    )
    parser.add_argument("--gps-lon-col", type=int)
    parser.add_argument("--gps-lat-col", type=int)
    parser.add_argument("--node-id-col", type=int)
    parser.add_argument("--node-lon-col", type=int)
    parser.add_argument("--node-lat-col", type=int)
    parser.add_argument("--arc-id-col", type=int)
    parser.add_argument("--arc-u-col", type=int)
    parser.add_argument("--arc-v-col", type=int)
    parser.add_argument("--route-arc-col", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    options = _parse_args(argv)
    try:
        train_ids, student_test_ids = _split_ids(options.splits)
        allowed_ids = set(train_ids) | set(student_test_ids)
        print(f"Data directory: {options.data_dir}")
        if not options.data_dir.is_dir():
            raise InspectionError(f"Data directory does not exist: {options.data_dir}")

        discovered = [
            path for path in options.data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        ]
        counts = {
            extension: sum(path.suffix.lower() == extension for path in discovered)
            for extension in EXTENSIONS
        }
        print(f"Discovered data files by extension: {counts}")
        print(
            f"Allowed trip counts: train={len(train_ids)}, "
            f"student_test={len(student_test_ids)}"
        )
        if options.trip_id is not None:
            trip_ids = [options.trip_id]
        else:
            trip_ids = train_ids if options.split == "train" else student_test_ids
        if not trip_ids:
            raise InspectionError(
                f"No IDs are listed in the requested {options.split!r} split; "
                "no trip files were read"
            )

        for trip_id in trip_ids:
            if trip_id not in allowed_ids:
                raise InspectionError(
                    f"Refusing to inspect {trip_id!r}: it is not listed in "
                    "train or student_test"
                )
            inspect_trip(trip_id, options.data_dir, allowed_ids, options)
        return 0
    except InspectionError as exc:
        print(f"Inspection error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
