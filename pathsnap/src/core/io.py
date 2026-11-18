from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_FILE = PROJECT_ROOT / "splits.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class Trip:
    trip_id: str
    gps: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    route: np.ndarray | None


def _read_split_ids() -> dict[str, list[str]]:
    try:
        splits = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Unable to read trip splits from {SPLITS_FILE}: {exc}"
        ) from exc

    if not isinstance(splits, dict):
        raise ValueError("splits.json must contain a JSON object")

    result: dict[str, list[str]] = {}
    for partition in ("train", "student_test"):
        ids = splits.get(partition)
        if not isinstance(ids, list) or not all(
            isinstance(trip_id, str) and trip_id for trip_id in ids
        ):
            raise ValueError(f"splits.json must contain a list of IDs for {partition}")
        result[partition] = ids

    if set(result["train"]) & set(result["student_test"]):
        raise ValueError("Trip IDs must not overlap between train and student_test")
    return result


def data_root() -> Path:
    return Path(os.environ.get("PATHSNAP_DATA_DIR", DEFAULT_DATA_DIR)).expanduser()


def _read_numeric_table(path: Path, columns: int, dtype: type) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required trip data file is missing: {path}")
    try:
        table = np.loadtxt(path, dtype=dtype, ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not parse trip data file {path}: {exc}") from exc

    if table.ndim != 2 or table.shape[1] != columns:
        raise ValueError(
            f"{path} must have {columns} columns; found shape {table.shape}"
        )
    if np.issubdtype(table.dtype, np.number) and not np.isfinite(table).all():
        raise ValueError(f"{path} contains missing or non-finite values")
    return table


def load_trip(trip_id: str, with_route: bool = False) -> Trip:
    """Load one split-listed trip; reference routes are opt-in only."""
    splits = _read_split_ids()
    partition = next(
        (name for name, ids in splits.items() if trip_id in ids),
        None,
    )
    if partition is None:
        raise ValueError(f"Trip '{trip_id}' is not present in the split file")

    trip_dir = data_root() / partition / trip_id
    gps = _read_numeric_table(trip_dir / f"{trip_id}.track", 3, float)
    nodes = _read_numeric_table(trip_dir / f"{trip_id}.nodes", 2, float)
    arcs = _read_numeric_table(trip_dir / f"{trip_id}.arcs", 2, int)

    lon = gps[:, 0]
    lat = gps[:, 1]
    if ((lon < -180) | (lon > 180)).any() or ((lat < -90) | (lat > 90)).any():
        raise ValueError(
            f"GPS coordinates in trip {trip_id} are outside lon/lat bounds"
        )
    if (arcs < 0).any() or (arcs >= len(nodes)).any():
        raise ValueError(f"Road arcs in trip {trip_id} reference invalid node indices")

    route = None
    if with_route:
        route_values = _read_numeric_table(trip_dir / f"{trip_id}.route", 1, int)
        route = route_values[:, 0]

    return Trip(
        trip_id=trip_id,
        gps=gps,
        nodes=nodes,
        arcs=arcs,
        route=route,
    )
