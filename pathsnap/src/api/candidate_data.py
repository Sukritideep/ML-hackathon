from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer

from ..models.data import FEATURE_COLS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPROCESSING_ROOT = PROJECT_ROOT.parent / "member1_preprocessing" / "outputs"
DEFAULT_FEATURES_DIR = PREPROCESSING_ROOT / "features"
DEFAULT_ROAD_GEOMETRY_DIR = PREPROCESSING_ROOT / "road_geometry"
DEFAULT_CRS_AUDIT = PREPROCESSING_ROOT / "reports" / "per_trip_crs_audit.csv"

_SOURCE_COLUMNS = (
    "trip_id",
    "obs_idx",
    "arc_id",
    "distance_m",
    "candidate_rank",
    "projection_fraction",
    "direction_difference_deg",
    "gps_speed_mps",
    "gps_distance_m",
    "long_gap",
    "candidate_progress_m",
    "projected_x",
    "projected_y",
)


def _configured_path(variable: str, default: Path) -> Path:
    return Path(os.environ.get(variable, default)).expanduser()


def features_dir() -> Path:
    return _configured_path("PATHSNAP_FEATURES_DIR", DEFAULT_FEATURES_DIR)


def road_geometry_dir() -> Path:
    return _configured_path("PATHSNAP_ROAD_GEOMETRY_DIR", DEFAULT_ROAD_GEOMETRY_DIR)


def crs_audit_path() -> Path:
    return _configured_path("PATHSNAP_CRS_AUDIT", DEFAULT_CRS_AUDIT)


def candidate_file(trip_id: str) -> Path:
    return features_dir() / f"{trip_id}.csv"


def road_geometry_file(trip_id: str) -> Path:
    return road_geometry_dir() / f"{trip_id}.npz"


def trip_feature_data_available(trip_id: str) -> bool:
    return (
        candidate_file(trip_id).is_file()
        and road_geometry_file(trip_id).is_file()
        and crs_audit_path().is_file()
    )


def _normalize_trip_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdecimal():
        text = text[:-2]
    return text.zfill(8) if text.isdecimal() else text


def load_trip_candidates(trip_id: str) -> pd.DataFrame:
    """Load split-listed, unlabeled per-trip features for API inference."""
    path = candidate_file(trip_id)
    if not path.is_file():
        raise FileNotFoundError(f"Candidate feature file is missing: {path}")

    try:
        frame = pd.read_csv(path, usecols=list(_SOURCE_COLUMNS))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(
            f"Could not read candidate features from {path}: {exc}"
        ) from exc

    if frame.empty:
        raise ValueError(f"Candidate feature file is empty: {path}")
    if frame["trip_id"].isna().any() or {
        _normalize_trip_id(value) for value in frame["trip_id"].unique()
    } != {trip_id}:
        raise ValueError(f"Candidate feature IDs in {path} do not match {trip_id}")

    frame["trip_id"] = trip_id
    for column in ("obs_idx", "arc_id"):
        values = pd.to_numeric(frame[column], errors="raise")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(
                f"{column} contains missing or non-finite values in {path}"
            )
        if (values % 1 != 0).any():
            raise ValueError(f"{column} must contain integers in {path}")
        frame[column] = values.astype("int64")

    numeric_columns = (
        "distance_m",
        "candidate_rank",
        "projection_fraction",
        "direction_difference_deg",
        "gps_speed_mps",
        "gps_distance_m",
        "candidate_progress_m",
        "projected_x",
        "projected_y",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    if (
        frame[["distance_m", "candidate_rank", "projection_fraction"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(f"Candidate model inputs contain missing values in {path}")
    if (frame["distance_m"] < 0).any():
        raise ValueError(f"Candidate distances cannot be negative in {path}")
    if ((frame["projection_fraction"] < 0) | (frame["projection_fraction"] > 1)).any():
        raise ValueError(f"Projection fractions must be in [0, 1] in {path}")

    frame["direction_difference_deg"] = frame["direction_difference_deg"].fillna(0.0)
    frame["gps_speed_mps"] = frame["gps_speed_mps"].fillna(0.0)
    frame["f_dist"] = frame["distance_m"]
    frame["f_rank"] = frame["candidate_rank"]
    frame["f_dist_gap"] = frame["distance_m"] - frame.groupby("obs_idx")[
        "distance_m"
    ].transform("min")
    frame["f_proj"] = frame["projection_fraction"]
    frame["f_bearing_diff"] = frame["direction_difference_deg"]
    frame["f_speed"] = frame["gps_speed_mps"]
    if not np.isfinite(frame.loc[:, FEATURE_COLS].to_numpy(dtype=float)).all():
        raise ValueError(f"Candidate model features are non-finite in {path}")
    if frame.duplicated(["trip_id", "obs_idx", "arc_id"]).any():
        raise ValueError(f"Duplicate candidate keys in {path}")

    frame["long_gap"] = frame["long_gap"].fillna(False).astype(bool)
    return frame.sort_values(["obs_idx", "arc_id"], kind="mergesort").reset_index(
        drop=True
    )


def route_coordinates(
    trip_id: str,
    candidates: pd.DataFrame,
    predictions: list[tuple[int, int | None]],
) -> list[list[float]]:
    """Map each selected candidate's projected point to [lon, lat]."""
    if not predictions:
        return []

    audit_path = crs_audit_path()
    if not audit_path.is_file():
        raise FileNotFoundError(f"Per-trip CRS audit is missing: {audit_path}")
    try:
        audit = pd.read_csv(audit_path, usecols=["trip_id", "suggested_epsg"])
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(
            f"Could not read per-trip CRS audit {audit_path}: {exc}"
        ) from exc
    matches = audit[audit["trip_id"].map(_normalize_trip_id).eq(trip_id)]
    if len(matches) != 1:
        raise ValueError(f"Expected one CRS audit row for trip {trip_id}")
    epsg_values = pd.to_numeric(matches["suggested_epsg"], errors="raise")
    if epsg_values.isna().any() or (epsg_values % 1 != 0).any():
        raise ValueError(f"Invalid projected CRS for trip {trip_id}")

    try:
        transformer = Transformer.from_crs(
            f"EPSG:{int(epsg_values.iloc[0])}", "EPSG:4326", always_xy=True
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid projected CRS for trip {trip_id}") from exc

    selected = pd.DataFrame(predictions, columns=["obs_idx", "arc_id"])
    selected = selected.dropna(subset=["arc_id"]).merge(
        candidates[["obs_idx", "arc_id", "projected_x", "projected_y"]],
        on=["obs_idx", "arc_id"],
        how="left",
        validate="one_to_one",
    )
    selected = selected.dropna(subset=["projected_x", "projected_y"])
    if selected.empty:
        return []

    longitude, latitude = transformer.transform(
        selected["projected_x"].to_numpy(dtype=float),
        selected["projected_y"].to_numpy(dtype=float),
    )
    coordinates = np.column_stack((longitude, latitude))
    valid = (
        np.isfinite(coordinates).all(axis=1)
        & (coordinates[:, 0] >= -180)
        & (coordinates[:, 0] <= 180)
        & (coordinates[:, 1] >= -90)
        & (coordinates[:, 1] <= 90)
    )
    return coordinates[valid].tolist()
