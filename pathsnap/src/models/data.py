"""Adapters for Member 1's per-trip feature CSVs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_SOURCES = {
    "f_dist": "distance_m",
    "f_rank": "candidate_rank",
    "f_proj": "projection_fraction",
    "f_bearing_diff": "direction_difference_deg",
    "f_speed": "gps_speed_mps",
}
FEATURE_COLS = (
    "f_dist",
    "f_rank",
    "f_dist_gap",
    "f_proj",
    "f_bearing_diff",
    "f_speed",
)
CANDIDATE_COLS = ("trip_id", "obs_idx", "arc_id", *FEATURE_COLS)
ZERO_FILLED_SOURCES = ("direction_difference_deg", "gps_speed_mps")


def _read_split_ids(splits_path: str | Path) -> dict[str, list[str]]:
    try:
        splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read split file {splits_path}: {exc}") from exc

    if not isinstance(splits, dict) or set(splits) != {"train", "student_test"}:
        raise ValueError("splits.json must contain only train and student_test lists")

    result: dict[str, list[str]] = {}
    for split_name in ("train", "student_test"):
        ids = splits[split_name]
        if not isinstance(ids, list) or not all(
            isinstance(trip_id, str) and trip_id.strip() == trip_id and trip_id
            for trip_id in ids
        ):
            raise ValueError(f"splits.json {split_name} must be a list of string IDs")
        if len(ids) != len(set(ids)):
            raise ValueError(f"splits.json {split_name} contains duplicate IDs")
        result[split_name] = ids

    if set(result["train"]) & set(result["student_test"]):
        raise ValueError("Train and student_test splits must not overlap")
    return result


def _normalize_source_trip_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdecimal():
        text = text[:-2]
    return text.zfill(8) if text.isdecimal() else text


def _integer_column(values: pd.Series, name: str, trip_id: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains missing or non-finite values in {trip_id}")
    if (numeric % 1 != 0).any():
        raise ValueError(f"{name} contains non-integer values in {trip_id}")
    return numeric.astype("int64")


def _load_trip_features(
    trip_id: str,
    feature_dir: Path,
    split_name: str,
    imputed_counts: dict[str, int],
) -> pd.DataFrame:
    path = feature_dir / f"{trip_id}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing feature CSV for split trip {trip_id}: {path}")

    required = [
        "trip_id",
        "obs_idx",
        "arc_id",
        *FEATURE_SOURCES.values(),
    ]
    columns = pd.read_csv(path, nrows=0).columns
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    # Read only identifiers and approved feature inputs, never route/label metadata.
    frame = pd.read_csv(path, usecols=required)
    if frame.empty:
        raise ValueError(f"Feature CSV contains no candidate rows: {path}")
    if frame["trip_id"].isna().any() or {
        _normalize_source_trip_id(value) for value in frame["trip_id"].unique()
    } != {trip_id}:
        raise ValueError(f"Trip IDs in {path} do not match split ID {trip_id}")

    frame["obs_idx"] = _integer_column(frame["obs_idx"], "obs_idx", trip_id)
    frame["arc_id"] = _integer_column(frame["arc_id"], "arc_id", trip_id)
    for source in FEATURE_SOURCES.values():
        frame[source] = pd.to_numeric(frame[source], errors="raise")

    for source in ZERO_FILLED_SOURCES:
        count = int(frame[source].isna().sum())
        if count:
            imputed_counts[f"{split_name}.{source}"] = (
                imputed_counts.get(f"{split_name}.{source}", 0) + count
            )
            frame[source] = frame[source].fillna(0.0)

    rename = {source: feature for feature, source in FEATURE_SOURCES.items()}
    frame = frame.rename(columns=rename)
    frame["trip_id"] = trip_id
    frame["f_dist_gap"] = frame["f_dist"] - frame.groupby("obs_idx")["f_dist"].transform("min")
    frame = frame.loc[:, CANDIDATE_COLS]

    if frame.duplicated(["trip_id", "obs_idx", "arc_id"]).any():
        raise ValueError(f"Duplicate candidate keys in {path}")
    if not np.isfinite(frame.loc[:, FEATURE_COLS].to_numpy(dtype=float)).all():
        raise ValueError(f"Mapped features contain non-finite values in {path}")
    if (frame["f_dist"] < 0).any():
        raise ValueError(f"Negative candidate distances in {path}")
    if ((frame["f_proj"] < 0) | (frame["f_proj"] > 1)).any():
        raise ValueError(f"Projection fractions outside [0, 1] in {path}")

    return frame


def load_candidate_tables(
    splits_path: str | Path,
    feature_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Load split-listed A features into contract-shaped candidate tables."""
    split_ids = _read_split_ids(splits_path)
    feature_dir = Path(feature_dir)
    imputed_counts: dict[str, int] = {}
    tables = {}

    for split_name, trip_ids in split_ids.items():
        frames = [
            _load_trip_features(trip_id, feature_dir, split_name, imputed_counts)
            for trip_id in trip_ids
        ]
        if not frames:
            raise ValueError(f"splits.json {split_name} list is empty")
        table = pd.concat(frames, ignore_index=True)
        table = table.sort_values(
            ["trip_id", "obs_idx", "arc_id"], kind="mergesort"
        ).reset_index(drop=True)
        if table.duplicated(["trip_id", "obs_idx", "arc_id"]).any():
            raise ValueError(f"Duplicate candidate keys in {split_name} table")
        tables[split_name] = table.loc[:, CANDIDATE_COLS]

    return tables["train"], tables["student_test"], imputed_counts
