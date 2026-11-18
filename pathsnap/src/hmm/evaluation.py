"""Split-isolated evaluation pipeline for HMM map-matching predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.hmm.emissions import DistanceEmissionModel
from src.hmm.metrics import DEFAULT_HOP_LIMIT, route_continuity, sequence_metrics
from src.hmm.transitions import DirectedRoadGraph
from src.hmm.viterbi import ViterbiParams, decode


def _canonical_trip_id(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        text = str(int(value))
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        text = str(int(value))
    else:
        text = str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
    return text.zfill(8)


def _obs_sort_key(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or not math.isfinite(float(numeric)):
        raise ValueError(f"Observation index is not finite and numeric: {value!r}")
    return float(numeric)


def _long_gap_for_group(group: pd.DataFrame) -> bool:
    if "long_gap" not in group.columns:
        return False
    flags = group["long_gap"]
    if flags.dtype == object:
        values = flags.astype(str).str.strip().str.lower()
        invalid = ~values.isin(["true", "false", "1", "0", ""])
        if invalid.any():
            raise ValueError(f"Invalid long_gap value: {flags[invalid].iloc[0]!r}")
        return values.isin(["true", "1"]).any()
    return bool(flags.fillna(False).astype(bool).any())


def _prediction_segments(
    features: pd.DataFrame, decoded: pd.DataFrame
) -> list[list[tuple[Any, Any]]]:
    predictions = decoded.set_index("obs_idx")["pred_arc_id"].to_dict()
    observation_groups = {
        obs_idx: group
        for obs_idx, group in features.groupby("obs_idx", sort=False, dropna=False)
    }
    ordered = sorted(observation_groups, key=_obs_sort_key)
    segments: list[list[tuple[Any, Any]]] = []
    current: list[tuple[Any, Any]] = []
    previous_numeric: float | None = None

    for obs_idx in ordered:
        group = observation_groups[obs_idx]
        numeric = _obs_sort_key(obs_idx)
        prediction = predictions.get(obs_idx)
        boundary = (
            previous_numeric is not None
            and numeric - previous_numeric != 1
        ) or _long_gap_for_group(group)
        if boundary and current:
            segments.append(current)
            current = []
        if prediction is None or pd.isna(prediction):
            if current:
                segments.append(current)
                current = []
        else:
            current.append((obs_idx, prediction))
        previous_numeric = numeric

    if current:
        segments.append(current)
    return segments


def _route_metrics(
    features: pd.DataFrame,
    decoded: pd.DataFrame,
    graph: DirectedRoadGraph,
    hop_limit: int,
) -> tuple[float, int, int]:
    segments = _prediction_segments(features, decoded)
    valid_pairs = 0
    total_pairs = 0
    unreachable_transitions = 0
    candidate_rows = {
        (row["obs_idx"], row["arc_id"]): row
        for row in features.to_dict("records")
    }

    for segment in segments:
        arcs = [arc_id for _, arc_id in segment]
        pair_count = max(0, len(arcs) - 1)
        if pair_count:
            score = route_continuity(
                arcs, graph.arc_connectivity, hop_limit=hop_limit
            )
            valid_pairs += int(round(score * pair_count))
            total_pairs += pair_count

        for (previous_obs, previous_arc), (current_obs, current_arc) in zip(
            segment, segment[1:]
        ):
            previous = candidate_rows.get((previous_obs, previous_arc))
            current = candidate_rows.get((current_obs, current_arc))
            if previous is None or current is None:
                unreachable_transitions += 1
                continue
            distance = graph.distance_between_positions(previous, current)
            if not math.isfinite(distance):
                unreachable_transitions += 1

    continuity = valid_pairs / total_pairs if total_pairs else 1.0
    return continuity, unreachable_transitions, total_pairs


def _unreachable_candidate_boundaries(
    features: pd.DataFrame,
    emission_probabilities: np.ndarray,
    graph: DirectedRoadGraph,
) -> int:
    groups = {
        obs_idx: np.asarray(indices, dtype=np.intp)
        for obs_idx, indices in features.groupby(
            "obs_idx", sort=False, dropna=False
        ).indices.items()
    }
    ordered = sorted(groups, key=_obs_sort_key)
    unreachable = 0
    previous_obs: Any | None = None
    previous_positions: list[int] = []

    for obs_idx in ordered:
        positions = [
            int(position)
            for position in groups[obs_idx]
            if emission_probabilities[position] > 0
            and features.iloc[int(position)]["arc_id"] in graph.arc_to_endpoints
        ]
        current_group = features.iloc[groups[obs_idx]]
        consecutive = (
            previous_obs is not None
            and _obs_sort_key(obs_idx) - _obs_sort_key(previous_obs) == 1
        )
        if (
            positions
            and previous_positions
            and consecutive
            and not _long_gap_for_group(current_group)
        ):
            reachable = any(
                math.isfinite(
                    graph.distance_between_positions(
                        features.iloc[previous_position].to_dict(),
                        features.iloc[current_position].to_dict(),
                    )
                )
                for previous_position in previous_positions
                for current_position in positions
            )
            if not reachable:
                unreachable += 1
        previous_obs = obs_idx
        previous_positions = positions
    return unreachable


def _decoder_segment_count(
    features: pd.DataFrame,
    emission_probabilities: np.ndarray,
    graph: DirectedRoadGraph,
) -> int:
    groups = {
        obs_idx: np.asarray(indices, dtype=np.intp)
        for obs_idx, indices in features.groupby(
            "obs_idx", sort=False, dropna=False
        ).indices.items()
    }
    ordered = sorted(groups, key=_obs_sort_key)
    segments = 0
    active = False
    previous_numeric: float | None = None

    for obs_idx in ordered:
        positions = groups[obs_idx]
        valid = any(
            emission_probabilities[int(position)] > 0
            and features.iloc[int(position)]["arc_id"] in graph.arc_to_endpoints
            for position in positions
        )
        numeric = _obs_sort_key(obs_idx)
        boundary = (
            previous_numeric is not None and numeric - previous_numeric != 1
        ) or _long_gap_for_group(features.iloc[positions])
        if not valid:
            active = False
        elif not active or boundary:
            segments += 1
            active = True
        previous_numeric = numeric
    return segments


def evaluate_trip_frames(
    trip_id: str,
    candidates: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    graph: DirectedRoadGraph,
    *,
    emission_scale_m: float = 10.0,
    transition_scale_m: float = 20.0,
    hop_limit: int = DEFAULT_HOP_LIMIT,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate one trip and return its metrics and label-aligned predictions."""
    required_candidates = {"trip_id", "obs_idx", "arc_id"}
    required_features = {
        "trip_id",
        "obs_idx",
        "arc_id",
        "distance_m",
        "gps_distance_m",
    }
    required_labels = {"trip_id", "obs_idx", "true_arc_id"}
    if missing := required_candidates - set(candidates.columns):
        raise ValueError(f"Candidate table is missing columns: {sorted(missing)}")
    if missing := required_features - set(features.columns):
        raise ValueError(f"Feature table is missing columns: {sorted(missing)}")
    if missing := required_labels - set(labels.columns):
        raise ValueError(f"Label table is missing columns: {sorted(missing)}")

    canonical_id = _canonical_trip_id(trip_id)
    candidates = candidates.copy()
    features = features.copy()
    labels = labels.copy()
    candidates["trip_id"] = canonical_id
    features["trip_id"] = canonical_id
    labels["trip_id"] = labels["trip_id"].map(_canonical_trip_id)

    candidate_keys = set(zip(candidates["obs_idx"], candidates["arc_id"]))
    feature_keys = set(zip(features["obs_idx"], features["arc_id"]))
    if len(candidates) != len(features) or candidate_keys != feature_keys:
        raise ValueError(
            f"Trip {canonical_id} candidate and feature rows do not align"
        )
    if candidates.duplicated(["trip_id", "obs_idx", "arc_id"]).any():
        raise ValueError(f"Trip {canonical_id} has duplicate candidate rows")
    trip_labels = labels[labels["trip_id"].eq(canonical_id)].copy()
    if trip_labels.empty:
        raise ValueError(f"Trip {canonical_id} has no ground-truth label rows")
    if trip_labels.duplicated(["trip_id", "obs_idx"]).any():
        raise ValueError(f"Trip {canonical_id} has duplicate label observations")

    emission_model = DistanceEmissionModel(scale_m=emission_scale_m)
    emission_probabilities = emission_model.predict_proba(features)
    decoded = decode(
        features,
        emission_probabilities,
        graph,
        ViterbiParams(transition_scale_m=transition_scale_m),
    )
    if "trip_id" not in decoded.columns:
        decoded.insert(0, "trip_id", canonical_id)
    decoded["trip_id"] = canonical_id

    trip_labels = trip_labels.sort_values("obs_idx", kind="stable")
    decoded_keys = set(decoded["obs_idx"])
    aligned = trip_labels.merge(
        decoded[["trip_id", "obs_idx", "pred_arc_id"]],
        on=["trip_id", "obs_idx"],
        how="left",
        validate="one_to_one",
    )
    aligned = aligned.sort_values("obs_idx", kind="stable").reset_index(drop=True)
    candidate_arcs = candidates.groupby("obs_idx")["arc_id"].agg(set).to_dict()
    valid_truth = aligned["true_arc_id"].notna()
    true_arcs = aligned.loc[valid_truth, "true_arc_id"].tolist()
    predicted_aligned = aligned.loc[valid_truth, "pred_arc_id"].tolist()
    non_null_predictions = [
        arc_id for arc_id in predicted_aligned if not pd.isna(arc_id)
    ]
    set_metrics = sequence_metrics(true_arcs, non_null_predictions)
    aligned_metrics = sequence_metrics(
        true_arcs,
        [
            None if pd.isna(arc_id) else arc_id
            for arc_id in predicted_aligned
        ],
    )
    correct = sum(
        not pd.isna(prediction) and prediction == truth
        for truth, prediction in zip(true_arcs, predicted_aligned)
    )
    valid_label_count = len(true_arcs)
    candidate_hits = sum(
        truth in candidate_arcs.get(obs_idx, set())
        for obs_idx, truth in zip(
            aligned.loc[valid_truth, "obs_idx"], true_arcs
        )
    )
    missing_candidates = sum(
        obs_idx not in candidate_arcs
        for obs_idx in aligned["obs_idx"]
    )
    candidate_misses = valid_label_count - candidate_hits

    continuity, unreachable_transitions, continuity_pairs = _route_metrics(
        features, decoded, graph, hop_limit
    )
    valid_candidate_mask = features["arc_id"].map(
        lambda arc_id: arc_id in graph.arc_to_endpoints
    )
    segments = _prediction_segments(features, decoded)
    metrics: dict[str, Any] = {
        "trip_id": canonical_id,
        "total_observations": int(len(trip_labels)),
        "valid_labels": int(valid_label_count),
        "predicted_observations": int(aligned["pred_arc_id"].notna().sum()),
        "null_predictions": int(aligned["pred_arc_id"].isna().sum()),
        "missing_predictions": int(
            len(set(trip_labels["obs_idx"]) - decoded_keys)
        ),
        "missing_candidate_observations": int(missing_candidates),
        "candidate_hits": int(candidate_hits),
        "candidate_misses": int(candidate_misses),
        "candidate_recall": (
            float(candidate_hits / valid_label_count)
            if valid_label_count
            else 0.0
        ),
        "precision": float(set_metrics["arc_set_precision"]),
        "recall": float(set_metrics["arc_set_recall"]),
        "sequence_accuracy": (
            float(correct / valid_label_count) if valid_label_count else 0.0
        ),
        "edit_distance": int(aligned_metrics["edit_distance"]),
        "longest_correct_run": int(aligned_metrics["longest_correct_run"]),
        "route_continuity": float(continuity),
        "unreachable_predicted_transitions": int(unreachable_transitions),
        "unreachable_candidate_boundaries": int(
            _unreachable_candidate_boundaries(
                features, emission_probabilities, graph
            )
        ),
        "decoding_segments": int(
            _decoder_segment_count(features, emission_probabilities, graph)
        ),
        "predicted_segments": int(len(segments)),
        "invalid_candidate_arc_rows": int((~valid_candidate_mask).sum()),
        "_continuity_pairs": int(continuity_pairs),
        "_correct_predictions": int(correct),
        "_non_null_predictions_on_valid_labels": int(
            sum(not pd.isna(prediction) for prediction in predicted_aligned)
        ),
    }
    return metrics, aligned


def _overall_metrics(
    per_trip_metrics: dict[str, dict[str, Any]],
    aligned_frames: list[pd.DataFrame],
) -> dict[str, Any]:
    if not per_trip_metrics:
        return {
            "trip_count": 0,
            "total_observations": 0,
            "valid_labels": 0,
            "predicted_observations": 0,
            "null_predictions": 0,
            "missing_predictions": 0,
            "missing_candidate_observations": 0,
            "candidate_hits": 0,
            "candidate_misses": 0,
            "candidate_recall": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "sequence_accuracy": 0.0,
            "edit_distance": 0,
            "longest_correct_run": 0,
            "route_continuity": 1.0,
            "unreachable_predicted_transitions": 0,
            "unreachable_candidate_boundaries": 0,
            "decoding_segments": 0,
            "predicted_segments": 0,
            "invalid_candidate_arc_rows": 0,
        }

    all_aligned = pd.concat(aligned_frames, ignore_index=True)
    valid = all_aligned["true_arc_id"].notna()
    true_keys = [
        (trip_id, arc_id)
        for trip_id, arc_id in zip(
            all_aligned.loc[valid, "trip_id"],
            all_aligned.loc[valid, "true_arc_id"],
        )
    ]
    predicted_keys = [
        (trip_id, arc_id)
        for trip_id, arc_id, prediction in zip(
            all_aligned.loc[valid, "trip_id"],
            all_aligned.loc[valid, "true_arc_id"],
            all_aligned.loc[valid, "pred_arc_id"],
        )
        if not pd.isna(prediction)
    ]
    set_metrics = sequence_metrics(true_keys, predicted_keys)
    continuity_pairs = sum(
        metrics["_continuity_pairs"] for metrics in per_trip_metrics.values()
    )
    continuity_total = sum(
        metrics["route_continuity"] * metrics["_continuity_pairs"]
        for metrics in per_trip_metrics.values()
    )
    valid_labels = int(valid.sum())
    correct = sum(
        int(metrics["_correct_predictions"])
        for metrics in per_trip_metrics.values()
    )
    candidate_hits = sum(
        int(metrics["candidate_hits"])
        for metrics in per_trip_metrics.values()
    )
    aggregate: dict[str, Any] = {
        "trip_count": len(per_trip_metrics),
        "total_observations": sum(
            int(metrics["total_observations"])
            for metrics in per_trip_metrics.values()
        ),
        "valid_labels": valid_labels,
        "predicted_observations": sum(
            int(metrics["predicted_observations"])
            for metrics in per_trip_metrics.values()
        ),
        "null_predictions": sum(
            int(metrics["null_predictions"])
            for metrics in per_trip_metrics.values()
        ),
        "missing_predictions": sum(
            int(metrics["missing_predictions"])
            for metrics in per_trip_metrics.values()
        ),
        "missing_candidate_observations": sum(
            int(metrics["missing_candidate_observations"])
            for metrics in per_trip_metrics.values()
        ),
        "candidate_hits": candidate_hits,
        "candidate_misses": sum(
            int(metrics["candidate_misses"])
            for metrics in per_trip_metrics.values()
        ),
        "candidate_recall": candidate_hits / valid_labels if valid_labels else 0.0,
        "precision": float(set_metrics["arc_set_precision"]),
        "recall": float(set_metrics["arc_set_recall"]),
        "sequence_accuracy": correct / valid_labels if valid_labels else 0.0,
        "edit_distance": sum(
            int(metrics["edit_distance"])
            for metrics in per_trip_metrics.values()
        ),
        "longest_correct_run": max(
            int(metrics["longest_correct_run"])
            for metrics in per_trip_metrics.values()
        ),
        "route_continuity": (
            continuity_total / continuity_pairs if continuity_pairs else 1.0
        ),
        "unreachable_predicted_transitions": sum(
            int(metrics["unreachable_predicted_transitions"])
            for metrics in per_trip_metrics.values()
        ),
        "unreachable_candidate_boundaries": sum(
            int(metrics["unreachable_candidate_boundaries"])
            for metrics in per_trip_metrics.values()
        ),
        "decoding_segments": sum(
            int(metrics["decoding_segments"])
            for metrics in per_trip_metrics.values()
        ),
        "predicted_segments": sum(
            int(metrics["predicted_segments"])
            for metrics in per_trip_metrics.values()
        ),
        "invalid_candidate_arc_rows": sum(
            int(metrics["invalid_candidate_arc_rows"])
            for metrics in per_trip_metrics.values()
        ),
    }
    return aggregate


def build_evaluation_report(
    repo_root: str | Path,
    *,
    emission_scale_m: float = 10.0,
    transition_scale_m: float = 20.0,
    hop_limit: int = DEFAULT_HOP_LIMIT,
    max_trips_per_split: int | None = None,
) -> dict[str, Any]:
    """Load each split independently and build overall/per-trip HMM metrics."""
    root = Path(repo_root)
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    train_ids = [_canonical_trip_id(value) for value in splits["train"]]
    student_ids = [
        _canonical_trip_id(value) for value in splits["student_test"]
    ]
    if set(train_ids) & set(student_ids):
        raise ValueError("train and student_test splits must be disjoint")
    if max_trips_per_split is not None:
        if (
            isinstance(max_trips_per_split, bool)
            or not isinstance(max_trips_per_split, int)
            or max_trips_per_split < 1
        ):
            raise ValueError("max_trips_per_split must be a positive integer")
        train_ids = train_ids[:max_trips_per_split]
        student_ids = student_ids[:max_trips_per_split]

    project_results = root / "pathsnap" / "results"
    split_specs = {
        "train": (
            train_ids,
            project_results / "labels_train.parquet",
        ),
        "student_test": (
            student_ids,
            project_results / "labels_student.parquet",
        ),
    }
    output: dict[str, Any] = {}
    for split_name, (trip_ids, labels_path) in split_specs.items():
        labels = pd.read_parquet(labels_path)
        per_trip: dict[str, dict[str, Any]] = {}
        aligned_frames: list[pd.DataFrame] = []
        for trip_id in trip_ids:
            candidate_path = (
                root
                / "member1_preprocessing"
                / "outputs"
                / "candidates"
                / f"{trip_id}.csv"
            )
            feature_path = (
                root
                / "member1_preprocessing"
                / "outputs"
                / "features"
                / f"{trip_id}.csv"
            )
            geometry_path = (
                root
                / "member1_preprocessing"
                / "outputs"
                / "road_geometry"
                / f"{trip_id}.npz"
            )
            for path in (candidate_path, feature_path, geometry_path):
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing {split_name} evaluation input: {path}"
                    )
            trip_labels = labels[
                labels["trip_id"].map(_canonical_trip_id).eq(trip_id)
            ]
            metrics, aligned = evaluate_trip_frames(
                trip_id,
                pd.read_csv(candidate_path),
                pd.read_csv(feature_path),
                trip_labels,
                DirectedRoadGraph.from_npz(geometry_path),
                emission_scale_m=emission_scale_m,
                transition_scale_m=transition_scale_m,
                hop_limit=hop_limit,
            )
            per_trip[trip_id] = metrics
            aligned_frames.append(aligned)
        output[split_name] = {
            "complete": len(per_trip) == len(splits[split_name]),
            "requested_trip_count": len(splits[split_name]),
            "processed_trip_count": len(per_trip),
            "overall": _overall_metrics(per_trip, aligned_frames),
            "per_trip": per_trip,
        }

    return {
        "schema_version": 1,
        "configuration": {
            "emission_scale_m": float(emission_scale_m),
            "transition_scale_m": float(transition_scale_m),
            "route_continuity_hop_limit": hop_limit,
            "break_on_long_gap": True,
            "max_trips_per_split": max_trips_per_split,
            "student_test_labels_used_for_training": False,
        },
        "metric_definitions": {
            "precision_recall": "Arc-set precision/recall from sequence_metrics, with arc IDs scoped by trip in overall metrics.",
            "sequence_accuracy": "Correct non-null arc predictions divided by observations with non-null DP labels; null predictions count as incorrect.",
            "edit_distance": "Per-trip Levenshtein distance over valid-label observations; null predictions count as a distinct missing state.",
            "longest_correct_run": "Longest chronologically aligned run of correct predictions among valid-label observations.",
            "missing_candidate_observations": "Labeled observations with no candidate rows.",
            "candidate_misses": "Valid DP labels whose arc ID is absent from that observation's candidate set.",
            "unreachable_predicted_transitions": "Predicted consecutive transitions within decoding segments with infinite directed road distance.",
            "unreachable_candidate_boundaries": "Consecutive candidate observations (excluding long-gap/index boundaries) with no reachable candidate-arc pair.",
        },
        "splits": output,
    }


def save_evaluation_report(
    report: dict[str, Any], report_path: str | Path, *, overwrite: bool = False
) -> Path:
    """Write a JSON report, refusing to replace an existing file by default."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Report already exists; pass overwrite=True to replace it: {path}"
        )
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def main() -> None:
    default_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--emission-scale-m", type=float, default=10.0)
    parser.add_argument("--transition-scale-m", type=float, default=20.0)
    parser.add_argument("--hop-limit", type=int, default=DEFAULT_HOP_LIMIT)
    parser.add_argument("--max-trips-per-split", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = build_evaluation_report(
        args.repo_root,
        emission_scale_m=args.emission_scale_m,
        transition_scale_m=args.transition_scale_m,
        hop_limit=args.hop_limit,
        max_trips_per_split=args.max_trips_per_split,
    )
    report_path = args.report or (
        args.repo_root / "pathsnap" / "results" / "hmm_evaluation_report.json"
    )
    saved = save_evaluation_report(
        report, report_path, overwrite=args.overwrite
    )
    print(f"Saved HMM evaluation report: {saved}")


if __name__ == "__main__":
    main()
