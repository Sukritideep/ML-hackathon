"""Student-test evaluation of the trained candidate model and HMM decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.hmm.evaluation import _route_metrics
from src.hmm.metrics import DEFAULT_HOP_LIMIT, sequence_metrics
from src.hmm.transitions import DirectedRoadGraph
from src.hmm.viterbi import ViterbiParams, decode
from src.models.data import FEATURE_COLS, _load_trip_features


_KEY_COLUMNS = ["trip_id", "obs_idx", "arc_id"]
_TRANSITION_COLUMNS = [
    "candidate_progress_m",
    "gps_distance_m",
    "long_gap",
]


def _canonical_trip_id(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        text = str(int(value))
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        text = str(int(value))
    else:
        text = str(value).strip()
        if text.endswith(".0") and text[:-2].isdecimal():
            text = text[:-2]
    return text.zfill(8)


def _integer_keys(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result = frame.copy()
    for column in ("obs_idx", "arc_id"):
        numeric = pd.to_numeric(result[column], errors="raise")
        if numeric.isna().any() or (numeric % 1 != 0).any():
            raise ValueError(f"{name}.{column} must contain finite integers")
        result[column] = numeric.astype(np.int64)
    if result["trip_id"].isna().any():
        raise ValueError(f"{name}.trip_id cannot contain null values")
    result["trip_id"] = result["trip_id"].map(_canonical_trip_id)
    return result


def _classifier_top1(
    candidates: pd.DataFrame, probabilities: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for obs_idx, positions in candidates.groupby(
        "obs_idx", sort=False, dropna=False
    ).indices.items():
        row_position = int(positions[int(np.argmax(probabilities[positions]))])
        rows.append(
            {
                "trip_id": candidates.iloc[row_position]["trip_id"],
                "obs_idx": obs_idx,
                "pred_arc_id": candidates.iloc[row_position]["arc_id"],
            }
        )
    return pd.DataFrame(rows, columns=["trip_id", "obs_idx", "pred_arc_id"])


def _prediction_metrics(
    labels: pd.DataFrame,
    decoded: pd.DataFrame,
    candidates: pd.DataFrame,
    graph: DirectedRoadGraph,
    hop_limit: int,
) -> dict[str, Any]:
    aligned = labels.merge(
        decoded[["trip_id", "obs_idx", "pred_arc_id"]],
        on=["trip_id", "obs_idx"],
        how="left",
        validate="one_to_one",
    ).sort_values("obs_idx", kind="stable")
    valid_truth = aligned["true_arc_id"].notna()
    true_arcs = aligned.loc[valid_truth, "true_arc_id"].tolist()
    aligned_predictions = aligned.loc[valid_truth, "pred_arc_id"].tolist()
    correct = sum(
        pd.notna(prediction) and prediction == truth
        for truth, prediction in zip(true_arcs, aligned_predictions)
    )
    predicted_on_valid = sum(pd.notna(value) for value in aligned_predictions)
    predicted_total = int(aligned["pred_arc_id"].notna().sum())
    valid_count = len(true_arcs)
    set_metrics = sequence_metrics(
        true_arcs,
        [value for value in aligned_predictions if pd.notna(value)],
    )
    aligned_metrics = sequence_metrics(
        true_arcs,
        [None if pd.isna(value) else value for value in aligned_predictions],
    )

    candidate_sets = (
        candidates.groupby("obs_idx")["arc_id"].agg(set).to_dict()
    )
    candidate_hits = sum(
        truth in candidate_sets.get(obs_idx, set())
        for obs_idx, truth in zip(
            aligned.loc[valid_truth, "obs_idx"], true_arcs
        )
    )
    candidate_observations = set(candidates["obs_idx"])
    decoded_keys = set(decoded["obs_idx"])
    continuity, unreachable, continuity_pairs = _route_metrics(
        candidates, decoded, graph, hop_limit
    )

    return {
        "total_observations": int(len(labels)),
        "valid_labels": int(valid_count),
        "valid_label_coverage": (
            float(valid_count / len(labels)) if len(labels) else 0.0
        ),
        "predicted_observations": predicted_total,
        "prediction_coverage": (
            float(predicted_total / len(labels)) if len(labels) else 0.0
        ),
        "null_predictions": int(len(labels) - predicted_total),
        "missing_predictions": int(len(set(labels["obs_idx"]) - decoded_keys)),
        "missing_candidate_observations": int(
            len(set(labels["obs_idx"]) - candidate_observations)
        ),
        "candidate_hits": int(candidate_hits),
        "candidate_misses": int(valid_count - candidate_hits),
        "candidate_recall": (
            float(candidate_hits / valid_count) if valid_count else 0.0
        ),
        "correct_predictions": int(correct),
        "pointwise_accuracy": float(correct / valid_count) if valid_count else 0.0,
        "pointwise_precision": (
            float(correct / predicted_on_valid) if predicted_on_valid else 0.0
        ),
        "pointwise_recall": float(correct / valid_count) if valid_count else 0.0,
        "arc_set_precision": float(set_metrics["arc_set_precision"]),
        "arc_set_recall": float(set_metrics["arc_set_recall"]),
        "edit_distance": int(aligned_metrics["edit_distance"]),
        "longest_correct_run": int(aligned_metrics["longest_correct_run"]),
        "route_continuity": float(continuity),
        "unreachable_predicted_transitions": int(unreachable),
        "_continuity_pairs": int(continuity_pairs),
    }


def evaluate_ml_trip_frames(
    trip_id: str,
    model_candidates: pd.DataFrame,
    transition_features: pd.DataFrame,
    labels: pd.DataFrame,
    graph: DirectedRoadGraph,
    model_bundle: Any,
    *,
    transition_scale_m: float = 20.0,
    hop_limit: int = DEFAULT_HOP_LIMIT,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compare model top-1 with Viterbi using one trip's original candidate order."""
    trip_id = _canonical_trip_id(trip_id)
    required_model = {"trip_id", "obs_idx", "arc_id", *FEATURE_COLS}
    required_transition = {
        "trip_id",
        "obs_idx",
        "arc_id",
        *_TRANSITION_COLUMNS,
    }
    required_labels = {"trip_id", "obs_idx", "true_arc_id"}
    for frame_name, frame, required in (
        ("model_candidates", model_candidates, required_model),
        ("transition_features", transition_features, required_transition),
        ("labels", labels, required_labels),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{frame_name} is missing columns: {missing}")

    candidates = _integer_keys(model_candidates, "model_candidates")
    features = _integer_keys(transition_features, "transition_features")
    trip_labels = labels.copy()
    trip_labels["trip_id"] = trip_labels["trip_id"].map(_canonical_trip_id)
    trip_labels = trip_labels[trip_labels["trip_id"].eq(trip_id)].copy()
    trip_labels["obs_idx"] = pd.to_numeric(
        trip_labels["obs_idx"], errors="raise"
    ).astype(np.int64)
    if candidates["trip_id"].nunique() != 1 or not candidates["trip_id"].eq(
        trip_id
    ).all():
        raise ValueError(f"Model candidate rows do not belong to trip {trip_id}")
    if features["trip_id"].nunique() != 1 or not features["trip_id"].eq(
        trip_id
    ).all():
        raise ValueError(f"Transition features do not belong to trip {trip_id}")
    if candidates.duplicated(_KEY_COLUMNS).any():
        raise ValueError(f"Trip {trip_id} has duplicate candidate rows")
    if features.duplicated(_KEY_COLUMNS).any():
        raise ValueError(f"Trip {trip_id} has duplicate transition-feature rows")
    if trip_labels.empty or trip_labels.duplicated(
        ["trip_id", "obs_idx"]
    ).any():
        raise ValueError(f"Trip {trip_id} has missing or duplicate label rows")
    candidate_keys = set(map(tuple, candidates[_KEY_COLUMNS].to_numpy()))
    feature_keys = set(map(tuple, features[_KEY_COLUMNS].to_numpy()))
    if candidate_keys != feature_keys or len(candidates) != len(features):
        raise ValueError(
            f"Trip {trip_id} model candidates and transition features do not align"
        )

    transition_columns = [
        *_KEY_COLUMNS,
        *_TRANSITION_COLUMNS,
    ]
    decoder_candidates = candidates.merge(
        features[transition_columns],
        on=_KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        sort=False,
    ).reset_index(drop=True)
    if not decoder_candidates[_KEY_COLUMNS].equals(
        candidates[_KEY_COLUMNS].reset_index(drop=True)
    ):
        raise ValueError("Joining transition features changed candidate row order")

    probabilities = np.asarray(model_bundle.predict_proba(candidates), dtype=float)
    if (
        probabilities.ndim != 1
        or probabilities.size != len(candidates)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("Model bundle returned invalid row-aligned probabilities")

    classifier_predictions = _classifier_top1(candidates, probabilities)
    hmm_predictions = decode(
        decoder_candidates,
        probabilities,
        graph,
        ViterbiParams(transition_scale_m=transition_scale_m),
    )
    if "trip_id" not in hmm_predictions.columns:
        hmm_predictions.insert(0, "trip_id", trip_id)
    hmm_predictions["trip_id"] = trip_id

    trip_labels["trip_id"] = trip_id
    if trip_labels["true_arc_id"].notna().any():
        numeric_truth = pd.to_numeric(
            trip_labels["true_arc_id"], errors="raise"
        ).dropna()
        if (numeric_truth % 1 != 0).any():
            raise ValueError(f"Trip {trip_id} labels contain non-integer arc IDs")
    trip_labels = trip_labels.sort_values("obs_idx", kind="stable")
    aligned = trip_labels.merge(
        classifier_predictions.rename(
            columns={"pred_arc_id": "classifier_pred_arc_id"}
        ),
        on=["trip_id", "obs_idx"],
        how="left",
        validate="one_to_one",
    ).merge(
        hmm_predictions.rename(columns={"pred_arc_id": "hmm_pred_arc_id"}),
        on=["trip_id", "obs_idx"],
        how="left",
        validate="one_to_one",
    )
    gap_by_observation = (
        decoder_candidates.groupby("obs_idx", sort=False)["long_gap"]
        .first()
        .rename("long_gap")
    )
    aligned = aligned.join(gap_by_observation, on="obs_idx")
    aligned = aligned.sort_values("obs_idx", kind="stable").reset_index(drop=True)

    classifier_metrics = _prediction_metrics(
        trip_labels,
        classifier_predictions,
        decoder_candidates,
        graph,
        hop_limit,
    )
    hmm_metrics = _prediction_metrics(
        trip_labels,
        hmm_predictions,
        decoder_candidates,
        graph,
        hop_limit,
    )
    metrics = {
        "trip_id": trip_id,
        "candidate_rows": int(len(candidates)),
        "invalid_candidate_arc_rows": int(
            sum(arc_id not in graph.arc_to_endpoints for arc_id in candidates["arc_id"])
        ),
        "classifier_only": classifier_metrics,
        "hmm_viterbi": hmm_metrics,
    }
    return metrics, aligned


def _overall_metrics(
    trip_metrics: dict[str, dict[str, Any]],
    prediction_frames: list[pd.DataFrame],
    method: str,
) -> dict[str, Any]:
    metric_rows = [values[method] for values in trip_metrics.values()]
    if not metric_rows:
        return {
            "trip_count": 0,
            "total_observations": 0,
            "valid_labels": 0,
            "pointwise_accuracy": 0.0,
            "pointwise_precision": 0.0,
            "pointwise_recall": 0.0,
            "route_continuity": 1.0,
        }

    predictions = pd.concat(prediction_frames, ignore_index=True)
    truth = predictions.loc[predictions["true_arc_id"].notna()]
    true_keys = [
        (trip_id, arc_id)
        for trip_id, arc_id in zip(truth["trip_id"], truth["true_arc_id"])
    ]
    prediction_column = (
        "classifier_pred_arc_id" if method == "classifier_only" else "hmm_pred_arc_id"
    )
    predicted_keys = [
        (trip_id, prediction)
        for trip_id, prediction in zip(
            truth["trip_id"], truth[prediction_column]
        )
        if pd.notna(prediction)
    ]
    set_metrics = sequence_metrics(true_keys, predicted_keys)
    valid_labels = sum(int(row["valid_labels"]) for row in metric_rows)
    correct = sum(int(row["correct_predictions"]) for row in metric_rows)
    predicted_on_valid = int(
        predictions.loc[
            predictions["true_arc_id"].notna(), prediction_column
        ].notna().sum()
    )
    continuity_pairs = sum(int(row["_continuity_pairs"]) for row in metric_rows)
    continuity_total = sum(
        float(row["route_continuity"]) * int(row["_continuity_pairs"])
        for row in metric_rows
    )
    total_observations = sum(int(row["total_observations"]) for row in metric_rows)

    return {
        "trip_count": len(trip_metrics),
        "total_observations": total_observations,
        "valid_labels": valid_labels,
        "valid_label_coverage": (
            valid_labels / total_observations
            if total_observations
            else 0.0
        ),
        "predicted_observations": sum(
            int(row["predicted_observations"]) for row in metric_rows
        ),
        "null_predictions": sum(
            int(row["null_predictions"]) for row in metric_rows
        ),
        "candidate_hits": sum(int(row["candidate_hits"]) for row in metric_rows),
        "candidate_misses": sum(
            int(row["candidate_misses"]) for row in metric_rows
        ),
        "candidate_recall": (
            sum(int(row["candidate_hits"]) for row in metric_rows) / valid_labels
            if valid_labels
            else 0.0
        ),
        "correct_predictions": correct,
        "pointwise_accuracy": correct / valid_labels if valid_labels else 0.0,
        "pointwise_precision": (
            correct / predicted_on_valid if predicted_on_valid else 0.0
        ),
        "pointwise_recall": correct / valid_labels if valid_labels else 0.0,
        "arc_set_precision": float(set_metrics["arc_set_precision"]),
        "arc_set_recall": float(set_metrics["arc_set_recall"]),
        "edit_distance": sum(int(row["edit_distance"]) for row in metric_rows),
        "longest_correct_run": max(
            int(row["longest_correct_run"]) for row in metric_rows
        ),
        "route_continuity": (
            continuity_total / continuity_pairs if continuity_pairs else 1.0
        ),
        "unreachable_predicted_transitions": sum(
            int(row["unreachable_predicted_transitions"])
            for row in metric_rows
        ),
    }


def evaluate_student_test(
    repo_root: str | Path,
    *,
    model_path: str | Path | None = None,
    transition_scale_m: float = 20.0,
    hop_limit: int = DEFAULT_HOP_LIMIT,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate the saved model on every root-splits.json student_test trip."""
    root = Path(repo_root)
    project = root / "pathsnap"
    split_data = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    if not isinstance(split_data, dict) or not {
        "train",
        "student_test",
    }.issubset(split_data):
        raise ValueError("Root splits.json must contain train and student_test lists")
    train_ids = [_canonical_trip_id(value) for value in split_data["train"]]
    student_ids = [
        _canonical_trip_id(value) for value in split_data["student_test"]
    ]
    if set(train_ids) & set(student_ids):
        raise ValueError("Root train and student_test splits overlap")
    if not student_ids or len(student_ids) != len(set(student_ids)):
        raise ValueError("Root student_test split must be nonempty and unique")

    labels_path = project / "results" / "labels_student.parquet"
    labels = pd.read_parquet(labels_path)
    required_labels = {"trip_id", "obs_idx", "true_arc_id"}
    if missing := required_labels - set(labels.columns):
        raise ValueError(f"Student label file is missing columns: {sorted(missing)}")
    labels["trip_id"] = labels["trip_id"].map(_canonical_trip_id)
    if set(labels["trip_id"]) != set(student_ids):
        raise ValueError("Student labels do not match root splits.json student_test IDs")

    bundle_file = Path(model_path) if model_path else project / "artifacts" / "model_bundle.joblib"
    if not bundle_file.is_file():
        raise FileNotFoundError(f"Trained model bundle not found: {bundle_file}")
    model_bundle = joblib.load(bundle_file)
    if not callable(getattr(model_bundle, "predict_proba", None)):
        raise TypeError("Model bundle must expose predict_proba(cands_df)")
    model_features = tuple(getattr(model_bundle, "feature_cols", ()))
    if model_features != tuple(FEATURE_COLS):
        raise ValueError(
            f"Model feature schema {model_features} does not match preprocessing "
            f"schema {tuple(FEATURE_COLS)}"
        )

    output_root = root / "member1_preprocessing" / "outputs"
    per_trip: dict[str, dict[str, Any]] = {}
    prediction_frames: list[pd.DataFrame] = []
    imputed_counts: dict[str, int] = {}
    for trip_id in student_ids:
        candidate_path = output_root / "candidates" / f"{trip_id}.csv"
        feature_path = output_root / "features" / f"{trip_id}.csv"
        geometry_path = output_root / "road_geometry" / f"{trip_id}.npz"
        for path in (candidate_path, feature_path, geometry_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing student evaluation input: {path}")

        source_candidates = pd.read_csv(
            candidate_path, usecols=["obs_idx", "arc_id"]
        )
        source_candidates["trip_id"] = trip_id
        source_candidates = _integer_keys(source_candidates, "source_candidates")
        if source_candidates.duplicated(_KEY_COLUMNS).any():
            raise ValueError(f"Trip {trip_id} has duplicate source candidate rows")

        trip_imputed_counts: dict[str, int] = {}
        model_frame = _load_trip_features(
            trip_id,
            output_root / "features",
            "student_test",
            trip_imputed_counts,
        )
        for key, count in trip_imputed_counts.items():
            imputed_counts[key] = imputed_counts.get(key, 0) + count
        if set(map(tuple, source_candidates[_KEY_COLUMNS].to_numpy())) != set(
            map(tuple, model_frame[_KEY_COLUMNS].to_numpy())
        ):
            raise ValueError(
                f"Trip {trip_id} candidate CSV and model feature keys do not match"
            )
        model_frame = source_candidates[_KEY_COLUMNS].merge(
            model_frame[_KEY_COLUMNS + list(FEATURE_COLS)],
            on=_KEY_COLUMNS,
            how="left",
            validate="one_to_one",
            sort=False,
        ).reset_index(drop=True)
        if not model_frame[_KEY_COLUMNS].equals(
            source_candidates[_KEY_COLUMNS].reset_index(drop=True)
        ):
            raise ValueError(f"Trip {trip_id} preprocessing changed candidate order")

        raw_features = pd.read_csv(feature_path)
        if missing := set(_KEY_COLUMNS + _TRANSITION_COLUMNS) - set(
            raw_features.columns
        ):
            raise ValueError(
                f"Trip {trip_id} feature CSV is missing HMM fields: {sorted(missing)}"
            )
        raw_features["trip_id"] = raw_features["trip_id"].map(_canonical_trip_id)
        raw_features = _integer_keys(raw_features, "raw_features")
        trip_labels = labels[labels["trip_id"].eq(trip_id)].copy()
        graph = DirectedRoadGraph.from_npz(geometry_path)
        trip_metrics, trip_predictions = evaluate_ml_trip_frames(
            trip_id,
            model_frame,
            raw_features,
            trip_labels,
            graph,
            model_bundle,
            transition_scale_m=transition_scale_m,
            hop_limit=hop_limit,
        )
        trip_metrics["imputed_feature_values"] = int(sum(trip_imputed_counts.values()))
        per_trip[trip_id] = trip_metrics
        prediction_frames.append(trip_predictions)

    overall = {
        method: _overall_metrics(per_trip, prediction_frames, method)
        for method in ("classifier_only", "hmm_viterbi")
    }
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.sort_values(
        ["trip_id", "obs_idx"], kind="mergesort"
    ).reset_index(drop=True)
    report = {
        "schema_version": 1,
        "model_bundle": str(bundle_file),
        "configuration": {
            "student_test_trip_count": len(student_ids),
            "transition_scale_m": float(transition_scale_m),
            "route_continuity_hop_limit": hop_limit,
            "break_on_long_gap": True,
            "restart_on_unreachable_transition": True,
            "student_test_labels_used_for_training_or_tuning": False,
            "model_retrained": False,
            "feature_columns": list(FEATURE_COLS),
            "imputed_feature_values": imputed_counts,
        },
        "metric_definitions": {
            "pointwise_accuracy": "Exact arc matches divided by observations with non-null DP labels; null predictions count as incorrect.",
            "pointwise_precision": "Exact arc matches divided by non-null predictions on observations with non-null DP labels.",
            "pointwise_recall": "Exact arc matches divided by observations with non-null DP labels.",
            "arc_set_precision_recall": "Unique-arc set precision and recall from sequence_metrics, scoped by trip in aggregate metrics.",
            "route_continuity": "Fraction of consecutive predicted-arc pairs connected within the configured directed hop limit; long-gap and missing-prediction boundaries start new segments.",
            "edit_distance": "Per-trip Levenshtein distance over valid-label observations; null predictions are represented as missing states.",
            "longest_correct_run": "Longest chronologically aligned run of exact predictions over valid-label observations.",
            "classifier_only": "Per-observation argmax of the trained bundle's normalized candidate probabilities.",
            "hmm_viterbi": "Viterbi decoding using those same trained-model probabilities as emissions and the existing directed-road transition model; decoding restarts at an observation when no incoming transition is valid.",
        },
        "student_test": {
            "complete": len(per_trip) == len(student_ids),
            "requested_trip_count": len(student_ids),
            "processed_trip_count": len(per_trip),
            "total_candidate_rows": sum(
                item["candidate_rows"] for item in per_trip.values()
            ),
            "overall": overall,
            "per_trip": per_trip,
        },
    }
    return report, predictions


def save_evaluation_outputs(
    report: dict[str, Any],
    predictions: pd.DataFrame,
    report_path: str | Path,
    predictions_path: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    report_file = Path(report_path)
    predictions_file = Path(predictions_path)
    existing = [path for path in (report_file, predictions_file) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Evaluation output already exists; pass overwrite=True to replace: "
            + ", ".join(map(str, existing))
        )
    report_file.parent.mkdir(parents=True, exist_ok=True)
    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    predictions.to_parquet(predictions_file, index=False)
    return report_file, predictions_file


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--maps-dir", type=Path)
    parser.add_argument("--transition-scale-m", type=float, default=20.0)
    parser.add_argument("--hop-limit", type=int, default=DEFAULT_HOP_LIMIT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    results_dir = args.repo_root / "pathsnap" / "results"
    report_path = args.report or results_dir / "hmm_ml_evaluation_report.json"
    predictions_path = (
        args.predictions or results_dir / "hmm_ml_predictions_student.parquet"
    )
    existing_outputs = [
        path for path in (report_path, predictions_path) if path.exists()
    ]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            "Evaluation output already exists; choose new output paths or pass "
            "--overwrite: "
            + ", ".join(map(str, existing_outputs))
        )

    from src.viz.make_map import DEFAULT_DATA_ROOT, generate_hmm_trip_map

    data_root = args.data_root or DEFAULT_DATA_ROOT
    maps_dir = args.maps_dir or results_dir / "maps"
    report, predictions = evaluate_student_test(
        args.repo_root,
        model_path=args.model,
        transition_scale_m=args.transition_scale_m,
        hop_limit=args.hop_limit,
    )
    map_results = []
    for trip_id, trip_predictions in predictions.groupby("trip_id", sort=False):
        trip_dir = data_root / "student_test" / trip_id
        required_inputs = [
            trip_dir / f"{trip_id}.track",
            trip_dir / f"{trip_id}.nodes",
            trip_dir / f"{trip_id}.route",
        ]
        missing_inputs = [str(path) for path in required_inputs if not path.is_file()]
        geometry_path = (
            args.repo_root
            / "member1_preprocessing"
            / "outputs"
            / "road_geometry"
            / f"{trip_id}.npz"
        )
        if not (trip_dir / f"{trip_id}.arcs").is_file() and not geometry_path.is_file():
            missing_inputs.append(
                f"{trip_dir / f'{trip_id}.arcs'} or {geometry_path}"
            )
        map_path = maps_dir / f"hmm_trip_{trip_id}.html"
        if missing_inputs:
            map_results.append(
                {
                    "trip_id": trip_id,
                    "status": "skipped_missing_inputs",
                    "missing_inputs": missing_inputs,
                }
            )
            continue
        if map_path.exists() and not args.overwrite:
            map_results.append(
                {
                    "trip_id": trip_id,
                    "status": "preserved_existing",
                    "output_path": str(map_path),
                }
            )
            continue
        _, map_stats = generate_hmm_trip_map(
            trip_id,
            trip_predictions,
            repo_root=args.repo_root,
            data_root=data_root,
            output_dir=maps_dir,
            overwrite=args.overwrite,
        )
        map_results.append(
            {"trip_id": trip_id, "status": "generated", **map_stats}
        )
    report["student_test"]["maps"] = {
        "requested_trip_count": len(map_results),
        "generated_count": sum(
            result["status"] == "generated" for result in map_results
        ),
        "skipped_count": sum(
            result["status"] == "skipped_missing_inputs" for result in map_results
        ),
        "results": map_results,
    }
    saved_report, saved_predictions = save_evaluation_outputs(
        report,
        predictions,
        report_path,
        predictions_path,
        overwrite=args.overwrite,
    )
    print(f"Saved evaluation report: {saved_report}")
    print(f"Saved aligned predictions: {saved_predictions}")
    print(json.dumps(report["student_test"]["maps"], indent=2))
    print(json.dumps(report["student_test"]["overall"], indent=2))


if __name__ == "__main__":
    main()
