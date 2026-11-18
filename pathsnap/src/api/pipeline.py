from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..core import io
from . import candidate_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_FILE = PROJECT_ROOT / "splits.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "model_bundle.joblib"
TRANSITION_SCALE_M = 20.0


class PipelineUnavailableError(RuntimeError):
    pass


class InvalidTripError(ValueError):
    pass


def permitted_trip_ids() -> list[str]:
    if not SPLITS_FILE.exists():
        raise PipelineUnavailableError("Evaluation splits file is missing.")

    try:
        with SPLITS_FILE.open("r", encoding="utf-8") as handle:
            splits = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineUnavailableError(
            f"Unable to read evaluation splits: {exc}"
        ) from exc

    if not isinstance(splits, dict) or not isinstance(splits.get("student_test"), list):
        raise PipelineUnavailableError(
            "student_test split must contain a list of trip IDs."
        )
    trips = splits["student_test"]
    if not all(isinstance(trip_id, str) and trip_id for trip_id in trips):
        raise PipelineUnavailableError("student_test contains an invalid trip ID.")
    return trips


def trip_loader_available() -> bool:
    root = io.data_root()
    return any(
        all(
            (root / "student_test" / trip_id / f"{trip_id}.{extension}").is_file()
            for extension in ("track", "nodes", "arcs")
        )
        for trip_id in permitted_trip_ids()
    )


def candidate_data_available() -> bool:
    trips = permitted_trip_ids()
    return bool(trips) and all(
        candidate_data.trip_feature_data_available(trip_id) for trip_id in trips
    )


def model_available() -> bool:
    path = Path(os.environ.get("PATHSNAP_MODEL_PATH", DEFAULT_MODEL_PATH)).expanduser()
    return path.is_file()


def load_model_bundle() -> Any:
    path = Path(os.environ.get("PATHSNAP_MODEL_PATH", DEFAULT_MODEL_PATH)).expanduser()
    if not path.is_file():
        raise PipelineUnavailableError(f"Classifier bundle is missing: {path}")
    try:
        bundle = joblib.load(path)
    except (OSError, ValueError, ImportError) as exc:
        raise PipelineUnavailableError(
            f"Unable to load classifier bundle {path}: {exc}"
        ) from exc
    if not callable(getattr(bundle, "predict_proba", None)):
        raise PipelineUnavailableError(
            "Classifier bundle must expose predict_proba(cands_df)."
        )
    return bundle


def load_trip(trip_id: str) -> Any:
    permitted = permitted_trip_ids()
    if trip_id not in permitted:
        raise InvalidTripError(f"Trip '{trip_id}' is not a permitted evaluation trip.")

    if not trip_loader_available():
        raise PipelineUnavailableError(
            "Raw trip data is unavailable. Set PATHSNAP_DATA_DIR to the dataset root."
        )

    try:
        return io.load_trip(trip_id, with_route=False)
    except FileNotFoundError as exc:
        raise PipelineUnavailableError(str(exc)) from exc


def evaluation_available() -> bool:
    if not candidate_data_available() or not model_available():
        return False

    try:
        from ..hmm import viterbi as hmm_module

        return callable(getattr(hmm_module, "decode", None))
    except ImportError:
        return False


def hmm_available() -> bool:
    try:
        from ..hmm import viterbi as hmm_module
    except ImportError:
        return False
    return callable(getattr(hmm_module, "decode", None))


def evaluate_trip(trip_id: str) -> dict[str, Any]:
    if trip_id not in permitted_trip_ids():
        raise InvalidTripError(f"Trip '{trip_id}' is not a permitted evaluation trip.")
    if not model_available():
        raise PipelineUnavailableError("No trained classifier bundle is available.")
    if not candidate_data.trip_feature_data_available(trip_id):
        raise PipelineUnavailableError(
            f"Candidate features, road geometry, or CRS metadata are unavailable "
            f"for trip {trip_id}."
        )
    if not hmm_available():
        raise PipelineUnavailableError("HMM/Viterbi decoding is unavailable.")

    started_at = time.perf_counter()
    try:
        from ..hmm.transitions import DirectedRoadGraph
        from ..hmm.viterbi import ViterbiParams, decode

        candidates = candidate_data.load_trip_candidates(trip_id)
        bundle = load_model_bundle()
        probabilities = np.asarray(bundle.predict_proba(candidates), dtype=float)
        if (
            probabilities.ndim != 1
            or len(probabilities) != len(candidates)
            or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
        ):
            raise PipelineUnavailableError(
                "Classifier returned invalid candidate probabilities."
            )

        scored = candidates.assign(_probability=probabilities)
        classifier_rows = (
            scored.sort_values(
                ["obs_idx", "_probability", "arc_id"],
                ascending=[True, False, True],
                kind="mergesort",
            )
            .groupby("obs_idx", sort=True, as_index=False)
            .head(1)
        )
        classifier_predictions = [
            [int(row.obs_idx), int(row.arc_id)] for row in classifier_rows.itertuples()
        ]

        graph_path = candidate_data.road_geometry_file(trip_id)
        graph = DirectedRoadGraph.from_npz(graph_path)
        decoded = decode(
            candidates,
            probabilities,
            graph,
            ViterbiParams(transition_scale_m=TRANSITION_SCALE_M),
        )
        hmm_predictions = [
            [
                int(row.obs_idx),
                None if pd.isna(row.pred_arc_id) else int(row.pred_arc_id),
            ]
            for row in decoded.itertuples()
        ]
        classifier_route = candidate_data.route_coordinates(
            trip_id, candidates, [(row[0], row[1]) for row in classifier_predictions]
        )
        hmm_route = candidate_data.route_coordinates(
            trip_id, candidates, [(row[0], row[1]) for row in hmm_predictions]
        )

        original_gps: list[list[float]] = []
        trip_dir = io.data_root() / "student_test" / trip_id
        if all(
            (trip_dir / f"{trip_id}.{extension}").is_file()
            for extension in ("track", "nodes", "arcs")
        ):
            original_gps = io.load_trip(trip_id, with_route=False).gps.tolist()

        return {
            "trip_id": trip_id,
            "original_gps": original_gps,
            "classifier_prediction": classifier_predictions,
            "hmm_prediction": hmm_predictions,
            "classifier_route": classifier_route,
            "hmm_route": hmm_route,
            "metrics": None,
            "processing_time_seconds": time.perf_counter() - started_at,
        }
    except PipelineUnavailableError:
        raise
    except (OSError, ValueError, ImportError) as exc:
        raise PipelineUnavailableError(
            f"Unable to evaluate trip {trip_id}: {exc}"
        ) from exc
