from __future__ import annotations

import importlib
from pathlib import Path
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_FILE = PROJECT_ROOT / "splits.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


class PipelineUnavailableError(RuntimeError):
    pass


class InvalidTripError(ValueError):
    pass


def permitted_trip_ids() -> list[str]:
    import json

    if not SPLITS_FILE.exists():
        raise PipelineUnavailableError("Evaluation splits file is missing.")

    with SPLITS_FILE.open("r", encoding="utf-8") as handle:
        splits = json.load(handle)

    trips = splits.get("student_test", [])
    if not isinstance(trips, list):
        raise PipelineUnavailableError("student_test split must contain a list of trip IDs.")
    return [str(trip_id) for trip_id in trips]


def trip_loader_available() -> bool:
    try:
        io_module = importlib.import_module("pathsnap.src.core.io")
        return callable(getattr(io_module, "load_trip", None))
    except (ImportError, AttributeError):
        return False


def model_available() -> bool:
    candidates = sorted(ARTIFACTS_DIR.glob("*.joblib")) + sorted(
        ARTIFACTS_DIR.glob("*.pkl")
    )
    return bool(candidates)


def load_trip(trip_id: str) -> Any:
    permitted = permitted_trip_ids()
    if trip_id not in permitted:
        raise InvalidTripError(f"Trip '{trip_id}' is not a permitted evaluation trip.")

    if not trip_loader_available():
        raise PipelineUnavailableError("The trip loader is not implemented or unavailable.")

    io_module = importlib.import_module("pathsnap.src.core.io")
    return io_module.load_trip(trip_id, with_route=False)


def evaluation_available() -> bool:
    if not trip_loader_available() or not model_available():
        return False

    try:
        candidates_module = importlib.import_module("pathsnap.src.features.candidates")
        hmm_module = importlib.import_module("pathsnap.src.hmm.viterbi")
        return callable(getattr(candidates_module, "build_candidates", None)) and callable(
            getattr(hmm_module, "decode", None)
        )
    except (ImportError, AttributeError):
        return False


def evaluate_trip(trip_id: str) -> dict[str, Any]:
    started_at = time.perf_counter()

    if trip_id not in permitted_trip_ids():
        raise InvalidTripError(f"Trip '{trip_id}' is not a permitted evaluation trip.")
    if not trip_loader_available():
        raise PipelineUnavailableError("The trip loader is not implemented or unavailable.")
    if not model_available():
        raise PipelineUnavailableError("No trained classifier artifact is available.")
    if not evaluation_available():
        raise PipelineUnavailableError(
            "Candidate generation or HMM/Viterbi decoding is not implemented."
        )

    elapsed = time.perf_counter() - started_at
    return {
        "trip_id": trip_id,
        "original_gps": [],
        "classifier_prediction": None,
        "hmm_prediction": None,
        "metrics": None,
        "processing_time_seconds": elapsed,
    }