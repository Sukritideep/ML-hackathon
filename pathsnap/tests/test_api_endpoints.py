import json
from types import SimpleNamespace

import numpy as np

from fastapi.testclient import TestClient

from src.api import main
from src.api.pipeline import PipelineUnavailableError, SPLITS_FILE

client = TestClient(main.app)


def test_status_and_health_endpoints_report_readiness():
    assert client.get("/").status_code == 200

    response = client.get("/health")
    assert response.status_code == 200
    assert {
        "trip_loader_available",
        "candidate_data_available",
        "model_available",
        "hmm_available",
        "evaluation_available",
    } <= response.json().keys()


def test_trip_list_contains_only_student_test_ids():
    splits = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))

    response = client.get("/trips")

    assert response.status_code == 200
    assert response.json()["trips"] == splits["student_test"]


def test_trip_endpoint_returns_503_when_raw_trip_data_is_unavailable(monkeypatch):
    def unavailable_trip_loader(trip_id):
        raise PipelineUnavailableError("raw data missing")

    monkeypatch.setattr(main, "load_trip", unavailable_trip_loader)

    response = client.get("/trips/00000000")

    assert response.status_code == 503
    assert response.json()["detail"] == "raw data missing"


def test_trip_endpoint_returns_gps_without_reference_route_or_full_network(
    monkeypatch,
):
    trip = SimpleNamespace(
        gps=np.array([[12.5, 48.1, 1.0]]),
        nodes=np.array([[12.5, 48.1], [12.6, 48.2]]),
        arcs=np.array([[0, 1]]),
        route=np.array([0]),
    )
    monkeypatch.setattr(main, "load_trip", lambda trip_id: trip)

    response = client.get("/trips/00000000")

    assert response.status_code == 200
    assert response.json() == {
        "trip_id": "00000000",
        "gps": [[12.5, 48.1, 1.0]],
    }


def test_evaluation_endpoint_does_not_return_empty_success(monkeypatch):
    def unavailable_evaluation(trip_id):
        raise PipelineUnavailableError("pipeline is not configured")

    monkeypatch.setattr(main, "evaluate_trip", unavailable_evaluation)

    response = client.post("/evaluate/00000000")

    assert response.status_code == 503
    assert response.json()["detail"] == "pipeline is not configured"


def test_evaluation_endpoint_rejects_non_permitted_trip():
    response = client.post("/evaluate/not-a-trip")

    assert response.status_code == 404
