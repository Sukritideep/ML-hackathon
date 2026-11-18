from typing import Any

from pydantic import BaseModel


class ServerStatus(BaseModel):
    name: str = "PathSnap"
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    trip_loader_available: bool
    candidate_data_available: bool
    model_available: bool
    hmm_available: bool
    evaluation_available: bool


class TripListResponse(BaseModel):
    trips: list[str]


class TripResponse(BaseModel):
    trip_id: str
    gps: list[list[Any]]


class EvaluationResponse(BaseModel):
    trip_id: str
    original_gps: list[list[Any]]
    classifier_prediction: list[list[Any]] | None = None
    hmm_prediction: list[list[Any]] | None = None
    classifier_route: list[list[float]] | None = None
    hmm_route: list[list[float]] | None = None
    metrics: dict[str, Any] | None = None
    processing_time_seconds: float
