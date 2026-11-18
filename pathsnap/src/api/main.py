import logging

from fastapi import FastAPI, HTTPException

from .evaluation import (
    EvaluationResponse,
    HealthResponse,
    ServerStatus,
    TripListResponse,
    TripResponse,
)
from .pipeline import (
    InvalidTripError,
    PipelineUnavailableError,
    candidate_data_available,
    evaluate_trip,
    evaluation_available,
    hmm_available,
    load_trip,
    model_available,
    permitted_trip_ids,
    trip_loader_available,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="PathSnap API",
    description="Map-matching evaluation API",
    version="0.1.0",
)


@app.get("/", response_model=ServerStatus)
def server_status() -> ServerStatus:
    return ServerStatus(
        status="running",
        message="PathSnap API is running. Use /health to check pipeline readiness.",
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loader_ready = trip_loader_available()
    model_ready = model_available()
    decoder_ready = hmm_available()
    pipeline_ready = evaluation_available()
    status = "ready" if pipeline_ready else "degraded"
    return HealthResponse(
        status=status,
        trip_loader_available=loader_ready,
        candidate_data_available=candidate_data_available(),
        model_available=model_ready,
        hmm_available=decoder_ready,
        evaluation_available=pipeline_ready,
    )


@app.get("/trips", response_model=TripListResponse)
def list_trips() -> TripListResponse:
    try:
        return TripListResponse(trips=permitted_trip_ids())
    except PipelineUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while loading evaluation trip IDs")
        raise HTTPException(
            status_code=500, detail="Unable to load the evaluation trip list."
        ) from exc


@app.get("/trips/{trip_id}", response_model=TripResponse)
def get_trip(trip_id: str) -> TripResponse:
    try:
        trip = load_trip(trip_id)
        gps = getattr(trip, "gps", None)

        if gps is None:
            raise PipelineUnavailableError(
                "Loaded trip does not contain GPS observations."
            )

        gps_values = gps.tolist() if hasattr(gps, "tolist") else list(gps)
        return TripResponse(
            trip_id=trip_id,
            gps=gps_values,
        )
    except InvalidTripError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PipelineUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while loading trip %s", trip_id)
        raise HTTPException(
            status_code=500, detail="Unable to load the selected trip."
        ) from exc


@app.post("/evaluate/{trip_id}", response_model=EvaluationResponse)
def evaluate(trip_id: str) -> EvaluationResponse:
    try:
        result = evaluate_trip(trip_id)
        return EvaluationResponse(**result)
    except InvalidTripError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PipelineUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected evaluation pipeline failure for trip %s", trip_id)
        raise HTTPException(
            status_code=500, detail="Unexpected evaluation pipeline failure."
        ) from exc
