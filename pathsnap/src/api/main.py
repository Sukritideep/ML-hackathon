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
    evaluate_trip,
    load_trip,
    model_available,
    permitted_trip_ids,
    trip_loader_available,
)


app = FastAPI(
    title="PathSnap API",
    description="Map-matching evaluation API",
    version="0.1.0",
)


@app.get("/", response_model=ServerStatus)
def server_status() -> ServerStatus:
    return ServerStatus(
        status="running",
        message="PathSnap API is ready. Use /docs for endpoint documentation.",
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loader_ready = trip_loader_available()
    model_ready = model_available()
    status = "ready" if loader_ready and model_ready else "degraded"
    return HealthResponse(
        status=status,
        trip_loader_available=loader_ready,
        model_available=model_ready,
    )


@app.get("/trips", response_model=TripListResponse)
def list_trips() -> TripListResponse:
    try:
        return TripListResponse(trips=permitted_trip_ids())
    except PipelineUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unable to load the evaluation trip list.") from exc


@app.get("/trips/{trip_id}", response_model=TripResponse)
def get_trip(trip_id: str) -> TripResponse:
    try:
        trip = load_trip(trip_id)
        gps = getattr(trip, "gps", None)
        nodes = getattr(trip, "nodes", None)
        arcs = getattr(trip, "arcs", None)

        if gps is None:
            raise PipelineUnavailableError("Loaded trip does not contain GPS observations.")

        def to_lists(value):
            if hasattr(value, "tolist"):
                return value.tolist()
            if isinstance(value, list):
                return value
            return list(value)

        return TripResponse(
            trip_id=trip_id,
            gps=to_lists(gps),
            nodes=to_lists(nodes) if nodes is not None else None,
            arcs=to_lists(arcs) if arcs is not None else None,
        )
    except InvalidTripError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PipelineUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unable to load the selected trip.") from exc


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
        raise HTTPException(status_code=500, detail="Unexpected evaluation pipeline failure.") from exc