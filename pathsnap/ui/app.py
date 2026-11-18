import os
import math
from typing import Any
from urllib.parse import urlparse

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

DEFAULT_API_URL = os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_SECONDS = 90


def api_request(method: str, base_url: str, path: str) -> tuple[bool, Any]:
    parsed_url = urlparse(base_url.strip())
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        return False, "Enter a valid FastAPI URL, including http:// or https://."

    try:
        response = requests.request(
            method,
            f"{base_url.rstrip('/')}{path}",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except (AttributeError, ValueError):
            detail = exc.response.text if exc.response is not None else str(exc)
        return False, f"HTTP {exc.response.status_code}: {detail}"
    except requests.Timeout:
        return False, f"Request timed out after {REQUEST_TIMEOUT_SECONDS} seconds."
    except requests.ConnectionError:
        return (
            False,
            "Cannot connect to the PathSnap API. Check the server URL and network.",
        )
    except requests.RequestException as exc:
        return False, str(exc)

    try:
        payload = response.json()
    except ValueError:
        return False, "The PathSnap API returned an invalid JSON response."
    if not isinstance(payload, dict):
        return False, "The PathSnap API returned an unexpected response format."
    return True, payload


def api_get(base_url: str, path: str) -> tuple[bool, Any]:
    return api_request("GET", base_url, path)


def api_post(base_url: str, path: str) -> tuple[bool, Any]:
    return api_request("POST", base_url, path)


def gps_locations(gps: list[list[Any]]) -> list[list[float]]:
    locations: list[list[float]] = []
    for point in gps:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        lon, lat = point[0], point[1]
        try:
            lon_value, lat_value = float(lon), float(lat)
        except (TypeError, ValueError):
            continue
        if (
            math.isfinite(lon_value)
            and math.isfinite(lat_value)
            and -180 <= lon_value <= 180
            and -90 <= lat_value <= 90
        ):
            locations.append([lat_value, lon_value])
    return locations


def route_locations(route: list[list[Any]] | None) -> list[list[float]]:
    locations: list[list[float]] = []
    for point in route or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        lon, lat = point[0], point[1]
        try:
            lon_value, lat_value = float(lon), float(lat)
        except (TypeError, ValueError):
            continue
        if (
            math.isfinite(lon_value)
            and math.isfinite(lat_value)
            and -180 <= lon_value <= 180
            and -90 <= lat_value <= 90
        ):
            locations.append([lat_value, lon_value])
    return locations


def prediction_count(predictions: list[list[Any]] | None) -> int:
    return sum(
        1
        for prediction in predictions or []
        if isinstance(prediction, (list, tuple))
        and len(prediction) > 1
        and prediction[1] is not None
    )


def create_map(
    gps: list[list[float]],
    classifier: list[list[float]] | None = None,
    hmm: list[list[float]] | None = None,
) -> folium.Map:
    all_locations = gps + (classifier or []) + (hmm or [])
    if all_locations:
        center_lat = sum(point[0] for point in all_locations) / len(all_locations)
        center_lon = sum(point[1] for point in all_locations) / len(all_locations)
    else:
        center_lat, center_lon = 20.5937, 78.9629

    route_map = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    if gps:
        gps_layer = folium.FeatureGroup(name="GPS observations", show=True)
        folium.PolyLine(
            gps, color="#2563eb", weight=3, opacity=0.85, tooltip="GPS trajectory"
        ).add_to(gps_layer)
        for location in gps:
            folium.CircleMarker(
                location=location,
                radius=3.5,
                color="#2563eb",
                fill=True,
                fill_opacity=0.85,
                tooltip="GPS observation",
            ).add_to(gps_layer)
        gps_layer.add_to(route_map)
    if classifier:
        classifier_layer = folium.FeatureGroup(name="Classifier route", show=True)
        folium.PolyLine(
            classifier,
            color="#f97316",
            weight=4.5,
            opacity=0.9,
            tooltip="Classifier prediction",
        ).add_to(classifier_layer)
        classifier_layer.add_to(route_map)
    if hmm:
        hmm_layer = folium.FeatureGroup(name="HMM/Viterbi route", show=True)
        folium.PolyLine(
            hmm,
            color="#16a34a",
            weight=5,
            opacity=0.9,
            tooltip="HMM/Viterbi route",
        ).add_to(hmm_layer)
        hmm_layer.add_to(route_map)
    if all_locations:
        route_map.fit_bounds(
            [
                [
                    min(point[0] for point in all_locations),
                    min(point[1] for point in all_locations),
                ],
                [
                    max(point[0] for point in all_locations),
                    max(point[1] for point in all_locations),
                ],
            ],
            padding=(24, 24),
        )
    folium.LayerControl(collapsed=False).add_to(route_map)
    return route_map


st.set_page_config(
    page_title="PathSnap Evaluator", page_icon=":world_map:", layout="wide"
)
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: rgba(120, 120, 120, 0.06);
        border: 1px solid rgba(120, 120, 120, 0.18);
        padding: 0.8rem 1rem;
        border-radius: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("PathSnap")
st.markdown("#### GPS map-matching evaluator")
st.caption(
    "Compare noisy GPS observations with the classifier's road choices and "
    "the HMM/Viterbi reconstructed route."
)

with st.sidebar:
    st.header("Server connection")
    api_url = st.text_input(
        "FastAPI server URL",
        value=DEFAULT_API_URL,
        help="On another laptop, use the FastAPI server's LAN IP, for example http://192.168.1.20:8000.",
    )

    status_ok, status = api_get(api_url, "/")
    health_ok, health = api_get(api_url, "/health")
    if status_ok:
        st.success("API online")
    else:
        st.error(status)

    if health_ok:
        if health.get("status") == "ready":
            st.success("Pipeline ready")
        else:
            st.warning("API connected; evaluation is not fully ready")
        for label, key in (
            ("Raw GPS data", "trip_loader_available"),
            ("Candidate features and road graph", "candidate_data_available"),
            ("Classifier", "model_available"),
            ("HMM decoder", "hmm_available"),
        ):
            state = "Ready" if health.get(key) else "Unavailable"
            st.caption(f"{label}: **{state}**")
    else:
        st.error(health)
    with st.expander("Connection help"):
        st.markdown(
            "Run FastAPI on the server laptop with `--host 0.0.0.0`. "
            "On the client laptop, enter the server's LAN URL "
            "(for example `http://192.168.1.20:8000`)."
        )

trips_ok, trips_response = api_get(api_url, "/trips")
if not trips_ok:
    st.error(f"Could not load evaluation trips: {trips_response}")
    st.stop()

trip_ids = trips_response.get("trips", [])
if not trip_ids:
    st.info("No permitted evaluation trips are currently available.")
    st.stop()

trip_column, button_column = st.columns([3, 1], vertical_alignment="bottom")
with trip_column:
    selected_trip = st.selectbox("Evaluation trip", trip_ids)
with button_column:
    run_clicked = st.button(
        "Run evaluation",
        type="primary",
        use_container_width=True,
    )

if run_clicked:
    with st.spinner("Loading trip and running evaluation..."):
        trip_ok, trip = api_get(api_url, f"/trips/{selected_trip}")
        result = {
            "api_url": api_url,
            "trip_id": selected_trip,
            "trip": trip if trip_ok else None,
            "trip_error": None if trip_ok else trip,
            "evaluation": None,
            "evaluation_error": None,
        }
        evaluate_ok, evaluation = api_post(api_url, f"/evaluate/{selected_trip}")
        result["evaluation"] = evaluation if evaluate_ok else None
        result["evaluation_error"] = None if evaluate_ok else evaluation
        st.session_state["last_evaluation"] = result

result = st.session_state.get("last_evaluation")
if result is None or result["api_url"] != api_url or result["trip_id"] != selected_trip:
    st.info(
        "Choose an evaluation trip and run the evaluation to see its map and results."
    )
elif result["trip_error"] and not result["evaluation"]:
    st.error(f"Could not load raw GPS observations: {result['trip_error']}")
    if result["evaluation_error"]:
        st.error(f"Evaluation unavailable: {result['evaluation_error']}")
else:
    trip = result["trip"] or {}
    evaluation = result["evaluation"]
    gps = gps_locations((evaluation or {}).get("original_gps", trip.get("gps", [])))
    classifier = (
        route_locations(evaluation.get("classifier_route")) if evaluation else []
    )
    hmm = route_locations(evaluation.get("hmm_route")) if evaluation else []

    metrics = evaluation.get("metrics", {}) if evaluation else {}
    st.markdown(f"**Trip:** `{result['trip_id']}`")
    if result["trip_error"]:
        st.warning(
            "Raw GPS observations are unavailable. Showing predictions from the "
            "precomputed candidate features instead."
        )
    metric_columns = st.columns(4)
    metric_columns[0].metric("GPS observations", f"{len(gps):,}")
    metric_columns[1].metric(
        "Classifier arcs",
        (
            f"{prediction_count(evaluation.get('classifier_prediction')):,}"
            if evaluation
            else "—"
        ),
    )
    metric_columns[2].metric(
        "HMM arcs",
        (
            f"{prediction_count(evaluation.get('hmm_prediction')):,}"
            if evaluation
            else "—"
        ),
    )
    processing_time = evaluation.get("processing_time_seconds") if evaluation else None
    metric_columns[3].metric(
        "Processing time",
        (
            f"{processing_time:.3f} s"
            if isinstance(processing_time, (int, float))
            else "—"
        ),
    )

    if gps or classifier or hmm:
        st_folium(
            create_map(gps, classifier, hmm),
            height=560,
            use_container_width=True,
            returned_objects=[],
        )
        st.caption(
            "Map layers: 🔵 GPS observations · 🟠 classifier route · 🟢 HMM/Viterbi route. "
            "Use the layer control to toggle routes."
        )
    else:
        st.warning("The API returned no plottable GPS or route coordinates.")

    if result["evaluation_error"]:
        st.error(f"Evaluation unavailable: {result['evaluation_error']}")
    elif evaluation:
        st.success("Evaluation completed")
        if prediction_count(evaluation.get("hmm_prediction")) == 0:
            st.info(
                "The HMM decoder did not find a connected candidate sequence for "
                "this trip; the classifier route is still shown."
            )
        if metrics:
            st.subheader("Evaluation metrics")
            scalar_metrics = [
                (key.replace("_", " ").title(), value)
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            if scalar_metrics:
                metric_columns = st.columns(min(4, len(scalar_metrics)))
                for index, (label, value) in enumerate(scalar_metrics):
                    metric_columns[index % len(metric_columns)].metric(
                        label,
                        f"{value:.3f}" if isinstance(value, float) else f"{value:,}",
                    )
            else:
                st.json(metrics)

        with st.expander("Prediction details"):
            classifier_prediction = evaluation.get("classifier_prediction")
            hmm_prediction = evaluation.get("hmm_prediction")
            if classifier_prediction:
                st.markdown("**Classifier-only arc IDs**")
                st.json(classifier_prediction)
            if hmm_prediction:
                st.markdown("**HMM/Viterbi arc IDs**")
                st.json(hmm_prediction)
            if not classifier_prediction and not hmm_prediction:
                st.info("The API returned route geometry but no arc-ID details.")
