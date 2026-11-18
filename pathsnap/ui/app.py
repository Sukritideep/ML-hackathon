import os
from typing import Any

import folium
import requests
import streamlit as st
from streamlit_folium import folium_static


DEFAULT_API_URL = os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_SECONDS = 10


def api_get(base_url: str, path: str) -> tuple[bool, Any]:
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}{path}",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return True, response.json()
    except requests.Timeout:
        return False, f"Request timed out after {REQUEST_TIMEOUT_SECONDS} seconds."
    except requests.ConnectionError:
        return False, "Cannot connect to the PathSnap API. Check the server URL and network."
    except requests.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except (AttributeError, ValueError):
            detail = exc.response.text
        return False, detail
    except requests.RequestException as exc:
        return False, str(exc)


def api_post(base_url: str, path: str) -> tuple[bool, Any]:
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}{path}",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return True, response.json()
    except requests.Timeout:
        return False, f"Request timed out after {REQUEST_TIMEOUT_SECONDS} seconds."
    except requests.ConnectionError:
        return False, "Cannot connect to the PathSnap API. Check the server URL and network."
    except requests.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except (AttributeError, ValueError):
            detail = exc.response.text
        return False, detail
    except requests.RequestException as exc:
        return False, str(exc)


def gps_locations(gps: list[list[Any]]) -> list[list[float]]:
    locations = []
    for point in gps:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        lon, lat = point[0], point[1]
        if isinstance(lon, (int, float)) and isinstance(lat, (int, float)):
            locations.append([float(lat), float(lon)])
    return locations


def create_map(locations: list[list[float]]) -> folium.Map:
    if locations:
        center_lat = sum(point[0] for point in locations) / len(locations)
        center_lon = sum(point[1] for point in locations) / len(locations)
    else:
        center_lat, center_lon = 20.5937, 78.9629

    route_map = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    if locations:
        folium.PolyLine(locations, color="#1f77b4", weight=3, opacity=0.8).add_to(route_map)
        for location in locations:
            folium.CircleMarker(
                location=location,
                radius=3,
                color="#1f77b4",
                fill=True,
                fill_opacity=0.9,
            ).add_to(route_map)
    return route_map


st.set_page_config(page_title="PathSnap Evaluator", page_icon=":world_map:", layout="wide")
st.title("PathSnap")
st.caption("GPS map-matching evaluator")

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
        st.success(status.get("status", "running"))
    else:
        st.error(status)

    if health_ok:
        if health.get("status") == "ready":
            st.success("Pipeline ready")
        else:
            st.warning("Pipeline not fully available")
    else:
        st.error(health)

trips_ok, trips_response = api_get(api_url, "/trips")
if not trips_ok:
    st.error(trips_response)
    st.stop()

trip_ids = trips_response.get("trips", [])
if not trip_ids:
    st.info("No permitted evaluation trips are currently available.")
    st.stop()

selected_trip = st.selectbox("Evaluation trip", trip_ids)

if st.button("Run Evaluation", type="primary"):
    with st.spinner("Loading trip and running evaluation..."):
        trip_ok, trip = api_get(api_url, f"/trips/{selected_trip}")
        if not trip_ok:
            st.error(trip)
        else:
            locations = gps_locations(trip.get("gps", []))
            route_map = create_map(locations)
            folium_static(route_map, width=1100, height=550)

        evaluate_ok, evaluation = api_post(api_url, f"/evaluate/{selected_trip}")
        if evaluate_ok:
            st.success("Evaluation completed")
            if evaluation.get("processing_time_seconds") is not None:
                st.metric(
                    "Processing time",
                    f"{evaluation['processing_time_seconds']:.3f} s",
                )
            if evaluation.get("classifier_prediction"):
                st.subheader("Classifier-only prediction")
                st.json(evaluation["classifier_prediction"])
            if evaluation.get("hmm_prediction"):
                st.subheader("HMM/Viterbi reconstruction")
                st.json(evaluation["hmm_prediction"])
            if evaluation.get("metrics"):
                st.subheader("Evaluation metrics")
                st.json(evaluation["metrics"])
        else:
            st.error(evaluation)