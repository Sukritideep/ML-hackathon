import numpy as np
import pytest

from tools.build_labels import LabelBuildError, build_trip_labels


@pytest.fixture
def projected_trip():
    return {
        "trip_id": "train-1",
        "projection_crs": "EPSG:32631",
        "gps_xy_m": np.array([[5.0, 5.0], [15.0, 5.0]]),
        "node_xy_m": {
            "n1": (0.0, 0.0),
            "n2": (10.0, 0.0),
            "n3": (20.0, 0.0),
        },
        "arc_endpoints": {
            "road-a": ("n1", "n2"),
            "road-b": ("n2", "n3"),
        },
        "route_arc_ids": ["road-a", "road-b"],
    }


def test_build_trip_labels_preserves_original_arc_ids(projected_trip) -> None:
    labels = build_trip_labels(
        "train-1", lambda _: projected_trip, expected_crs="EPSG:32631"
    )

    assert list(labels.columns) == ["trip_id", "obs_idx", "true_arc_id"]
    assert labels["obs_idx"].tolist() == [0, 1]
    assert labels["true_arc_id"].tolist() == ["road-a", "road-b"]


def test_build_trip_labels_rejects_unknown_route_arc(projected_trip) -> None:
    projected_trip["route_arc_ids"] = ["missing"]

    with pytest.raises(LabelBuildError, match="unknown original arc ID"):
        build_trip_labels(
            "train-1", lambda _: projected_trip, expected_crs="EPSG:32631"
        )


def test_build_trip_labels_rejects_disconnected_route(projected_trip) -> None:
    projected_trip["arc_endpoints"]["road-b"] = ("n3", "n2")

    with pytest.raises(LabelBuildError, match="not directionally connected"):
        build_trip_labels(
            "train-1", lambda _: projected_trip, expected_crs="EPSG:32631"
        )


def test_build_trip_labels_requires_matching_explicit_crs(projected_trip) -> None:
    with pytest.raises(LabelBuildError, match="does not match"):
        build_trip_labels(
            "train-1", lambda _: projected_trip, expected_crs="EPSG:32610"
        )
