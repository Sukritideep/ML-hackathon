import numpy as np
import pandas as pd

from src.labels.assign_labels import (
    assign_labels,
    monotonic_route_alignment,
    point_to_segment_distance,
)


def _four_segment_route() -> np.ndarray:
    return np.array(
        [
            [[0.0, 0.0], [100.0, 0.0]],
            [[100.0, 0.0], [200.0, 0.0]],
            [[200.0, 0.0], [300.0, 0.0]],
            [[300.0, 0.0], [400.0, 0.0]],
        ]
    )


def test_alignment_tracks_connected_route_with_small_noise() -> None:
    rng = np.random.default_rng(42)
    x = np.concatenate(
        [np.linspace(start + 10, start + 90, 5) for start in (0, 100, 200, 300)]
    )
    gps_xy = np.column_stack((x, np.zeros_like(x)))
    gps_xy += rng.normal(loc=0.0, scale=1.0, size=gps_xy.shape)

    assignments, distances = monotonic_route_alignment(
        gps_xy, _four_segment_route()
    )

    expected = np.repeat(np.arange(4), 5)
    assert assignments == expected.tolist()
    assert np.all(np.diff(np.asarray(assignments)) >= 0)
    assert np.all(distances < 5.0)


def test_observation_farther_than_limit_is_unassigned() -> None:
    assignments, distances = monotonic_route_alignment(
        np.array([[50.0, 101.0]]),
        np.array([[[0.0, 0.0], [100.0, 0.0]]]),
    )

    assert assignments == [None]
    assert distances.tolist() == [101.0]


def test_zero_length_segment_uses_distance_to_endpoint() -> None:
    assert point_to_segment_distance(3.0, 4.0, 0.0, 0.0, 0.0, 0.0) == 5.0

    assignments, distances = monotonic_route_alignment(
        np.array([[3.0, 4.0]]),
        np.array([[[0.0, 0.0], [0.0, 0.0]]]),
    )
    assert assignments == [0]
    assert distances.tolist() == [5.0]


def test_empty_gps_observations_return_empty_results() -> None:
    assignments, distances = monotonic_route_alignment(
        np.empty((0, 2)), _four_segment_route()
    )

    assert assignments == []
    assert distances.shape == (0,)


def test_empty_route_returns_unassigned_observations() -> None:
    assignments, distances = monotonic_route_alignment(
        np.array([[10.0, 0.0], [20.0, 1.0]]),
        np.empty((0, 2, 2)),
    )

    assert assignments == [None, None]
    assert np.isinf(distances).all()


def test_assign_labels_maps_segment_indices_to_arc_ids() -> None:
    trip = {
        "trip_id": "trip-1",
        "gps_xy_m": np.array([[25.0, 0.0], [50.0, 150.0]]),
        "route_segments_xy_m": np.array([[[0.0, 0.0], [100.0, 0.0]]]),
        "route_arc_ids": ["arc-a"],
    }

    labels = assign_labels(trip, graph=object())

    assert list(labels.columns) == ["trip_id", "obs_idx", "true_arc_id"]
    assert labels["trip_id"].tolist() == ["trip-1", "trip-1"]
    assert labels["obs_idx"].tolist() == [0, 1]
    assert labels.loc[0, "true_arc_id"] == "arc-a"
    assert pd.isna(labels.loc[1, "true_arc_id"])
