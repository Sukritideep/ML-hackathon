"""Monotonic alignment of GPS observations to route segments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def point_to_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """Return the Euclidean distance from a point to a closed line segment."""
    point = np.array([px, py], dtype=float)
    start = np.array([ax, ay], dtype=float)
    vector = np.array([bx, by], dtype=float) - start
    squared_length = float(np.dot(vector, vector))

    if squared_length == 0.0:
        return float(np.linalg.norm(point - start))

    fraction = float(np.dot(point - start, vector) / squared_length)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    closest = start + fraction * vector
    return float(np.linalg.norm(point - closest))


def monotonic_route_alignment(
    gps_xy: np.ndarray,
    route_segments: np.ndarray,
    max_distance: float = 100.0,
) -> tuple[list[int | None], np.ndarray]:
    """Align GPS points to nondecreasing route-segment indices.

    Returns an assignment for each observation and its selected point-to-segment
    distance. Assignments farther than ``max_distance`` are ``None``; their
    measured distance is still returned. If there are no route segments, each
    assignment is ``None`` and its distance is ``inf``.
    """
    gps_xy = np.asarray(gps_xy, dtype=float)
    route_segments = np.asarray(route_segments, dtype=float)

    if gps_xy.ndim != 2 or gps_xy.shape[1:] != (2,):
        raise ValueError("gps_xy must have shape (T, 2)")
    if route_segments.ndim != 3 or route_segments.shape[1:] != (2, 2):
        raise ValueError("route_segments must have shape (M, 2, 2)")
    if not np.isfinite(gps_xy).all():
        raise ValueError("gps_xy must contain only finite coordinates")
    if not np.isfinite(route_segments).all():
        raise ValueError("route_segments must contain only finite coordinates")
    if max_distance < 0:
        raise ValueError("max_distance must be nonnegative")

    observation_count = gps_xy.shape[0]
    segment_count = route_segments.shape[0]
    if observation_count == 0:
        return [], np.empty(0, dtype=float)
    if segment_count == 0:
        return [None] * observation_count, np.full(observation_count, np.inf)

    starts = route_segments[:, 0, :]
    vectors = route_segments[:, 1, :] - starts
    squared_lengths = np.einsum("ij,ij->i", vectors, vectors)
    offsets = gps_xy[:, None, :] - starts[None, :, :]
    numerators = np.einsum("tmi,mi->tm", offsets, vectors)

    fractions = np.zeros_like(numerators)
    np.divide(
        numerators,
        squared_lengths[None, :],
        out=fractions,
        where=squared_lengths[None, :] != 0.0,
    )
    np.clip(fractions, 0.0, 1.0, out=fractions)
    closest = starts[None, :, :] + fractions[:, :, None] * vectors[None, :, :]
    distances = np.linalg.norm(gps_xy[:, None, :] - closest, axis=2)

    # Each row stores the best path cost ending at that segment. The prefix
    # minimum enforces j' <= j without evaluating all segment pairs.
    costs = distances[0].copy()
    backpointers = np.full((observation_count, segment_count), -1, dtype=np.intp)
    for obs_idx in range(1, observation_count):
        previous_costs = costs
        prefix_costs = np.empty(segment_count, dtype=float)
        prefix_indices = np.empty(segment_count, dtype=np.intp)
        best_cost = np.inf
        best_index = 0

        for segment_idx in range(segment_count):
            candidate_cost = previous_costs[segment_idx]
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_index = segment_idx
            prefix_costs[segment_idx] = best_cost
            prefix_indices[segment_idx] = best_index

        costs = distances[obs_idx] + prefix_costs
        backpointers[obs_idx] = prefix_indices

    selected = np.empty(observation_count, dtype=np.intp)
    selected[-1] = int(np.argmin(costs))
    for obs_idx in range(observation_count - 1, 0, -1):
        selected[obs_idx - 1] = backpointers[obs_idx, selected[obs_idx]]

    selected_distances = distances[np.arange(observation_count), selected]
    assignments: list[int | None] = [
        int(segment_idx) if distance <= max_distance else None
        for segment_idx, distance in zip(selected, selected_distances)
    ]
    return assignments, selected_distances


def _trip_value(trip: Any, name: str) -> Any:
    if isinstance(trip, dict):
        if name in trip:
            return trip[name]
    elif hasattr(trip, name):
        return getattr(trip, name)
    raise ValueError(
        f"trip must provide {name!r}; projected trip geometry is not yet "
        "defined by the core interface"
    )


def assign_labels(trip: Any, graph: Any) -> pd.DataFrame:
    """Return ``trip_id, obs_idx, true_arc_id`` labels for a trip.

    The core loader and graph interfaces are not implemented yet, so this
    adapter deliberately does not call graph methods or project raw lon/lat.
    Until those interfaces are available, ``trip`` must provide:

    - ``trip_id``
    - ``gps_xy_m``: GPS coordinates as an array of shape (T, 2), in metres
    - ``route_segments_xy_m``: route geometries of shape (M, 2, 2), in metres
    - ``route_arc_ids``: the arc ID corresponding to each route segment

    ``graph`` is retained for compatibility with CONTRACTS.md and intentionally
    unused until its representation is defined.
    """
    del graph
    trip_id = _trip_value(trip, "trip_id")
    gps_xy = np.asarray(_trip_value(trip, "gps_xy_m"), dtype=float)
    route_segments = np.asarray(
        _trip_value(trip, "route_segments_xy_m"), dtype=float
    )
    route_arc_ids = list(_trip_value(trip, "route_arc_ids"))

    if route_segments.ndim != 3 or route_segments.shape[1:] != (2, 2):
        raise ValueError("trip.route_segments_xy_m must have shape (M, 2, 2)")
    if len(route_arc_ids) != route_segments.shape[0]:
        raise ValueError(
            "trip.route_arc_ids must contain one arc ID per route segment"
        )

    assignments, _ = monotonic_route_alignment(gps_xy, route_segments)
    true_arc_ids = [
        route_arc_ids[route_idx] if route_idx is not None else None
        for route_idx in assignments
    ]
    return pd.DataFrame(
        {
            "trip_id": [trip_id] * len(assignments),
            "obs_idx": np.arange(len(assignments), dtype=np.intp),
            "true_arc_id": pd.Series(true_arc_ids, dtype=object),
        },
        columns=["trip_id", "obs_idx", "true_arc_id"],
    )