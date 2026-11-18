"""Directed road-network distances and HMM transition likelihoods."""

from __future__ import annotations

import heapq
import itertools
import math
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.hmm.metrics import ArcConnectivityAdapter, DEFAULT_HOP_LIMIT


@dataclass(frozen=True)
class CandidatePosition:
    """Position along a directed candidate arc, measured from its start node."""

    arc_id: Hashable
    progress_m: float


def candidate_position_from_row(
    candidate: Mapping[str, Any] | None,
) -> CandidatePosition | None:
    """Adapt a candidate/feature CSV row to an arc position.

    Feature rows provide ``candidate_progress_m`` directly. Candidate rows
    instead provide ``projection_fraction`` and ``arc_length_m``.
    """
    if candidate is None:
        return None
    arc_id = candidate.get("arc_id")
    if arc_id is None:
        return None

    try:
        if candidate.get("candidate_progress_m") is not None:
            progress = float(candidate["candidate_progress_m"])
        else:
            fraction = float(candidate["projection_fraction"])
            length = float(candidate["arc_length_m"])
            progress = fraction * length
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(progress):
        return None
    return CandidatePosition(_python_scalar(arc_id), progress)


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


class DirectedRoadGraph:
    """Weighted directed graph built from Person 1's road-geometry arrays.

    Original arc IDs and node IDs are retained as graph keys. No reverse arcs
    are inferred. ``arc_connectivity`` is an ``ArcConnectivityAdapter`` over
    the same original arc IDs, allowing ``route_continuity`` to reuse its
    existing connectivity logic.
    """

    def __init__(
        self,
        arc_ids: Sequence[Hashable] | np.ndarray,
        from_nodes: Sequence[Hashable] | np.ndarray,
        to_nodes: Sequence[Hashable] | np.ndarray,
        length_m: Sequence[float] | np.ndarray,
        hop_limit: int = DEFAULT_HOP_LIMIT,
    ) -> None:
        arrays = [
            np.asarray(values)
            for values in (arc_ids, from_nodes, to_nodes, length_m)
        ]
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("arc IDs, endpoints, and lengths must be 1D arrays")
        if len({len(array) for array in arrays}) != 1:
            raise ValueError("road geometry arrays must have matching lengths")

        ids, starts, ends, lengths = arrays
        self.arc_to_endpoints: dict[Hashable, tuple[Hashable, Hashable]] = {}
        self.arc_lengths_m: dict[Hashable, float] = {}
        self._outgoing: dict[Hashable, list[tuple[Hashable, float]]] = defaultdict(
            list
        )
        self._node_distance_cache: dict[tuple[Hashable, Hashable], float] = {}

        for arc_id, start_node, end_node, raw_length in zip(
            ids, starts, ends, lengths
        ):
            arc_id = _python_scalar(arc_id)
            start_node = _python_scalar(start_node)
            end_node = _python_scalar(end_node)
            length = float(raw_length)
            if not math.isfinite(length) or length < 0:
                raise ValueError(
                    f"Arc {arc_id!r} has invalid length {raw_length!r}"
                )
            try:
                hash(arc_id)
                hash(start_node)
                hash(end_node)
            except TypeError as exc:
                raise ValueError("arc and node IDs must be hashable") from exc
            if arc_id in self.arc_to_endpoints:
                raise ValueError(f"Duplicate arc ID {arc_id!r}")

            self.arc_to_endpoints[arc_id] = (start_node, end_node)
            self.arc_lengths_m[arc_id] = length
            self._outgoing[start_node].append((end_node, length))

        self.arc_connectivity = ArcConnectivityAdapter(
            self.arc_to_endpoints, hop_limit=hop_limit
        )

    @classmethod
    def from_npz(
        cls, path: str | Path, hop_limit: int = DEFAULT_HOP_LIMIT
    ) -> "DirectedRoadGraph":
        """Load the original directed arc table from a geometry NPZ."""
        required = {"arc_id", "from_node", "to_node", "length_m"}
        try:
            with np.load(path, allow_pickle=False) as data:
                missing = required - set(data.files)
                if missing:
                    raise ValueError(
                        f"Geometry NPZ is missing arrays: {sorted(missing)}"
                    )
                arrays = {key: data[key].copy() for key in required}
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not load directed road graph from {path}: {exc}") from exc
        return cls(
            arrays["arc_id"],
            arrays["from_node"],
            arrays["to_node"],
            arrays["length_m"],
            hop_limit=hop_limit,
        )

    def _arc(self, arc_id: Hashable) -> tuple[Hashable, Hashable, float] | None:
        try:
            endpoints = self.arc_to_endpoints.get(arc_id)
            if endpoints is None:
                return None
            return (
                endpoints[0],
                endpoints[1],
                self.arc_lengths_m[arc_id],
            )
        except TypeError:
            return None

    def shortest_node_distance(
        self, start_node: Hashable, end_node: Hashable
    ) -> float:
        """Return directed Dijkstra distance between nodes, or infinity."""
        try:
            cache_key = (start_node, end_node)
            cached = self._node_distance_cache.get(cache_key)
        except TypeError:
            return math.inf
        if cached is not None:
            return cached
        if start_node == end_node:
            self._node_distance_cache[cache_key] = 0.0
            return 0.0

        distances = {start_node: 0.0}
        sequence = itertools.count()
        queue: list[tuple[float, int, Hashable]] = [
            (0.0, next(sequence), start_node)
        ]
        while queue:
            distance, _, node = heapq.heappop(queue)
            if distance != distances.get(node):
                continue
            if node == end_node:
                self._node_distance_cache[cache_key] = distance
                return distance
            for neighbor, edge_length in self._outgoing.get(node, ()):
                next_distance = distance + edge_length
                if next_distance < distances.get(neighbor, math.inf):
                    distances[neighbor] = next_distance
                    heapq.heappush(
                        queue, (next_distance, next(sequence), neighbor)
                    )

        self._node_distance_cache[cache_key] = math.inf
        return math.inf

    def distance_between_positions(
        self,
        start: CandidatePosition | Mapping[str, Any] | None,
        end: CandidatePosition | Mapping[str, Any] | None,
    ) -> float:
        """Return shortest directed distance between two candidate positions.

        The distance includes the remaining length of the start arc, complete
        directed intermediate arcs, and progress along the end arc. A forward
        move along one arc is handled directly; a backwards move on the same
        arc is possible only if the graph contains a directed cycle back to it.
        Invalid or missing candidates return infinity.
        """
        start_position = (
            candidate_position_from_row(start)
            if isinstance(start, Mapping)
            else start
        )
        end_position = (
            candidate_position_from_row(end) if isinstance(end, Mapping) else end
        )
        if start_position is None or end_position is None:
            return math.inf
        if not isinstance(start_position, CandidatePosition) or not isinstance(
            end_position, CandidatePosition
        ):
            return math.inf

        start_arc = self._arc(start_position.arc_id)
        end_arc = self._arc(end_position.arc_id)
        if start_arc is None or end_arc is None:
            return math.inf
        _, start_end_node, start_length = start_arc
        end_start_node, _, end_length = end_arc
        try:
            start_progress = float(start_position.progress_m)
            end_progress = float(end_position.progress_m)
        except (TypeError, ValueError, OverflowError):
            return math.inf
        if (
            not math.isfinite(start_progress)
            or not math.isfinite(end_progress)
            or start_progress < 0
            or end_progress < 0
            or start_progress > start_length
            or end_progress > end_length
        ):
            return math.inf

        if start_position.arc_id == end_position.arc_id:
            if end_progress >= start_progress:
                return end_progress - start_progress

        middle_distance = self.shortest_node_distance(
            start_end_node, end_start_node
        )
        if math.isinf(middle_distance):
            return math.inf
        return (
            (start_length - start_progress) + middle_distance + end_progress
        )


def transition_log_probability(
    graph: DirectedRoadGraph,
    previous_candidate: CandidatePosition | Mapping[str, Any] | None,
    current_candidate: CandidatePosition | Mapping[str, Any] | None,
    observed_distance_m: float,
    scale_m: float,
) -> float:
    """Return a Laplace transition log-density for observed vs road distance.

    ``scale_m`` is the configurable Laplace scale in metres. The log-density is
    ``-|observed - network| / scale_m - log(2 * scale_m)``. Missing/invalid
    candidates and unreachable transitions have log-probability ``-inf``.
    Invalid scale or observed-distance parameters raise ``ValueError``.
    """
    if not math.isfinite(scale_m) or scale_m <= 0:
        raise ValueError("scale_m must be finite and positive")
    if previous_candidate is None or current_candidate is None:
        return -math.inf
    if not math.isfinite(observed_distance_m) or observed_distance_m < 0:
        raise ValueError("observed_distance_m must be finite and nonnegative")

    network_distance = graph.distance_between_positions(
        previous_candidate, current_candidate
    )
    if not math.isfinite(network_distance):
        return -math.inf
    return (
        -abs(observed_distance_m - network_distance) / scale_m
        - math.log(2.0 * scale_m)
    )
