"""Evaluation metrics for predicted road-arc sequences."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Hashable, Mapping, Sequence
from typing import Any


DEFAULT_HOP_LIMIT = 3


class ArcConnectivityAdapter:
    """Directed connectivity adapter backed by ``arc_id -> (start, end)``.

    The core graph interface is not implemented yet. This adapter interprets
    each mapping value as a directed road arc and connects an arc to every arc
    whose start node equals its end node.
    """

    def __init__(
        self,
        arc_to_endpoints: Mapping[Hashable, Sequence[Hashable]],
        hop_limit: int = DEFAULT_HOP_LIMIT,
    ) -> None:
        if not isinstance(arc_to_endpoints, Mapping):
            raise TypeError("arc_to_endpoints must be a mapping")
        if isinstance(hop_limit, bool) or not isinstance(hop_limit, int):
            raise TypeError("hop_limit must be an integer")
        if hop_limit < 0:
            raise ValueError("hop_limit must be nonnegative")

        self.hop_limit = hop_limit
        self.arc_to_endpoints: dict[Hashable, tuple[Hashable, Hashable]] = {}
        self._outgoing: dict[Hashable, list[Hashable]] = defaultdict(list)

        for arc_id, endpoints in arc_to_endpoints.items():
            if not isinstance(endpoints, Sequence) or isinstance(
                endpoints, (str, bytes)
            ) or len(endpoints) != 2:
                raise ValueError(
                    f"Arc {arc_id!r} must map to exactly (start_node, end_node)"
                )
            start_node, end_node = endpoints
            try:
                hash(arc_id)
                hash(start_node)
                hash(end_node)
            except TypeError as exc:
                raise ValueError(
                    f"Arc {arc_id!r} and its endpoint IDs must be hashable"
                ) from exc
            self.arc_to_endpoints[arc_id] = (start_node, end_node)
            self._outgoing[start_node].append(arc_id)

    def contains(self, arc_id: Any) -> bool:
        try:
            return arc_id in self.arc_to_endpoints
        except TypeError:
            return False

    def reachable(
        self, first_arc: Hashable, second_arc: Hashable, hop_limit: int | None = None
    ) -> bool:
        """Return whether ``second_arc`` follows ``first_arc`` within hops.

        A direct successor is one hop. The first and second arcs themselves are
        included in no additional intermediate-hop count.
        """
        limit = self.hop_limit if hop_limit is None else hop_limit
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("hop_limit must be an integer")
        if limit < 0:
            raise ValueError("hop_limit must be nonnegative")
        if not self.contains(first_arc) or not self.contains(second_arc):
            return False
        if first_arc == second_arc:
            return True
        if limit == 0:
            return False

        queue = deque([(first_arc, 0)])
        visited = {first_arc}
        while queue:
            current_arc, hops = queue.popleft()
            if hops >= limit:
                continue
            _, end_node = self.arc_to_endpoints[current_arc]
            for next_arc in self._outgoing.get(end_node, ()):
                next_hops = hops + 1
                if next_arc == second_arc:
                    return True
                if next_arc not in visited:
                    visited.add(next_arc)
                    queue.append((next_arc, next_hops))
        return False


def route_continuity(
    pred_arcs: Sequence[Hashable],
    graph: ArcConnectivityAdapter | Mapping[Hashable, Sequence[Hashable]],
    hop_limit: int | None = None,
) -> float:
    """Return the fraction of consecutive predicted-arc pairs that are valid.

    Consecutive duplicates of a known arc are valid. Otherwise, a pair is valid
    when the second arc is reachable from the first using directed connectivity
    in at most ``hop_limit`` arc-to-arc steps. Invalid arc IDs make every pair
    that contains them invalid. Empty and single-arc sequences return 1.0
    because they contain no discontinuous pairs.

    ``graph`` must be an :class:`ArcConnectivityAdapter` or an
    ``arc_id -> (start_node, end_node)`` mapping. Since the repository's graph
    interface is not implemented, the mapping adapter uses a provisional
    default of ``DEFAULT_HOP_LIMIT`` (3); callers can pass the agreed value
    explicitly or construct the adapter with it.
    """
    if isinstance(graph, ArcConnectivityAdapter):
        connectivity = graph
    else:
        connectivity = ArcConnectivityAdapter(
            graph,
            hop_limit=DEFAULT_HOP_LIMIT if hop_limit is None else hop_limit,
        )
    effective_hop_limit = (
        connectivity.hop_limit if hop_limit is None else hop_limit
    )
    if isinstance(effective_hop_limit, bool) or not isinstance(
        effective_hop_limit, int
    ):
        raise TypeError("hop_limit must be an integer")
    if effective_hop_limit < 0:
        raise ValueError("hop_limit must be nonnegative")

    arcs = list(pred_arcs)
    if len(arcs) < 2:
        return 1.0
    valid_pairs = sum(
        connectivity.reachable(first, second, effective_hop_limit)
        for first, second in zip(arcs, arcs[1:])
    )
    return valid_pairs / (len(arcs) - 1)


def _edit_distance(
    true_arcs: Sequence[Hashable], predicted_arcs: Sequence[Hashable]
) -> int:
    """Compute Levenshtein distance using O(min(n, m)) memory."""
    if len(true_arcs) < len(predicted_arcs):
        true_arcs, predicted_arcs = predicted_arcs, true_arcs

    previous = list(range(len(predicted_arcs) + 1))
    for true_idx, true_arc in enumerate(true_arcs, start=1):
        current = [true_idx]
        for pred_idx, pred_arc in enumerate(predicted_arcs, start=1):
            current.append(
                min(
                    previous[pred_idx] + 1,
                    current[pred_idx - 1] + 1,
                    previous[pred_idx - 1] + (true_arc != pred_arc),
                )
            )
        previous = current
    return previous[-1]


def sequence_metrics(
    true_arcs: Sequence[Hashable], predicted_arcs: Sequence[Hashable]
) -> dict[str, float | int]:
    """Return set precision/recall, aligned correct run, and edit distance.

    Set precision/recall ignore repeated occurrences. If both sets are empty,
    both scores are 1.0; if only the denominator set is empty, the score is 0.0.
    The longest correct consecutive run counts equal arc IDs at the same
    chronological positions in the two sequences.
    """
    true_sequence = list(true_arcs)
    predicted_sequence = list(predicted_arcs)
    true_set = set(true_sequence)
    predicted_set = set(predicted_sequence)
    common_count = len(true_set & predicted_set)

    precision = (
        common_count / len(predicted_set) if predicted_set else float(not true_set)
    )
    recall = common_count / len(true_set) if true_set else float(not predicted_set)

    longest_run = 0
    current_run = 0
    for true_arc, predicted_arc in zip(true_sequence, predicted_sequence):
        if true_arc == predicted_arc:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    return {
        "arc_set_precision": float(precision),
        "arc_set_recall": float(recall),
        "longest_correct_run": longest_run,
        "edit_distance": _edit_distance(true_sequence, predicted_sequence),
    }