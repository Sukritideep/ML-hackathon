import math

import numpy as np
import pytest

from src.hmm.metrics import route_continuity
from src.hmm.transitions import (
    CandidatePosition,
    DirectedRoadGraph,
    candidate_position_from_row,
    transition_log_probability,
)


@pytest.fixture
def graph() -> DirectedRoadGraph:
    return DirectedRoadGraph(
        arc_ids=[10, 20, 30, 40, 41, 50],
        from_nodes=[1, 2, 3, 4, 5, 8],
        to_nodes=[2, 3, 4, 5, 6, 9],
        length_m=[10.0, 20.0, 30.0, 0.0, 3.0, 5.0],
    )


def test_same_arc_forward_movement(graph) -> None:
    distance = graph.distance_between_positions(
        CandidatePosition(10, 2.0), CandidatePosition(10, 8.0)
    )

    assert distance == 6.0


def test_same_arc_forward_movement_skips_graph_search(graph, monkeypatch) -> None:
    def unexpected_search(*args, **kwargs):
        raise AssertionError("same-arc forward distance must not run Dijkstra")

    monkeypatch.setattr(graph, "shortest_node_distance", unexpected_search)

    assert (
        graph.distance_between_positions(
            CandidatePosition(10, 2.0), CandidatePosition(10, 8.0)
        )
        == 6.0
    )


def test_directly_connected_arcs_include_partial_start_and_end(graph) -> None:
    distance = graph.distance_between_positions(
        CandidatePosition(10, 3.0), CandidatePosition(20, 4.0)
    )

    assert distance == 11.0


def test_multi_arc_path_includes_complete_intermediate_arc(graph) -> None:
    distance = graph.distance_between_positions(
        CandidatePosition(10, 3.0), CandidatePosition(30, 5.0)
    )

    assert distance == 32.0


def test_shortest_node_distances_batches_reachable_and_disconnected_targets(
    graph,
) -> None:
    distances = graph.shortest_node_distances(1, [1, 2, 3, 8])

    assert distances == {1: 0.0, 2: 10.0, 3: 30.0, 8: math.inf}


def test_reverse_direction_without_directed_path_is_unreachable(graph) -> None:
    distance = graph.distance_between_positions(
        CandidatePosition(20, 10.0), CandidatePosition(10, 5.0)
    )

    assert math.isinf(distance)


def test_disconnected_arcs_are_unreachable(graph) -> None:
    distance = graph.distance_between_positions(
        CandidatePosition(10, 2.0), CandidatePosition(50, 1.0)
    )

    assert math.isinf(distance)


def test_same_arc_backward_requires_a_directed_cycle(graph) -> None:
    distance = graph.distance_between_positions(
        CandidatePosition(10, 8.0), CandidatePosition(10, 2.0)
    )

    assert math.isinf(distance)


def test_zero_length_arc_can_be_traversed(graph) -> None:
    assert (
        graph.distance_between_positions(
            CandidatePosition(40, 0.0), CandidatePosition(41, 2.0)
        )
        == 2.0
    )
    assert (
        graph.distance_between_positions(
            CandidatePosition(40, 0.0), CandidatePosition(40, 0.0)
        )
        == 0.0
    )


def test_invalid_arc_and_out_of_range_progress_are_unreachable(graph) -> None:
    assert math.isinf(
        graph.distance_between_positions(
            CandidatePosition(999, 0.0), CandidatePosition(10, 1.0)
        )
    )
    assert math.isinf(
        graph.distance_between_positions(
            CandidatePosition(10, 11.0), CandidatePosition(20, 1.0)
        )
    )


def test_empty_graph_and_missing_candidates_are_safe() -> None:
    empty = DirectedRoadGraph([], [], [], [])
    assert math.isinf(
        empty.distance_between_positions(
            CandidatePosition(1, 0.0), CandidatePosition(2, 0.0)
        )
    )
    assert math.isinf(empty.distance_between_positions(None, CandidatePosition(2, 0.0)))


def test_dijkstra_supports_mixed_node_id_types() -> None:
    mixed = DirectedRoadGraph(
        arc_ids=[1, 2, 3, 4],
        from_nodes=["start", "start", 1, "1"],
        to_nodes=[1, "1", "goal", "goal"],
        length_m=[1.0, 1.0, 1.0, 1.0],
    )

    assert mixed.shortest_node_distance("start", "goal") == 2.0


def test_candidate_row_adapter_supports_both_actual_schemas() -> None:
    candidate_row = {
        "arc_id": np.int64(10),
        "projection_fraction": 0.25,
        "arc_length_m": 40.0,
    }
    feature_row = {
        "arc_id": 10,
        "candidate_progress_m": 12.5,
        "projection_fraction": 0.25,
        "arc_length_m": 40.0,
    }

    assert candidate_position_from_row(candidate_row) == CandidatePosition(10, 10.0)
    assert candidate_position_from_row(feature_row) == CandidatePosition(10, 12.5)
    assert candidate_position_from_row(None) is None


def test_transition_log_probability_uses_configured_laplace_scale(graph) -> None:
    result = transition_log_probability(
        graph,
        CandidatePosition(10, 3.0),
        CandidatePosition(20, 4.0),
        observed_distance_m=12.0,
        scale_m=5.0,
    )

    assert result == pytest.approx(-1.0 / 5.0 - math.log(10.0))


def test_unreachable_or_missing_transition_has_negative_infinity(graph) -> None:
    assert (
        transition_log_probability(
            graph,
            CandidatePosition(10, 0.0),
            CandidatePosition(50, 0.0),
            observed_distance_m=10.0,
            scale_m=5.0,
        )
        == -math.inf
    )
    assert (
        transition_log_probability(graph, None, CandidatePosition(10, 1.0), 10.0, 5.0)
        == -math.inf
    )


@pytest.mark.parametrize("scale", [0.0, -1.0, math.inf, math.nan])
def test_transition_scale_must_be_finite_and_positive(graph, scale) -> None:
    with pytest.raises(ValueError, match="scale_m"):
        transition_log_probability(
            graph,
            CandidatePosition(10, 1.0),
            CandidatePosition(10, 2.0),
            observed_distance_m=1.0,
            scale_m=scale,
        )


def test_from_npz_reads_original_arc_and_node_ids(tmp_path) -> None:
    path = tmp_path / "geometry.npz"
    np.savez(
        path,
        arc_id=np.array([104, 209], dtype=np.int64),
        from_node=np.array([23, 45], dtype=np.int64),
        to_node=np.array([45, 67], dtype=np.int64),
        length_m=np.array([8.0, 12.0], dtype=np.float64),
    )

    loaded = DirectedRoadGraph.from_npz(path)

    assert loaded.arc_to_endpoints == {104: (23, 45), 209: (45, 67)}
    assert loaded.arc_lengths_m == {104: 8.0, 209: 12.0}
    assert route_continuity([104, 209], loaded.arc_connectivity) == 1.0
