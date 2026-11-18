import pytest

from src.hmm.metrics import (
    ArcConnectivityAdapter,
    route_continuity,
    sequence_metrics,
)


@pytest.fixture
def arc_graph():
    return {
        "a": ("n1", "n2"),
        "b": ("n2", "n3"),
        "c": ("n3", "n4"),
        "d": ("n8", "n9"),
    }


def test_route_continuity_same_arc(arc_graph) -> None:
    assert route_continuity(["a", "a"], arc_graph) == 1.0


def test_route_continuity_direct_successor(arc_graph) -> None:
    assert route_continuity(["a", "b"], arc_graph) == 1.0


def test_route_continuity_multi_hop(arc_graph) -> None:
    assert route_continuity(["a", "c"], arc_graph, hop_limit=2) == 1.0
    assert route_continuity(["a", "c"], arc_graph, hop_limit=1) == 0.0


def test_route_continuity_disconnected_pair(arc_graph) -> None:
    assert route_continuity(["a", "d"], arc_graph) == 0.0


def test_route_continuity_handles_empty_single_and_invalid_ids(arc_graph) -> None:
    assert route_continuity([], arc_graph) == 1.0
    assert route_continuity(["unknown"], arc_graph) == 1.0
    assert route_continuity(["a", "unknown"], arc_graph) == 0.0


def test_adapter_can_be_reused_with_its_configured_hop_limit(arc_graph) -> None:
    graph = ArcConnectivityAdapter(arc_graph, hop_limit=2)
    assert route_continuity(["a", "c"], graph) == 1.0


def test_sequence_metrics() -> None:
    metrics = sequence_metrics(
        ["a", "b", "c", "d"],
        ["a", "b", "x", "d", "e"],
    )

    assert metrics == {
        "arc_set_precision": 0.6,
        "arc_set_recall": 0.75,
        "longest_correct_run": 2,
        "edit_distance": 2,
    }


def test_sequence_metrics_empty_inputs() -> None:
    assert sequence_metrics([], []) == {
        "arc_set_precision": 1.0,
        "arc_set_recall": 1.0,
        "longest_correct_run": 0,
        "edit_distance": 0,
    }


def test_invalid_hop_limit_is_rejected(arc_graph) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        route_continuity(["a", "b"], arc_graph, hop_limit=-1)
