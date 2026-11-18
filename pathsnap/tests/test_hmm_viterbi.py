import math

import numpy as np
import pandas as pd
import pytest

from src.hmm.transitions import DirectedRoadGraph
from src.hmm.viterbi import ViterbiParams, decode


@pytest.fixture
def graph() -> DirectedRoadGraph:
    return DirectedRoadGraph(
        arc_ids=[10, 11, 20, 21, 30, 40, 41],
        from_nodes=[1, 2, 1, 3, 4, 8, 9],
        to_nodes=[2, 4, 3, 4, 5, 9, 10],
        length_m=[10.0, 10.0, 20.0, 20.0, 10.0, 10.0, 10.0],
    )


def _row(
    trip_id,
    obs_idx,
    arc_id,
    gps_distance_m,
    long_gap=False,
    candidate_progress_m=0.0,
):
    return {
        "trip_id": trip_id,
        "obs_idx": obs_idx,
        "arc_id": arc_id,
        "candidate_progress_m": candidate_progress_m,
        "gps_distance_m": gps_distance_m,
        "long_gap": long_gap,
    }


def test_viterbi_finds_global_path_and_backtracks_in_time_order(graph) -> None:
    candidates = pd.DataFrame(
        [
            _row("trip-a", 2, 30, 10.0),
            _row("trip-a", 0, 10, np.nan),
            _row("trip-a", 0, 20, np.nan),
            _row("trip-a", 1, 11, 10.0),
            _row("trip-a", 1, 21, 10.0),
        ]
    )
    probabilities = [1.0, 0.9, 0.1, 0.1, 0.9]

    decoded = decode(
        candidates,
        probabilities,
        graph,
        ViterbiParams(transition_scale_m=1.0),
    )

    assert decoded["obs_idx"].tolist() == [0, 1, 2]
    assert decoded["pred_arc_id"].tolist() == [10, 11, 30]


def test_transition_scores_can_override_local_emission_winners(graph) -> None:
    candidates = pd.DataFrame(
        [
            _row("trip-a", 0, 10, np.nan),
            _row("trip-a", 0, 20, np.nan),
            _row("trip-a", 1, 11, 10.0),
            _row("trip-a", 1, 21, 10.0),
        ]
    )
    probabilities = [0.2, 0.8, 0.2, 0.8]

    tight_transition = decode(
        candidates,
        probabilities,
        graph,
        {"transition_scale_m": 1.0},
    )
    loose_transition = decode(
        candidates,
        probabilities,
        graph,
        {"transition_scale_m": 100.0},
    )

    assert tight_transition["pred_arc_id"].tolist() == [10, 11]
    assert loose_transition["pred_arc_id"].tolist() == [20, 21]


def test_long_gap_starts_an_independent_segment_by_default(graph) -> None:
    candidates = pd.DataFrame(
        [
            _row("trip-a", 0, 10, np.nan),
            _row("trip-a", 0, 20, np.nan),
            _row("trip-a", 1, 11, 10.0, long_gap=True),
            _row("trip-a", 1, 21, 10.0, long_gap=True),
        ]
    )

    decoded = decode(candidates, [0.2, 0.8, 0.2, 0.8], graph, {
        "transition_scale_m": 1.0
    })

    assert decoded["pred_arc_id"].tolist() == [20, 21]


def test_long_gap_segmentation_can_be_disabled(graph) -> None:
    candidates = pd.DataFrame(
        [
            _row("trip-a", 0, 10, np.nan),
            _row("trip-a", 0, 20, np.nan),
            _row("trip-a", 1, 11, 10.0, long_gap=True),
            _row("trip-a", 1, 21, 10.0, long_gap=True),
        ]
    )

    decoded = decode(
        candidates,
        [0.2, 0.8, 0.2, 0.8],
        graph,
        ViterbiParams(transition_scale_m=1.0, break_on_long_gap=False),
    )

    assert decoded["pred_arc_id"].tolist() == [10, 11]


def test_different_candidate_counts_and_single_candidate_states(graph) -> None:
    candidates = pd.DataFrame(
        [
            _row("trip-a", 0, 10, np.nan),
            _row("trip-a", 1, 11, 10.0),
            _row("trip-a", 1, 21, 10.0),
        ]
    )

    decoded = decode(
        candidates,
        [1.0, 0.1, 0.9],
        graph,
        ViterbiParams(transition_scale_m=1.0),
    )

    assert decoded["pred_arc_id"].tolist() == [10, 11]


def test_unreachable_transition_restarts_decoding_without_fabricating_a_path(
    graph,
) -> None:
    candidates = pd.DataFrame(
        [
            _row("trip-a", 0, 10, np.nan),
            _row("trip-a", 1, 40, 10.0),
        ]
    )

    decoded = decode(
        candidates,
        [1.0, 1.0],
        graph,
        ViterbiParams(transition_scale_m=5.0),
    )

    assert decoded["pred_arc_id"].tolist() == [10, 40]


def test_viterbi_resumes_with_a_valid_path_after_unreachable_transition(graph) -> None:
    candidates = pd.DataFrame(
        [
            _row("trip-a", 0, 10, np.nan),
            _row("trip-a", 1, 40, 10.0),
            _row("trip-a", 2, 41, 10.0),
        ]
    )

    decoded = decode(
        candidates,
        [1.0, 1.0, 1.0],
        graph,
        ViterbiParams(transition_scale_m=5.0),
    )

    assert decoded["pred_arc_id"].tolist() == [10, 40, 41]


def test_invalid_arc_and_zero_probability_candidates_are_excluded(graph) -> None:
    candidates = pd.DataFrame(
        [
            _row("trip-a", 0, 999, np.nan),
            _row("trip-a", 0, 10, np.nan),
            _row("trip-a", 1, 11, 10.0),
        ]
    )

    decoded = decode(
        candidates,
        [1.0, 0.0, 1.0],
        graph,
        ViterbiParams(transition_scale_m=5.0),
    )

    assert decoded["pred_arc_id"].tolist() == [None, 11]


def test_empty_input_and_single_observation_are_supported(graph) -> None:
    empty = decode(
        pd.DataFrame(),
        [],
        graph,
        ViterbiParams(transition_scale_m=5.0),
    )
    one = decode(
        pd.DataFrame([_row("trip-a", 42, 10, np.nan)]),
        [1.0],
        graph,
        ViterbiParams(transition_scale_m=5.0),
    )

    assert empty.empty
    assert empty.columns.tolist() == ["obs_idx", "pred_arc_id"]
    assert one["obs_idx"].tolist() == [42]
    assert one["pred_arc_id"].tolist() == [10]


def test_multiple_trips_are_decoded_independently(graph) -> None:
    candidates = pd.DataFrame(
        [
            _row("trip-a", 0, 10, np.nan),
            _row("trip-b", 0, 40, np.nan),
            _row("trip-a", 1, 11, 10.0),
        ]
    )

    decoded = decode(
        candidates,
        [1.0, 1.0, 1.0],
        graph,
        ViterbiParams(transition_scale_m=5.0),
    )

    assert decoded[["trip_id", "obs_idx"]].to_records(index=False).tolist() == [
        ("trip-a", 0),
        ("trip-a", 1),
        ("trip-b", 0),
    ]
    assert decoded["pred_arc_id"].tolist() == [10, 11, 40]


def test_nonconsecutive_observation_indices_start_new_segments(graph) -> None:
    candidates = pd.DataFrame(
        [
            _row("trip-a", 2, 10, np.nan),
            _row("trip-a", 5, 40, np.nan),
        ]
    )

    decoded = decode(
        candidates,
        [1.0, 1.0],
        graph,
        ViterbiParams(transition_scale_m=5.0),
    )

    assert decoded["obs_idx"].tolist() == [2, 5]
    assert decoded["pred_arc_id"].tolist() == [10, 40]


def test_duplicate_candidate_rows_do_not_change_output_shape(graph) -> None:
    candidates = pd.DataFrame(
        [
            _row("trip-a", 0, 10, np.nan),
            _row("trip-a", 0, 10, np.nan),
            _row("trip-a", 1, 11, 10.0),
        ]
    )

    decoded = decode(
        candidates,
        [0.3, 0.7, 1.0],
        graph,
        ViterbiParams(transition_scale_m=5.0),
    )

    assert len(decoded) == 2
    assert decoded["pred_arc_id"].tolist() == [10, 11]


def test_invalid_probability_inputs_are_rejected(graph) -> None:
    candidates = pd.DataFrame([_row("trip-a", 0, 10, np.nan)])
    params = ViterbiParams(transition_scale_m=5.0)

    with pytest.raises(ValueError, match="one value"):
        decode(candidates, [], graph, params)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        decode(candidates, [math.nan], graph, params)
