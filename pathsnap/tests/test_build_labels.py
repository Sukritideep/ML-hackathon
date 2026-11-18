import json

import numpy as np
import pandas as pd
import pytest

from tools.build_labels import (
    LABEL_COLUMNS,
    LabelBuildError,
    _read_splits,
    _validate_labels,
    align_route_observations,
    candidate_hit_miss_counts,
)


def test_dp_alignment_preserves_original_arc_ids_and_nulls_far_observations() -> None:
    gps_xy = np.array([[50.0, 0.0], [150.0, 0.0], [150.0, 101.0]])
    route_arc_ids = [49003, 60483]
    route_segments = np.array(
        [
            [[0.0, 0.0], [100.0, 0.0]],
            [[100.0, 0.0], [200.0, 0.0]],
        ]
    )

    labels, route_indices, distances = align_route_observations(
        gps_xy, route_arc_ids, route_segments
    )

    assert labels == [49003, 60483, None]
    assert route_indices.tolist() == [0, 1, 1]
    assert np.all(np.diff(route_indices) >= 0)
    assert distances.tolist() == [0.0, 0.0, 101.0]


def test_empty_route_keeps_every_observation_as_null() -> None:
    labels, route_indices, distances = align_route_observations(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        [],
        np.empty((0, 2, 2)),
    )

    assert labels == [None, None]
    assert route_indices.tolist() == [-1, -1]
    assert np.isinf(distances).all()


def test_route_geometry_count_must_match_route_ids() -> None:
    with pytest.raises(LabelBuildError, match="one 2-endpoint segment"):
        align_route_observations(
            np.array([[0.0, 0.0]]),
            [12],
            np.empty((0, 2, 2)),
        )


def test_candidate_hits_and_misses_use_only_valid_dp_labels() -> None:
    hits, misses = candidate_hit_miss_counts(
        [49003, None, 60483],
        {0: {49003, 99}, 1: {777}, 2: {88}},
    )

    assert hits == 1
    assert misses == 1


def test_root_splits_are_disjoint_and_string_normalized(tmp_path) -> None:
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps({"train": ["00000001"], "student_test": ["00000000"]}),
        encoding="utf-8",
    )

    assert _read_splits(path) == {
        "train": ["00000001"],
        "student_test": ["00000000"],
    }


def test_label_schema_split_isolation_and_duplicate_validation() -> None:
    labels = pd.DataFrame(
        {
            "trip_id": pd.Series(["00000001", "00000001"], dtype="string"),
            "obs_idx": [0, 1],
            "true_arc_id": pd.array([4, None], dtype="Int64"),
        },
        columns=LABEL_COLUMNS,
    )

    _validate_labels(labels, ["00000001"], "train")
    with pytest.raises(LabelBuildError, match="duplicate"):
        _validate_labels(
            pd.concat([labels, labels.iloc[[0]]], ignore_index=True),
            ["00000001"],
            "train",
        )
    with pytest.raises(LabelBuildError, match="IDs differ"):
        _validate_labels(labels, ["00000002"], "train")
