import math

import numpy as np
import pandas as pd
import pytest

from src.hmm.emissions import DistanceEmissionModel


@pytest.fixture
def model() -> DistanceEmissionModel:
    return DistanceEmissionModel(scale_m=2.0)


def _candidates(rows) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["trip_id", "obs_idx", "arc_id", "distance_m"]
    )


def test_closer_candidate_receives_higher_probability(model) -> None:
    candidates = _candidates(
        [("trip-a", 4, 701, 1.0), ("trip-a", 4, 904, 9.0)]
    )

    probabilities = model.predict_proba(candidates)

    assert probabilities[0] > probabilities[1]


def test_probabilities_are_normalized_per_observation(model) -> None:
    candidates = _candidates(
        [
            ("trip-a", 0, 10, 2.0),
            ("trip-a", 0, 11, 4.0),
            ("trip-a", 1, 12, 3.0),
            ("trip-b", 0, 13, 1.0),
            ("trip-b", 0, 14, 6.0),
        ]
    )

    scored = model.score_candidates(candidates)

    totals = scored.groupby(["trip_id", "obs_idx"])["emission_probability"].sum()
    assert np.allclose(totals.to_numpy(), 1.0)


def test_log_probabilities_match_normalized_distance_weights(model) -> None:
    candidates = _candidates(
        [("trip-a", 4, 701, 2.0), ("trip-a", 4, 904, 4.0)]
    )

    log_probabilities = model.predict_log_proba(candidates)
    normalizer = math.log1p(math.exp(-1.0))

    assert log_probabilities == pytest.approx([-normalizer, -1.0 - normalizer])
    assert np.exp(log_probabilities).sum() == pytest.approx(1.0)


def test_missing_and_invalid_distances_have_zero_probability(model) -> None:
    candidates = _candidates(
        [
            ("trip-a", 0, 1, 2.0),
            ("trip-a", 0, 2, None),
            ("trip-a", 0, 3, -1.0),
            ("trip-a", 0, 4, "not-a-distance"),
            ("trip-a", 0, 5, np.inf),
        ]
    )

    probabilities = model.predict_proba(candidates)
    log_probabilities = model.predict_log_proba(candidates)

    assert probabilities[0] == 1.0
    assert np.all(probabilities[1:] == 0.0)
    assert np.all(np.isneginf(log_probabilities[1:]))


def test_all_invalid_candidates_remain_zero_weight(model) -> None:
    candidates = _candidates(
        [("trip-a", 0, 1, None), ("trip-a", 0, 2, -3.0)]
    )

    assert np.all(model.predict_proba(candidates) == 0.0)
    assert np.all(np.isneginf(model.predict_log_proba(candidates)))


def test_empty_candidate_table_returns_empty_scores(model) -> None:
    candidates = _candidates([])
    schema_free_candidates = pd.DataFrame()

    assert model.predict_proba(candidates).size == 0
    assert model.predict_log_proba(candidates).size == 0
    scored = model.score_candidates(candidates)
    assert len(scored) == 0
    assert "emission_probability" in scored
    assert "emission_log_probability" in scored
    assert model.predict_proba(schema_free_candidates).size == 0


def test_extreme_distances_do_not_cause_nan_underflow() -> None:
    model = DistanceEmissionModel(scale_m=1e-300)
    candidates = _candidates(
        [("trip-a", 0, 1, 0.0), ("trip-a", 0, 2, 1e308)]
    )

    log_probabilities = model.predict_log_proba(candidates)
    probabilities = model.predict_proba(candidates)

    assert log_probabilities[0] == 0.0
    assert log_probabilities[1] == -math.inf
    assert probabilities.tolist() == [1.0, 0.0]
    assert not np.isnan(log_probabilities).any()


def test_scoring_preserves_original_observation_indices_and_arc_ids(model) -> None:
    candidates = _candidates(
        [("trip-a", 77, "arc-009", 4.0), ("trip-a", 77, "arc-100", 2.0)]
    )
    candidates.index = [9, 3]

    scored = model.score_candidates(candidates)

    assert scored.index.tolist() == [9, 3]
    assert scored["obs_idx"].tolist() == [77, 77]
    assert scored["arc_id"].tolist() == ["arc-009", "arc-100"]
    assert "f_dist" not in scored.columns


def test_candidates_are_normalized_separately_for_each_trip_and_observation(
    model,
) -> None:
    candidates = _candidates(
        [
            ("trip-a", 5, 1, 2.0),
            ("trip-b", 5, 2, 2.0),
            ("trip-b", 5, 3, 4.0),
        ]
    )

    probabilities = model.predict_proba(candidates)

    assert probabilities[0] == 1.0
    assert probabilities[1:].sum() == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [0.0, -1.0, math.inf, math.nan])
def test_scale_must_be_finite_and_positive(scale) -> None:
    with pytest.raises(ValueError, match="scale_m"):
        DistanceEmissionModel(scale_m=scale)
