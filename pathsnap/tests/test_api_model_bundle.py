import numpy as np
import pandas as pd
import pytest

from src.api.model_bundle import ClassifierBundle
from src.models.data import FEATURE_COLS


class _FakeEstimator:
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        assert list(features.columns) == list(FEATURE_COLS)
        return np.array([[0.75, 0.25], [0.1, 0.9]])


def test_bundle_returns_positive_class_probabilities():
    candidates = pd.DataFrame({column: [0.0, 1.0] for column in FEATURE_COLS})

    probabilities = ClassifierBundle(_FakeEstimator()).predict_proba(candidates)

    np.testing.assert_allclose(probabilities, [0.25, 0.9])


def test_bundle_rejects_missing_model_features():
    candidates = pd.DataFrame({"f_dist": [1.0]})

    with pytest.raises(ValueError, match="missing model features"):
        ClassifierBundle(_FakeEstimator()).predict_proba(candidates)
