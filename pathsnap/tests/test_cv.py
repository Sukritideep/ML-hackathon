import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.models.cv import run_experiment


class _CountingRng:
    def __init__(self):
        self.calls = 0

    def choice(self, values, size, replace=False):
        self.calls += 1
        return np.asarray(values)[:size]


class _ConstantProbabilityEstimator(BaseEstimator):
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.tile([0.5, 0.5], (len(X), 1))


class TestRunExperiment(unittest.TestCase):
    def test_subsamples_once_per_fold(self):
        df = pd.DataFrame(
            {
                "trip_id": [f"trip-{i}" for i in range(6)],
                "obs_idx": [0] * 6,
                "label": [0, 1, 0, 1, 0, 1],
                "f_value": range(6),
            }
        )
        rng = _CountingRng()

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("numpy.random.default_rng", return_value=rng):
                run_experiment(
                    "test",
                    "test",
                    _ConstantProbabilityEstimator(),
                    {},
                    df,
                    df,
                    ["f_value"],
                    n_splits=3,
                    max_train_rows=1,
                    out_csv=f"{temp_dir}\\experiments.csv",
                )

        self.assertEqual(rng.calls, 4)


if __name__ == "__main__":
    unittest.main()
