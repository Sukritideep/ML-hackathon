import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.bundle import LightGBMBundle
from src.models.tune import _validate_training_frame


class _ProbabilityModel:
    def predict_proba(self, features):
        values = features["f_dist"].to_numpy(dtype=float)
        positive = 1.0 / (1.0 + np.exp(values))
        return np.column_stack((1.0 - positive, positive))


class TestLightGBMBundle(unittest.TestCase):
    def test_probabilities_are_normalized_per_observation(self):
        candidates = pd.DataFrame(
            {
                "trip_id": ["a", "a", "a"],
                "obs_idx": [0, 0, 1],
                "f_dist": [0.0, 1.0, 2.0],
                "f_rank": [0, 1, 0],
                "f_dist_gap": [0.0, 1.0, 0.0],
                "f_proj": [0.2, 0.4, 0.5],
                "f_bearing_diff": [1.0, 2.0, 3.0],
                "f_speed": [4.0, 5.0, 6.0],
            }
        )

        probabilities = LightGBMBundle(_ProbabilityModel()).predict_proba(candidates)

        self.assertEqual(probabilities.shape, (3,))
        self.assertAlmostEqual(probabilities[:2].sum(), 1.0)
        self.assertAlmostEqual(probabilities[2], 1.0)
        self.assertGreater(probabilities[0], probabilities[1])

    def test_rejects_missing_features(self):
        candidates = pd.DataFrame({"trip_id": ["a"], "obs_idx": [0]})

        with self.assertRaisesRegex(ValueError, "missing required columns"):
            LightGBMBundle(_ProbabilityModel()).predict_proba(candidates)

    def test_bundle_can_be_joblib_round_tripped(self):
        bundle = LightGBMBundle(_ProbabilityModel())

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bundle.joblib"
            joblib.dump(bundle, path)
            loaded = joblib.load(path)

        self.assertIsInstance(loaded, LightGBMBundle)


class TestTrainingFrameValidation(unittest.TestCase):
    def test_requires_enough_distinct_trip_groups(self):
        frame = pd.DataFrame(
            {
                "trip_id": ["a", "a", "b", "b"],
                "obs_idx": [0, 0, 0, 0],
                "label": [0, 1, 0, 1],
                "feature": [0.0, 1.0, 2.0, 3.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "available trips"):
            _validate_training_frame(frame, ["feature"], n_splits=3)


if __name__ == "__main__":
    unittest.main()
