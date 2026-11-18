import unittest

import pandas as pd

from src.models.stub_bundle import DistanceStubBundle


class TestDistanceStubBundle(unittest.TestCase):
    def test_probabilities_are_softmaxed_per_observation(self):
        candidates = pd.DataFrame(
            {
                "trip_id": ["a", "a", "b", "a"],
                "obs_idx": [0, 0, 0, 1],
                "f_dist": [0.0, 10.0, 5.0, 0.0],
            }
        )

        probabilities = DistanceStubBundle().predict_proba(candidates)

        self.assertEqual(probabilities.shape, (4,))
        self.assertGreater(probabilities[0], probabilities[1])
        self.assertAlmostEqual(probabilities[0] + probabilities[1], 1.0)
        self.assertAlmostEqual(probabilities[2], 1.0)
        self.assertAlmostEqual(probabilities[3], 1.0)


if __name__ == "__main__":
    unittest.main()
