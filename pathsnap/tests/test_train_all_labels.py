import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.models.train_all import load_real


class TestRealLabelJoin(unittest.TestCase):
    def test_joins_separate_labels_by_trip_and_observation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidates_train = pd.DataFrame(
                {
                    "trip_id": ["00000001"] * 4,
                    "obs_idx": [0, 0, 1, 2],
                    "arc_id": [10, 11, 12, 13],
                    "f_dist": [1.0, 2.0, 3.0, 4.0],
                }
            )
            labels_train = pd.DataFrame(
                {
                    "trip_id": ["00000001"] * 3,
                    "obs_idx": [0, 1, 2],
                    "true_arc_id": [11, 99, None],
                }
            )
            candidates_student = pd.DataFrame(
                {
                    "trip_id": ["00000002", "00000002"],
                    "obs_idx": [0, 0],
                    "arc_id": [20, 21],
                    "f_dist": [1.0, 2.0],
                }
            )
            labels_student = pd.DataFrame(
                {
                    "trip_id": ["00000002"],
                    "obs_idx": [0],
                    "true_arc_id": [20],
                }
            )

            paths = {
                "candidates_train": root / "candidates_train.parquet",
                "candidates_student": root / "candidates_student.parquet",
                "labels_train": root / "labels_train.parquet",
                "labels_student": root / "labels_student.parquet",
            }
            candidates_train.to_parquet(paths["candidates_train"], index=False)
            candidates_student.to_parquet(paths["candidates_student"], index=False)
            labels_train.to_parquet(paths["labels_train"], index=False)
            labels_student.to_parquet(paths["labels_student"], index=False)

            train, student = load_real(
                paths["candidates_train"],
                paths["candidates_student"],
                paths["labels_train"],
                paths["labels_student"],
            )

        self.assertEqual(
            train[["obs_idx", "arc_id", "label"]].values.tolist(),
            [[0, 10, 0], [0, 11, 1], [1, 12, 0]],
        )
        self.assertEqual(
            student[["obs_idx", "arc_id", "label"]].values.tolist(),
            [[0, 20, 1], [0, 21, 0]],
        )
        self.assertEqual(train["true_arc_id"].tolist(), [11, 11, 99])


if __name__ == "__main__":
    unittest.main()
