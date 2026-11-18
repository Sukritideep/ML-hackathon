import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.models.data import CANDIDATE_COLS, load_candidate_tables


def _feature_rows(trip_id, rows):
    return pd.DataFrame(
        [
            {
                "trip_id": int(trip_id),
                "obs_idx": obs_idx,
                "arc_id": arc_id,
                "distance_m": distance,
                "candidate_rank": rank,
                "projection_fraction": projection,
                "direction_difference_deg": direction,
                "gps_speed_mps": speed,
                "aligned_route_arc_id": 999,
                "target_label": 1,
            }
            for obs_idx, arc_id, distance, rank, projection, direction, speed in rows
        ]
    )


class TestCandidateDataAdapter(unittest.TestCase):
    def test_reads_only_split_ids_maps_features_and_sorts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            feature_dir = root / "features"
            feature_dir.mkdir()
            splits_path = root / "splits.json"
            splits_path.write_text(
                json.dumps(
                    {"train": ["00000001"], "student_test": ["00000002"]}
                ),
                encoding="utf-8",
            )
            _feature_rows(
                "00000001",
                [
                    (1, 13, 8.0, 2, 0.2, 30.0, 2.0),
                    (0, 12, 5.0, 2, 0.8, None, None),
                    (0, 11, 2.0, 1, 0.5, 15.0, 4.0),
                ],
            ).to_csv(feature_dir / "00000001.csv", index=False)
            _feature_rows(
                "00000002", [(0, 21, 4.0, 1, 0.3, 10.0, 3.0)]
            ).to_csv(feature_dir / "00000002.csv", index=False)
            (feature_dir / "unlisted.csv").write_text("not,a,feature,file\n")

            train, student, imputed = load_candidate_tables(
                splits_path, feature_dir
            )

        self.assertEqual(list(train.columns), list(CANDIDATE_COLS))
        self.assertEqual(train["trip_id"].unique().tolist(), ["00000001"])
        self.assertEqual(student["trip_id"].unique().tolist(), ["00000002"])
        self.assertEqual(
            train[["obs_idx", "arc_id"]].values.tolist(),
            [[0, 11], [0, 12], [1, 13]],
        )
        self.assertEqual(train["f_dist_gap"].tolist(), [0.0, 3.0, 0.0])
        self.assertEqual(train.loc[1, "f_bearing_diff"], 0.0)
        self.assertEqual(train.loc[1, "f_speed"], 0.0)
        self.assertEqual(imputed["train.direction_difference_deg"], 1)
        self.assertEqual(imputed["train.gps_speed_mps"], 1)
        self.assertNotIn("target_label", train.columns)
        self.assertNotIn("aligned_route_arc_id", train.columns)

    def test_rejects_split_files_with_extra_partitions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            splits_path = root / "splits.json"
            splits_path.write_text(
                json.dumps(
                    {
                        "train": ["00000001"],
                        "student_test": ["00000002"],
                        "evaluator_test": ["00000003"],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "only train and student_test"):
                load_candidate_tables(splits_path, root)


if __name__ == "__main__":
    unittest.main()
