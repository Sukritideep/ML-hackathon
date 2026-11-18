import json

import numpy as np
import pandas as pd

from src.hmm.evaluation import (
    build_evaluation_report,
    evaluate_trip_frames,
    save_evaluation_report,
)
from src.hmm.transitions import DirectedRoadGraph


def _trip_rows(trip_id: str, arc_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = []
    features = []
    for obs_idx in range(3):
        candidates.append(
            {"trip_id": trip_id, "obs_idx": obs_idx, "arc_id": arc_id}
        )
        features.append(
            {
                "trip_id": trip_id,
                "obs_idx": obs_idx,
                "arc_id": arc_id,
                "distance_m": 2.0,
                "candidate_progress_m": float(obs_idx),
                "gps_distance_m": np.nan if obs_idx == 0 else 1.0,
                "long_gap": False,
            }
        )
    return pd.DataFrame(candidates), pd.DataFrame(features)


def _graph() -> DirectedRoadGraph:
    return DirectedRoadGraph(
        arc_ids=[100, 101],
        from_nodes=[1, 2],
        to_nodes=[2, 3],
        length_m=[10.0, 10.0],
    )


def test_split_evaluation_is_separate_and_writes_report(tmp_path) -> None:
    (tmp_path / "splits.json").write_text(
        json.dumps({"train": ["00000001"], "student_test": ["00000002"]}),
        encoding="utf-8",
    )
    output_root = tmp_path / "member1_preprocessing" / "outputs"
    labels_root = tmp_path / "pathsnap" / "results"
    for directory in (
        output_root / "candidates",
        output_root / "features",
        output_root / "road_geometry",
        labels_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    label_tables = {}
    for trip_id, arc_id, split_name in (
        ("00000001", 100, "train"),
        ("00000002", 101, "student"),
    ):
        candidates, features = _trip_rows(trip_id, arc_id)
        candidates.to_csv(output_root / "candidates" / f"{trip_id}.csv", index=False)
        features.to_csv(output_root / "features" / f"{trip_id}.csv", index=False)
        np.savez(
            output_root / "road_geometry" / f"{trip_id}.npz",
            arc_id=np.array([100, 101], dtype=np.int64),
            from_node=np.array([1, 2], dtype=np.int64),
            to_node=np.array([2, 3], dtype=np.int64),
            length_m=np.array([10.0, 10.0], dtype=np.float64),
        )
        label_tables.setdefault(split_name, []).extend(
            [
                {"trip_id": trip_id, "obs_idx": idx, "true_arc_id": arc_id}
                for idx in range(3)
            ]
        )
    pd.DataFrame(label_tables["train"]).to_parquet(
        labels_root / "labels_train.parquet", index=False
    )
    pd.DataFrame(label_tables["student"]).to_parquet(
        labels_root / "labels_student.parquet", index=False
    )

    report = build_evaluation_report(tmp_path)

    assert set(report["splits"]) == {"train", "student_test"}
    assert list(report["splits"]["train"]["per_trip"]) == ["00000001"]
    assert list(report["splits"]["student_test"]["per_trip"]) == ["00000002"]
    assert report["configuration"]["student_test_labels_used_for_training"] is False
    assert report["splits"]["train"]["overall"]["sequence_accuracy"] == 1.0
    assert report["splits"]["student_test"]["overall"]["route_continuity"] == 1.0
    assert report["splits"]["train"]["per_trip"]["00000001"]["edit_distance"] == 0

    report_path = labels_root / "hmm_evaluation_report.json"
    save_evaluation_report(report, report_path)
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 1
    assert saved["splits"]["student_test"]["processed_trip_count"] == 1


def test_missing_candidate_observation_is_reported_as_null_and_miss() -> None:
    candidates, features = _trip_rows("00000001", 100)
    candidates = candidates[candidates.obs_idx.ne(1)].reset_index(drop=True)
    features = features[features.obs_idx.ne(1)].reset_index(drop=True)
    labels = pd.DataFrame(
        {
            "trip_id": ["00000001"] * 3,
            "obs_idx": [0, 1, 2],
            "true_arc_id": pd.Series([100, 100, 100], dtype="Int64"),
        }
    )

    metrics, aligned = evaluate_trip_frames(
        "00000001", candidates, features, labels, _graph()
    )

    assert metrics["missing_candidate_observations"] == 1
    assert metrics["candidate_misses"] == 1
    assert metrics["missing_predictions"] == 1
    assert metrics["null_predictions"] == 1
    assert metrics["sequence_accuracy"] == 2 / 3
    assert aligned.loc[aligned.obs_idx.eq(1), "pred_arc_id"].isna().all()
