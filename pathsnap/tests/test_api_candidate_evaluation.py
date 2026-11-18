import json

import numpy as np
import pandas as pd

from src.api import candidate_data, pipeline


class _Bundle:
    def predict_proba(self, candidates):
        return np.full(len(candidates), 0.9)


def test_evaluate_trip_uses_unlabeled_features_and_returns_routes(
    tmp_path, monkeypatch
):
    trip_id = "00000000"
    monkeypatch.setattr(pipeline, "SPLITS_FILE", tmp_path / "splits.json")
    pipeline.SPLITS_FILE.write_text(
        json.dumps({"train": ["00000001"], "student_test": [trip_id]}),
        encoding="utf-8",
    )

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    pd.DataFrame(
        {
            "trip_id": [0, 0],
            "obs_idx": [0, 1],
            "arc_id": [10, 10],
            "distance_m": [4.0, 5.0],
            "candidate_rank": [1, 1],
            "projection_fraction": [0.1, 0.2],
            "direction_difference_deg": [0.0, 0.0],
            "gps_speed_mps": [0.0, 10.0],
            "gps_distance_m": [np.nan, 10.0],
            "long_gap": [False, False],
            "candidate_progress_m": [10.0, 20.0],
            "projected_x": [500000.0, 500010.0],
            "projected_y": [5500000.0, 5500000.0],
        }
    ).to_csv(feature_dir / f"{trip_id}.csv", index=False)

    graph_dir = tmp_path / "road_geometry"
    graph_dir.mkdir()
    np.savez(
        graph_dir / f"{trip_id}.npz",
        arc_id=np.array([10]),
        from_node=np.array([1]),
        to_node=np.array([2]),
        length_m=np.array([100.0]),
    )
    audit_path = tmp_path / "crs.csv"
    pd.DataFrame({"trip_id": [0], "suggested_epsg": [32632]}).to_csv(
        audit_path, index=False
    )

    monkeypatch.setattr(candidate_data, "features_dir", lambda: feature_dir)
    monkeypatch.setattr(candidate_data, "road_geometry_dir", lambda: graph_dir)
    monkeypatch.setattr(candidate_data, "crs_audit_path", lambda: audit_path)
    monkeypatch.setattr(pipeline, "model_available", lambda: True)
    monkeypatch.setattr(pipeline, "load_model_bundle", lambda: _Bundle())
    monkeypatch.setattr(pipeline.io, "data_root", lambda: tmp_path / "missing-data")

    result = pipeline.evaluate_trip(trip_id)

    assert result["trip_id"] == trip_id
    assert result["original_gps"] == []
    assert result["classifier_prediction"] == [[0, 10], [1, 10]]
    assert result["hmm_prediction"] == [[0, 10], [1, 10]]
    assert len(result["classifier_route"]) == 2
    assert len(result["hmm_route"]) == 2
    assert result["metrics"] is None
