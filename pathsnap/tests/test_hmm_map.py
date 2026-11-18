import json

import numpy as np
import pandas as pd
import pytest

from src.viz.make_map import generate_hmm_trip_map


def test_hmm_map_shows_ground_truth_and_viterbi_routes(tmp_path):
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "data"
    trip_id = "00000001"
    repo_root.mkdir()
    (repo_root / "splits.json").write_text(
        json.dumps({"train": [], "student_test": [trip_id]}),
        encoding="utf-8",
    )
    trip_dir = data_root / "student_test" / trip_id
    trip_dir.mkdir(parents=True)
    (trip_dir / f"{trip_id}.track").write_text(
        "-122.0000 37.0000 1\n-121.9995 37.0000 2\n",
        encoding="utf-8",
    )
    (trip_dir / f"{trip_id}.nodes").write_text(
        "-122.0000 37.0000\n-121.9995 37.0000\n-121.9990 37.0000\n",
        encoding="utf-8",
    )
    (trip_dir / f"{trip_id}.route").write_text("0\n1\n", encoding="utf-8")
    geometry_dir = (
        repo_root / "member1_preprocessing" / "outputs" / "road_geometry"
    )
    geometry_dir.mkdir(parents=True)
    np.savez(
        geometry_dir / f"{trip_id}.npz",
        arc_id=np.arange(2),
        from_node=np.array([0, 1]),
        to_node=np.array([1, 2]),
    )
    predictions = pd.DataFrame(
        {
            "trip_id": [trip_id, trip_id],
            "obs_idx": [0, 1],
            "true_arc_id": [0, 1],
            "classifier_pred_arc_id": [0, 0],
            "hmm_pred_arc_id": [0, 1],
            "long_gap": [False, False],
        }
    )

    output_path, stats = generate_hmm_trip_map(
        trip_id,
        predictions,
        repo_root=repo_root,
        data_root=data_root,
        output_dir=tmp_path / "maps",
    )
    html = output_path.read_text(encoding="utf-8")

    assert "Ground-truth reference route" in html
    assert "HMM-predicted route" in html
    assert stats["hmm_predicted_observations"] == 2
    with pytest.raises(FileExistsError):
        generate_hmm_trip_map(
            trip_id,
            predictions,
            repo_root=repo_root,
            data_root=data_root,
            output_dir=tmp_path / "maps",
        )
