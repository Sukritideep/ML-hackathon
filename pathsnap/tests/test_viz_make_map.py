import json

import pandas as pd
import pytest

from src.viz.make_map import generate_trip_map


def test_generate_trip_map_uses_split_labels_and_preserves_no_overwrite(tmp_path):
    repo_root = tmp_path / "repo"
    data_root = tmp_path / "data"
    trip_id = "00000001"
    (repo_root / "pathsnap" / "results").mkdir(parents=True)
    candidate_dir = (
        repo_root / "member1_preprocessing" / "outputs" / "candidates"
    )
    candidate_dir.mkdir(parents=True)
    trip_dir = data_root / "train" / trip_id
    trip_dir.mkdir(parents=True)
    (repo_root / "splits.json").write_text(
        json.dumps({"train": [trip_id], "student_test": []}),
        encoding="utf-8",
    )
    (trip_dir / f"{trip_id}.track").write_text(
        "-122.0000 37.0000 1\n"
        "-121.9995 37.0000 2\n"
        "-121.9990 37.0000 3\n",
        encoding="utf-8",
    )
    (trip_dir / f"{trip_id}.nodes").write_text(
        "-122.0000 37.0000\n"
        "-121.9995 37.0000\n"
        "-121.9990 37.0000\n",
        encoding="utf-8",
    )
    (trip_dir / f"{trip_id}.arcs").write_text("0 1\n1 2\n", encoding="utf-8")
    (trip_dir / f"{trip_id}.route").write_text("0\n1\n", encoding="utf-8")
    pd.DataFrame(
        {
            "trip_id": [trip_id] * 3,
            "obs_idx": [0, 1, 2],
            "true_arc_id": pd.Series([0, 1, None], dtype="Int64"),
        }
    ).to_parquet(
        repo_root / "pathsnap" / "results" / "labels_train.parquet",
        index=False,
    )
    pd.DataFrame(
        {"obs_idx": [0, 1], "arc_id": [0, 0]}
    ).to_csv(candidate_dir / f"{trip_id}.csv", index=False)

    output_dir = tmp_path / "maps"
    output_path, stats = generate_trip_map(
        trip_id,
        repo_root=repo_root,
        data_root=data_root,
        output_dir=output_dir,
    )
    html = output_path.read_text(encoding="utf-8")

    assert output_path.is_file()
    assert "GPS trajectory" in html
    assert "Reference route" in html
    assert "Monotonic-DP labeled arcs" in html
    assert stats["partition"] == "train"
    assert stats["gps_observations"] == 3
    assert stats["valid_dp_labels"] == 2
    assert stats["candidate_misses"] == 1
    assert stats["null_labels"] == 1
    with pytest.raises(FileExistsError):
        generate_trip_map(
            trip_id,
            repo_root=repo_root,
            data_root=data_root,
            output_dir=output_dir,
        )
