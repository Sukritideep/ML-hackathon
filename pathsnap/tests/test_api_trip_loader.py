import json

import numpy as np
import pytest

from src.core.io import load_trip


def _write_trip(data_root, partition, trip_id):
    trip_dir = data_root / partition / trip_id
    trip_dir.mkdir(parents=True)
    np.savetxt(trip_dir / f"{trip_id}.track", [[12.5, 48.1, 1], [12.6, 48.2, 2]])
    np.savetxt(trip_dir / f"{trip_id}.nodes", [[12.5, 48.1], [12.6, 48.2]])
    np.savetxt(trip_dir / f"{trip_id}.arcs", [[0, 1]], fmt="%d")
    np.savetxt(trip_dir / f"{trip_id}.route", [[0]], fmt="%d")


def test_load_trip_uses_split_partition_and_hides_route_by_default(
    tmp_path, monkeypatch
):
    split_file = tmp_path / "splits.json"
    split_file.write_text(
        json.dumps({"train": ["00000001"], "student_test": ["00000000"]}),
        encoding="utf-8",
    )
    data_root = tmp_path / "data"
    _write_trip(data_root, "student_test", "00000000")
    monkeypatch.setattr("src.core.io.SPLITS_FILE", split_file)
    monkeypatch.setenv("PATHSNAP_DATA_DIR", str(data_root))

    trip = load_trip("00000000")

    assert trip.gps.tolist() == [[12.5, 48.1, 1.0], [12.6, 48.2, 2.0]]
    assert trip.nodes.shape == (2, 2)
    assert trip.arcs.tolist() == [[0, 1]]
    assert trip.route is None


def test_load_trip_rejects_ids_outside_split(tmp_path, monkeypatch):
    split_file = tmp_path / "splits.json"
    split_file.write_text(
        json.dumps({"train": ["00000001"], "student_test": ["00000000"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.core.io.SPLITS_FILE", split_file)

    with pytest.raises(ValueError, match="not present in the split"):
        load_trip("not-a-trip")


def test_load_trip_with_route_is_explicit_opt_in(tmp_path, monkeypatch):
    split_file = tmp_path / "splits.json"
    split_file.write_text(
        json.dumps({"train": ["00000001"], "student_test": ["00000000"]}),
        encoding="utf-8",
    )
    data_root = tmp_path / "data"
    _write_trip(data_root, "train", "00000001")
    monkeypatch.setattr("src.core.io.SPLITS_FILE", split_file)
    monkeypatch.setenv("PATHSNAP_DATA_DIR", str(data_root))

    trip = load_trip("00000001", with_route=True)

    assert trip.route.tolist() == [0]
