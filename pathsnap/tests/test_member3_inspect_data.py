import json

from tools.member3_inspect_data import main


def test_inspects_only_split_trip_and_validates_route(tmp_path, capsys) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train-1.track").write_text(
        "0.0000 0.0000 0\n0.0010 0.0000 1\n", encoding="utf-8"
    )
    (data_dir / "train-1.route").write_text("10\n11\n", encoding="utf-8")
    (data_dir / "network.nodes").write_text(
        "1 0.0000 0.0000\n2 0.0010 0.0000\n3 0.0020 0.0000\n",
        encoding="utf-8",
    )
    (data_dir / "network.arcs").write_text(
        "10 1 2\n11 2 3\n", encoding="utf-8"
    )
    # This file is discoverable by extension, but must never be read because
    # its trip ID is absent from both permitted splits.
    (data_dir / "evaluator-only.track").write_text(
        "not valid trip data\n", encoding="utf-8"
    )
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(
        json.dumps({"train": ["train-1"], "student_test": []}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--data-dir",
            str(data_dir),
            "--splits",
            str(splits_path),
            "--trip-id",
            "train-1",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Arc IDs equal zero-based row indices: False" in output
    assert "Invalid route arc IDs (0): []" in output
    assert "Consecutive directed arc links: 1/1" in output
    assert "Projected route length:" in output
    assert "evaluator-only" not in output


def test_refuses_trip_not_in_allowed_splits(tmp_path, capsys) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(
        json.dumps({"train": [], "student_test": []}), encoding="utf-8"
    )

    exit_code = main(
        [
            "--data-dir",
            str(data_dir),
            "--splits",
            str(splits_path),
            "--trip-id",
            "evaluator-only",
        ]
    )

    assert exit_code == 2
    assert "Refusing to inspect 'evaluator-only'" in capsys.readouterr().err
