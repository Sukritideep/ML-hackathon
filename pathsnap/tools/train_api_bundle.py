"""Fit the selected Member 2 classifier using training-trip labels only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.api.model_bundle import ClassifierBundle
from src.models.data import FEATURE_COLS, _load_trip_features

SEED = 42
DEFAULT_FEATURE_DIR = Path("..") / "member1_preprocessing" / "outputs" / "features"


def load_training_data(
    split_file: Path,
    feature_dir: Path,
    labels_path: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    splits = json.loads(split_file.read_text(encoding="utf-8"))
    train_ids = splits.get("train")
    if not isinstance(train_ids, list) or not train_ids:
        raise ValueError("Split file must contain a non-empty train list")

    imputed_counts: dict[str, int] = {}
    candidates = pd.concat(
        [
            _load_trip_features(
                trip_id,
                feature_dir,
                "train",
                imputed_counts,
            )
            for trip_id in train_ids
        ],
        ignore_index=True,
    )
    labels = pd.read_parquet(labels_path)
    required = {"trip_id", "obs_idx", "true_arc_id"}
    if not required.issubset(labels.columns):
        raise ValueError(f"Training labels must contain columns {sorted(required)}")
    if labels.duplicated(["trip_id", "obs_idx"]).any():
        raise ValueError("Training labels contain duplicate trip/observation keys")

    labels = labels.loc[:, ["trip_id", "obs_idx", "true_arc_id"]].copy()
    labels["true_arc_id"] = pd.to_numeric(labels["true_arc_id"], errors="raise")
    labels = labels.loc[labels["true_arc_id"].notna()]
    if labels.empty:
        raise ValueError("Training labels contain no aligned observations")

    frame = candidates.merge(
        labels,
        on=["trip_id", "obs_idx"],
        how="inner",
        validate="many_to_one",
    )
    frame["label"] = frame["arc_id"].eq(frame["true_arc_id"]).astype("int8")
    eligible = frame.groupby(["trip_id", "obs_idx"])["label"].transform("max").eq(1)
    frame = frame.loc[eligible].sort_values(
        ["trip_id", "obs_idx", "arc_id"],
        kind="mergesort",
    )
    if frame.empty or frame["label"].nunique() != 2:
        raise ValueError("Training candidates must contain both binary label classes")
    if not np.isfinite(frame.loc[:, FEATURE_COLS].to_numpy(dtype=float)).all():
        raise ValueError("Training features contain non-finite values")

    return frame.loc[:, FEATURE_COLS], frame["label"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, default=Path("splits.json"))
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("results/labels_train.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/classifier_bundle.joblib"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        parser.error(
            f"Model artifact already exists: {args.output}; use --overwrite to replace it"
        )

    features, labels = load_training_data(
        args.split_file,
        args.feature_dir,
        args.labels,
    )
    classifier = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=SEED,
        verbose=-1,
    )
    classifier.fit(features, labels)

    bundle = ClassifierBundle(classifier)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output)
    print(
        f"Saved classifier bundle to {args.output} "
        f"(training rows={len(features):,}, features={len(FEATURE_COLS)})"
    )


if __name__ == "__main__":
    main()
