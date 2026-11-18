"""Inference bundle for the PathSnap LightGBM candidate model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data import FEATURE_COLS


@dataclass
class LightGBMBundle:
    """Expose normalized per-observation candidate probabilities."""

    model: Any
    feature_cols: tuple[str, ...] = tuple(FEATURE_COLS)

    def predict_proba(self, cands_df: pd.DataFrame) -> np.ndarray:
        required = ("trip_id", "obs_idx", *self.feature_cols)
        missing = [column for column in required if column not in cands_df.columns]
        if missing:
            raise ValueError(f"Candidate table is missing required columns: {missing}")
        if cands_df[["trip_id", "obs_idx"]].isna().any().any():
            raise ValueError("Candidate trip_id and obs_idx values cannot be null")

        features = cands_df.loc[:, self.feature_cols]
        values = features.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Candidate features must be finite numeric values")
        if cands_df.empty:
            return np.empty(0, dtype=float)

        positive = np.asarray(self.model.predict_proba(features)[:, 1], dtype=float)
        if not np.isfinite(positive).all() or (positive < 0).any():
            raise ValueError("Model returned invalid positive-class probabilities")

        probabilities = np.empty(len(cands_df), dtype=float)
        groups = cands_df.groupby(
            ["trip_id", "obs_idx"], sort=False, dropna=False
        ).indices
        for indices in groups.values():
            indices = np.asarray(indices, dtype=np.intp)
            total = positive[indices].sum()
            if not np.isfinite(total) or total <= 0:
                raise ValueError("Model probabilities for an observation sum to zero")
            probabilities[indices] = positive[indices] / total
        return probabilities
