"""Temporary distance-only model interface for downstream development."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DistanceStubBundle:
    """Return a per-observation softmax of negative candidate distance."""

    temperature_m: float = 10.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.temperature_m) or self.temperature_m <= 0:
            raise ValueError("temperature_m must be finite and positive")

    def predict_proba(self, cands_df: pd.DataFrame) -> np.ndarray:
        required = ("trip_id", "obs_idx", "f_dist")
        missing = [col for col in required if col not in cands_df.columns]
        if missing:
            raise ValueError(f"Candidate table is missing stub inputs: {missing}")

        distances = pd.to_numeric(cands_df["f_dist"], errors="raise").to_numpy(
            dtype=float
        )
        if not np.isfinite(distances).all() or (distances < 0).any():
            raise ValueError("f_dist must contain finite, nonnegative distances")

        probabilities = np.empty(len(cands_df), dtype=float)
        groups = cands_df.groupby(
            ["trip_id", "obs_idx"], sort=False, dropna=False
        ).indices
        for indices in groups.values():
            indices = np.asarray(indices, dtype=np.intp)
            logits = -distances[indices] / self.temperature_m
            weights = np.exp(logits - logits.max())
            probabilities[indices] = weights / weights.sum()
        return probabilities
