"""Distance-based HMM emission scores for road candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = ("trip_id", "obs_idx", "arc_id", "distance_m")


@dataclass
class DistanceEmissionModel:
    """Baseline candidate emission scores from GPS-to-road distance.

    The baseline assigns weights proportional to
    ``exp(-distance_m / scale_m)`` and normalizes them over candidates for each
    ``(trip_id, obs_idx)``. This is a distance-only baseline, not a trained ML
    emission model. Output arrays retain the input DataFrame's row order, so
    each probability/log probability remains aligned with its original
    observation index and arc ID.
    """

    scale_m: float = 10.0

    def __post_init__(self) -> None:
        try:
            scale = float(self.scale_m)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("scale_m must be finite and positive") from exc
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale_m must be finite and positive")
        self.scale_m = scale

    @staticmethod
    def _validate_candidates(candidates: pd.DataFrame) -> None:
        if not isinstance(candidates, pd.DataFrame):
            raise TypeError("candidates must be a pandas DataFrame")
        if candidates.empty:
            return
        missing = [column for column in _REQUIRED_COLUMNS if column not in candidates]
        if missing:
            raise ValueError(f"Candidate table is missing required columns: {missing}")

    def predict_log_proba(self, candidates: pd.DataFrame) -> np.ndarray:
        """Return normalized per-row log probabilities in input row order.

        Missing, nonnumeric, negative, or non-finite distances receive
        ``-inf`` (zero probability). Valid candidates in the same observation
        are normalized together. An observation with no valid distances has
        zero probability for every candidate.
        """
        self._validate_candidates(candidates)
        row_count = len(candidates)
        log_probabilities = np.full(row_count, -np.inf, dtype=float)
        if row_count == 0:
            return log_probabilities

        distances = pd.to_numeric(
            candidates["distance_m"], errors="coerce"
        ).to_numpy(dtype=float, na_value=np.nan)
        valid = np.isfinite(distances) & (distances >= 0)
        groups = candidates.groupby(
            ["trip_id", "obs_idx"], sort=False, dropna=False
        ).indices.values()

        for indices in groups:
            indices = np.asarray(indices, dtype=np.intp)
            valid_indices = indices[valid[indices]]
            if valid_indices.size == 0:
                continue

            group_distances = distances[valid_indices]
            min_distance = float(group_distances.min())
            with np.errstate(over="ignore", invalid="ignore"):
                logits = -(group_distances - min_distance) / self.scale_m
            log_normalizer = float(np.logaddexp.reduce(logits))
            log_probabilities[valid_indices] = logits - log_normalizer

        return log_probabilities

    def predict_proba(self, candidates: pd.DataFrame) -> np.ndarray:
        """Return per-row probabilities in the same order as ``candidates``."""
        return np.exp(self.predict_log_proba(candidates))

    def score_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Copy candidate rows and append emission probability columns."""
        log_probabilities = self.predict_log_proba(candidates)
        scored = candidates.copy()
        scored["emission_log_probability"] = log_probabilities
        scored["emission_probability"] = np.exp(log_probabilities)
        return scored
