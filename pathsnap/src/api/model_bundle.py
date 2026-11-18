from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..models.data import FEATURE_COLS


@dataclass
class ClassifierBundle:
    estimator: Any
    feature_cols: tuple[str, ...] = FEATURE_COLS

    def predict_proba(self, cands_df: pd.DataFrame) -> np.ndarray:
        missing = [column for column in self.feature_cols if column not in cands_df]
        if missing:
            raise ValueError(f"Candidate table is missing model features: {missing}")

        values = np.asarray(
            self.estimator.predict_proba(cands_df.loc[:, self.feature_cols]),
            dtype=float,
        )
        classes = list(getattr(self.estimator, "classes_", ()))
        if values.ndim != 2 or values.shape[0] != len(cands_df) or 1 not in classes:
            raise ValueError("Classifier must return binary probabilities for class 1")
        probabilities = values[:, classes.index(1)]
        if (
            not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
        ):
            raise ValueError("Classifier returned invalid probabilities")
        return probabilities
