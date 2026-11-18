# Metrics for candidate rows and observation-level top-1.
import numpy as np
from sklearn.metrics import (log_loss, f1_score, precision_score,
                             recall_score, confusion_matrix)

def evaluate(df, proba, thr=0.5):
    """df has trip_id, obs_idx, label (one row per candidate).
    proba = predicted P(label=1), same length and order as df."""
    y = df["label"].values
    pred = (proba >= thr).astype(int)

    d = df[["trip_id", "obs_idx", "label"]].copy().reset_index(drop=True)
    d["p"] = np.asarray(proba)
    # Top-1: for each GPS point, is the highest-probability candidate the true one?
    top1 = d.loc[d.groupby(["trip_id", "obs_idx"])["p"].idxmax()]["label"].mean()
    # Candidate miss: points where no candidate is true
    has_true = d.groupby(["trip_id", "obs_idx"])["label"].max()

    return {
        "macro_f1": f1_score(y, pred, average="macro"),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "log_loss": log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1]),
        "top1": top1,
        "miss_rate": 1 - has_true.mean(),
        "cm": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }