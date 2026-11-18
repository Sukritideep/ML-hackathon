import numpy as np
import pandas as pd

from src.hmm.ml_evaluation import evaluate_ml_trip_frames
from src.hmm.transitions import DirectedRoadGraph
from src.models.bundle import LightGBMBundle


class _ProbabilityModel:
    def predict_proba(self, features):
        positive = np.where(features["f_dist"].to_numpy() == 0.0, 0.6, 0.4)
        return np.column_stack((1.0 - positive, positive))


def test_trained_probabilities_preserve_candidate_order_and_feed_viterbi():
    candidates = pd.DataFrame(
        {
            "trip_id": ["00000001"] * 4,
            "obs_idx": [0, 0, 1, 1],
            "arc_id": [11, 10, 10, 11],
            "f_dist": [1.0, 0.0, 0.0, 1.0],
            "f_rank": [0, 1, 0, 1],
            "f_dist_gap": [1.0, 0.0, 0.0, 1.0],
            "f_proj": [0.2, 0.3, 0.4, 0.5],
            "f_bearing_diff": [0.0] * 4,
            "f_speed": [2.0] * 4,
        }
    )
    transition_features = pd.DataFrame(
        {
            "trip_id": ["00000001"] * 4,
            "obs_idx": [1, 0, 1, 0],
            "arc_id": [11, 10, 10, 11],
            "candidate_progress_m": [0.0, 0.0, 0.0, 0.0],
            "gps_distance_m": [10.0, np.nan, 10.0, np.nan],
            "long_gap": [False] * 4,
        }
    )
    labels = pd.DataFrame(
        {
            "trip_id": ["00000001"] * 2,
            "obs_idx": [0, 1],
            "true_arc_id": pd.Series([10, 11], dtype="Int64"),
        }
    )
    graph = DirectedRoadGraph(
        arc_ids=[10, 11],
        from_nodes=[1, 2],
        to_nodes=[2, 3],
        length_m=[10.0, 10.0],
    )

    metrics, predictions = evaluate_ml_trip_frames(
        "00000001",
        candidates,
        transition_features,
        labels,
        graph,
        LightGBMBundle(_ProbabilityModel()),
        transition_scale_m=0.1,
    )

    assert predictions["classifier_pred_arc_id"].tolist() == [10, 10]
    assert predictions["hmm_pred_arc_id"].tolist() == [10, 11]
    assert metrics["classifier_only"]["pointwise_accuracy"] == 0.5
    assert metrics["hmm_viterbi"]["pointwise_accuracy"] == 1.0
