"""Viterbi decoding for candidate road-arc sequences."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.hmm.transitions import DirectedRoadGraph, transition_log_probability


@dataclass(frozen=True)
class ViterbiParams:
    """Decoder settings; transition scale is measured in metres."""

    transition_scale_m: float
    break_on_long_gap: bool = True

    def __post_init__(self) -> None:
        try:
            scale = float(self.transition_scale_m)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("transition_scale_m must be finite and positive") from exc
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("transition_scale_m must be finite and positive")
        if not isinstance(self.break_on_long_gap, bool):
            raise TypeError("break_on_long_gap must be a bool")
        object.__setattr__(self, "transition_scale_m", scale)


def _read_params(params: ViterbiParams | Mapping[str, Any]) -> ViterbiParams:
    if isinstance(params, ViterbiParams):
        return params
    if not isinstance(params, Mapping):
        raise TypeError("params must be ViterbiParams or a mapping")
    if "transition_scale_m" not in params:
        raise ValueError("params must include transition_scale_m")
    return ViterbiParams(
        transition_scale_m=params["transition_scale_m"],
        break_on_long_gap=params.get("break_on_long_gap", True),
    )


def _bool_flag(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1"):
            return True
        if normalized in ("false", "0", ""):
            return False
    raise ValueError(f"Invalid long_gap value: {value!r}")


def _is_known_arc(graph: DirectedRoadGraph, arc_id: Any) -> bool:
    try:
        return arc_id in graph.arc_to_endpoints
    except TypeError:
        return False


def _observed_distance(
    candidates: pd.DataFrame, row_positions: Sequence[int], obs_idx: Any
) -> float:
    if "gps_distance_m" not in candidates.columns:
        raise ValueError("candidates must include gps_distance_m to decode transitions")
    distances = pd.to_numeric(
        candidates.iloc[list(row_positions)]["gps_distance_m"], errors="coerce"
    ).to_numpy(dtype=float, na_value=np.nan)
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        raise ValueError(
            f"Observation {obs_idx!r} has no finite gps_distance_m for transition"
        )
    if (distances < 0).any():
        raise ValueError(f"Observation {obs_idx!r} has negative gps_distance_m")
    if not np.allclose(distances, distances[0], rtol=1e-9, atol=1e-9):
        raise ValueError(
            f"Observation {obs_idx!r} has inconsistent gps_distance_m values"
        )
    return float(distances[0])


def _decode_segment(
    candidates: pd.DataFrame,
    segment: Sequence[tuple[Any, list[int]]],
    log_emissions: np.ndarray,
    graph: DirectedRoadGraph,
    transition_scale_m: float,
) -> list[Any] | None:
    """Decode a segment, restarting after transitions with no valid path."""
    candidate_rows = {
        position: candidates.iloc[position].to_dict()
        for _, positions in segment
        for position in positions
    }
    state_rows = [
        [position for position in positions if math.isfinite(log_emissions[position])]
        for _, positions in segment
    ]
    if any(not rows for rows in state_rows):
        return None

    def backtrack(
        rows: Sequence[list[int]],
        pointers: Sequence[np.ndarray],
        final_scores: np.ndarray,
    ) -> list[Any] | None:
        final_index = int(np.argmax(final_scores))
        if not math.isfinite(final_scores[final_index]):
            return None
        selected_rows = [rows[-1][final_index]]
        for pointer_index in range(len(rows) - 1, 0, -1):
            previous_index = int(pointers[pointer_index - 1][final_index])
            if previous_index < 0:
                return None
            final_index = previous_index
            selected_rows.append(rows[pointer_index - 1][final_index])
        selected_rows.reverse()
        return [candidates.iloc[position]["arc_id"] for position in selected_rows]

    decoded_arcs: list[Any] = []
    run_rows = [state_rows[0]]
    backpointers: list[np.ndarray] = []
    scores = log_emissions[run_rows[0]].copy()
    for step in range(1, len(segment)):
        obs_idx, row_positions = segment[step]
        observed_distance = _observed_distance(candidates, row_positions, obs_idx)
        previous_rows = run_rows[-1]
        current_rows = state_rows[step]
        targets_by_source: dict[Any, set[Any]] = {}
        for previous_row in previous_rows:
            previous_endpoints = graph.arc_to_endpoints.get(
                candidate_rows[previous_row]["arc_id"]
            )
            if previous_endpoints is None:
                continue
            source_node = previous_endpoints[1]
            targets = targets_by_source.setdefault(source_node, set())
            for current_row in current_rows:
                current_endpoints = graph.arc_to_endpoints.get(
                    candidate_rows[current_row]["arc_id"]
                )
                if current_endpoints is not None:
                    targets.add(current_endpoints[0])
        for source_node, target_nodes in targets_by_source.items():
            graph.shortest_node_distances(source_node, tuple(target_nodes))

        next_scores = np.full(len(current_rows), -np.inf, dtype=float)
        backpointer = np.full(len(current_rows), -1, dtype=np.intp)

        for current_index, current_row in enumerate(current_rows):
            current_candidate = candidate_rows[current_row]
            current_emission = log_emissions[current_row]
            for previous_index, previous_row in enumerate(previous_rows):
                if not math.isfinite(scores[previous_index]):
                    continue
                previous_candidate = candidate_rows[previous_row]
                transition = transition_log_probability(
                    graph,
                    previous_candidate,
                    current_candidate,
                    observed_distance,
                    transition_scale_m,
                )
                total = scores[previous_index] + transition + current_emission
                if math.isfinite(total) and total > next_scores[current_index]:
                    next_scores[current_index] = total
                    backpointer[current_index] = previous_index

        if not np.isfinite(next_scores).any():
            decoded_run = backtrack(run_rows, backpointers, scores)
            if decoded_run is None:
                return None
            decoded_arcs.extend(decoded_run)
            run_rows = [current_rows]
            backpointers = []
            scores = log_emissions[current_rows].copy()
            continue
        scores = next_scores
        run_rows.append(current_rows)
        backpointers.append(backpointer)

    decoded_run = backtrack(run_rows, backpointers, scores)
    if decoded_run is None:
        return None
    decoded_arcs.extend(decoded_run)
    return decoded_arcs


def decode(
    cands: pd.DataFrame,
    probs: Sequence[float] | np.ndarray,
    graph: DirectedRoadGraph,
    params: ViterbiParams | Mapping[str, Any],
) -> pd.DataFrame:
    """Decode the globally best candidate arc per observation.

    ``probs`` must contain one emission probability per input row, in the same
    order as ``cands``. The input must include ``obs_idx``, ``arc_id``, and,
    for transitions, ``gps_distance_m``. If present, ``trip_id`` keeps trips
    independent and is included in the output. ``long_gap`` is optional; when
    present, a true value starts a new decoding segment by default. Non-
    consecutive observation indices also start new segments because the
    missing observations have no candidate states to connect through.

    The result contains one row per represented observation in chronological
    order. It has ``obs_idx`` and ``pred_arc_id`` columns, plus ``trip_id``
    when that column exists in the input. If all transitions into an
    observation are impossible, the best preceding path is finalized and
    decoding restarts at that observation. No transition is fabricated across
    the break.
    """
    if not isinstance(cands, pd.DataFrame):
        raise TypeError("cands must be a pandas DataFrame")
    options = _read_params(params)

    output_columns = (
        ["trip_id", "obs_idx", "pred_arc_id"]
        if "trip_id" in cands.columns
        else ["obs_idx", "pred_arc_id"]
    )
    if cands.empty:
        return pd.DataFrame(
            {column: pd.Series(dtype="object") for column in output_columns}
        )
    missing_columns = [
        column for column in ("obs_idx", "arc_id") if column not in cands
    ]
    if missing_columns:
        raise ValueError(f"Candidate table is missing columns: {missing_columns}")
    if not isinstance(graph, DirectedRoadGraph):
        raise TypeError("graph must be a DirectedRoadGraph")

    try:
        emission_probabilities = np.asarray(probs, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("probs must be a one-dimensional numeric sequence") from exc
    if emission_probabilities.ndim != 1:
        raise ValueError("probs must be a one-dimensional sequence")
    if emission_probabilities.size != len(cands):
        raise ValueError("probs must have one value for each candidate row")
    if (
        not np.isfinite(emission_probabilities).all()
        or (emission_probabilities < 0).any()
        or (emission_probabilities > 1).any()
    ):
        raise ValueError("probs must contain finite values in [0, 1]")

    log_emissions = np.full(len(cands), -np.inf, dtype=float)
    positive = emission_probabilities > 0
    log_emissions[positive] = np.log(emission_probabilities[positive])
    arc_ids = cands["arc_id"].to_numpy()
    for position, arc_id in enumerate(arc_ids):
        if not _is_known_arc(graph, arc_id):
            log_emissions[position] = -np.inf

    obs_numeric = pd.to_numeric(cands["obs_idx"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    if not np.isfinite(obs_numeric).all():
        raise ValueError("obs_idx values must be finite numeric indices")

    if "trip_id" in cands.columns:
        trip_groups = cands.groupby(
            "trip_id", sort=False, dropna=False
        ).indices.values()
    else:
        trip_groups = [np.arange(len(cands), dtype=np.intp)]

    output_trip_ids: list[Any] = []
    output_obs_indices: list[Any] = []
    output_arcs: list[Any] = []
    obs_values = cands["obs_idx"].to_numpy()
    long_gap_values = (
        cands["long_gap"].to_numpy()
        if "long_gap" in cands.columns
        else np.zeros(len(cands), dtype=bool)
    )

    for trip_positions in trip_groups:
        grouped_rows: dict[Any, list[int]] = {}
        for position in np.asarray(trip_positions, dtype=np.intp):
            obs_idx = obs_values[position]
            if pd.isna(obs_idx):
                raise ValueError("obs_idx values cannot be missing")
            grouped_rows.setdefault(obs_idx, []).append(int(position))

        ordered_observations = sorted(
            grouped_rows,
            key=lambda obs_idx: float(
                pd.to_numeric(pd.Series([obs_idx]), errors="raise").iloc[0]
            ),
        )
        trip_id = (
            cands.iloc[int(trip_positions[0])]["trip_id"]
            if "trip_id" in cands.columns
            else None
        )

        segments: list[list[tuple[Any, list[int]]]] = []
        current_segment: list[tuple[Any, list[int]]] = []
        isolated_observations: list[tuple[Any, list[int]]] = []
        previous_numeric_idx: float | None = None

        for obs_idx in ordered_observations:
            positions = grouped_rows[obs_idx]
            numeric_idx = float(
                pd.to_numeric(pd.Series([obs_idx]), errors="raise").iloc[0]
            )
            long_gap = any(
                _bool_flag(long_gap_values[position]) for position in positions
            )
            has_candidate = any(
                math.isfinite(log_emissions[position]) for position in positions
            )
            if current_segment and (
                (options.break_on_long_gap and long_gap)
                or (
                    previous_numeric_idx is not None
                    and numeric_idx - previous_numeric_idx != 1
                )
            ):
                segments.append(current_segment)
                current_segment = []

            if not has_candidate:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
                isolated_observations.append((obs_idx, positions))
            else:
                current_segment.append((obs_idx, positions))
            previous_numeric_idx = numeric_idx

        if current_segment:
            segments.append(current_segment)

        predictions: dict[Any, Any] = {
            obs_idx: None for obs_idx, _ in isolated_observations
        }
        for segment in segments:
            decoded = _decode_segment(
                cands,
                segment,
                log_emissions,
                graph,
                options.transition_scale_m,
            )
            if decoded is None:
                predictions.update({obs_idx: None for obs_idx, _ in segment})
            else:
                predictions.update(
                    {obs_idx: arc_id for (obs_idx, _), arc_id in zip(segment, decoded)}
                )

        for obs_idx in ordered_observations:
            output_trip_ids.append(trip_id)
            output_obs_indices.append(obs_idx)
            prediction = predictions.get(obs_idx)
            output_arcs.append(None if prediction is None else prediction)

    result_data: dict[str, Any] = {
        "obs_idx": pd.Series(output_obs_indices),
        "pred_arc_id": pd.Series(output_arcs, dtype=object),
    }
    if "trip_id" in cands.columns:
        result_data = {
            "trip_id": pd.Series(output_trip_ids),
            **result_data,
        }
    return pd.DataFrame(result_data, columns=output_columns)
