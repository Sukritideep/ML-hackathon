"""Optuna tuning and export for the PathSnap LightGBM candidate model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Callable

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from .bundle import LightGBMBundle
from .data import FEATURE_COLS
from .evaluate import evaluate
from .train_all import load_real

SEED = 42
DEFAULT_TRIALS = 20
DEFAULT_SPLITS = 3
BASELINE_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": SEED,
    "verbosity": -1,
}


def _validate_training_frame(
    df: pd.DataFrame, feature_cols: list[str] | tuple[str, ...], n_splits: int
) -> None:
    required = {"trip_id", "obs_idx", "label", *feature_cols}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Training table is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Training table is empty")
    if df["trip_id"].isna().any() or df["obs_idx"].isna().any():
        raise ValueError("Training trip_id and obs_idx values cannot be null")
    if not set(df["label"].unique()).issubset({0, 1}) or df["label"].nunique() != 2:
        raise ValueError("Training labels must contain both binary classes")
    if not np.isfinite(df.loc[:, feature_cols].to_numpy(dtype=float)).all():
        raise ValueError("Training features must be finite numeric values")
    group_count = df["trip_id"].nunique()
    if n_splits < 2 or n_splits > group_count:
        raise ValueError(
            f"n_splits must be between 2 and the {group_count} available trips"
        )


def _lgbm_params(
    search_params: dict[str, Any], seed: int = SEED
) -> dict[str, Any]:
    return {
        **search_params,
        "objective": "binary",
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
        "subsample_freq": 1,
        "force_col_wise": True,
    }


def _suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    max_depth = trial.suggest_int("max_depth", 4, 10)
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, min(63, 2**max_depth)),
        "max_depth": max_depth,
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }


def _fold_log_losses(
    estimator_factory: Callable[[], Any],
    df: pd.DataFrame,
    feature_cols: list[str] | tuple[str, ...],
    n_splits: int,
) -> list[float]:
    X = df.loc[:, feature_cols]
    y = df["label"].to_numpy(dtype=np.int8)
    groups = df["trip_id"].to_numpy()
    losses = []
    for train_idx, valid_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = estimator_factory()
        model.fit(X.iloc[train_idx], y[train_idx])
        probabilities = model.predict_proba(X.iloc[valid_idx])[:, 1]
        losses.append(
            float(
                evaluate(
                    df.iloc[valid_idx],
                    np.asarray(probabilities, dtype=float),
                )["log_loss"]
            )
        )
    return losses


def _objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    feature_cols: list[str] | tuple[str, ...],
    n_splits: int,
    seed: int,
) -> float:
    params = _lgbm_params(_suggest_params(trial), seed=seed)
    losses = _fold_log_losses(
        lambda: lgb.LGBMClassifier(**params), df, feature_cols, n_splits
    )
    trial.set_user_attr("fold_log_losses", losses)
    return float(np.mean(losses))


def _write_trial_log(study: optuna.Study, path: Path) -> None:
    rows = []
    for trial in study.get_trials(deepcopy=False):
        duration = (
            trial.duration.total_seconds() if trial.duration is not None else None
        )
        rows.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "cv_log_loss": trial.value,
                "duration_s": duration,
                "params_json": json.dumps(trial.params, sort_keys=True),
                "fold_log_losses_json": json.dumps(
                    trial.user_attrs.get("fold_log_losses", [])
                ),
                "started_at": (
                    trial.datetime_start.isoformat()
                    if trial.datetime_start is not None
                    else None
                ),
                "completed_at": (
                    trial.datetime_complete.isoformat()
                    if trial.datetime_complete is not None
                    else None
                ),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_csv(temp_path, index=False)
    temp_path.replace(path)


def _evaluate_configuration(
    label: str,
    params: dict[str, Any],
    df_train: pd.DataFrame,
    df_student: pd.DataFrame,
    feature_cols: list[str] | tuple[str, ...],
    n_splits: int,
) -> tuple[dict[str, Any], Any]:
    X = df_train.loc[:, feature_cols]
    y = df_train["label"].to_numpy(dtype=np.int8)
    groups = df_train["trip_id"].to_numpy()
    oof = np.empty(len(df_train), dtype=float)

    for train_idx, valid_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = lgb.LGBMClassifier(**params)
        model.fit(X.iloc[train_idx], y[train_idx])
        oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
    cv_metrics = evaluate(df_train, oof)

    final_model = lgb.LGBMClassifier(**params)
    train_started = time.perf_counter()
    final_model.fit(X, y)
    train_seconds = time.perf_counter() - train_started

    inference_started = time.perf_counter()
    student_probabilities = final_model.predict_proba(
        df_student.loc[:, feature_cols]
    )[:, 1]
    inference_seconds = time.perf_counter() - inference_started
    student_metrics = evaluate(df_student, student_probabilities)
    row = {
        "model": label,
        "config": json.dumps(params, sort_keys=True),
        "cv_log_loss": cv_metrics["log_loss"],
        "cv_macro_f1": cv_metrics["macro_f1"],
        "cv_top1": cv_metrics["top1"],
        "student_log_loss": student_metrics["log_loss"],
        "student_macro_f1": student_metrics["macro_f1"],
        "student_top1": student_metrics["top1"],
        "train_time_s": train_seconds,
        "inference_time_s": inference_seconds,
        "inference_us_per_candidate": (
            1e6 * inference_seconds / len(df_student) if len(df_student) else 0.0
        ),
    }
    return row, final_model


def run_tuning(
    df_train: pd.DataFrame,
    df_student: pd.DataFrame,
    *,
    n_trials: int = DEFAULT_TRIALS,
    n_splits: int = DEFAULT_SPLITS,
    seed: int = SEED,
    study_name: str = "lgbm_logloss",
    database_path: str | Path = "results/optuna/lgbm_tuning.db",
    trial_log_path: str | Path = "results/optuna/lgbm_trials.csv",
    comparison_path: str | Path = "results/optuna/tuning_comparison.csv",
    model_path: str | Path = "artifacts/model_bundle.joblib",
    config_path: str | Path = "artifacts/config.json",
    summary_path: str | Path = "results/optuna/tuning_summary.json",
    timeout: int | None = None,
) -> dict[str, Any]:
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    _validate_training_frame(df_train, FEATURE_COLS, n_splits)
    _validate_training_frame(df_student, FEATURE_COLS, 2)

    database_path = Path(database_path)
    trial_log_path = Path(trial_log_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{database_path.resolve().as_posix()}"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage,
        load_if_exists=True,
    )
    started = time.perf_counter()
    study.optimize(
        lambda trial: _objective(
            trial, df_train, FEATURE_COLS, n_splits, seed=seed
        ),
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[lambda current_study, _: _write_trial_log(current_study, trial_log_path)],
    )
    elapsed = time.perf_counter() - started
    _write_trial_log(study, trial_log_path)

    completed_trials = [
        trial
        for trial in study.get_trials(deepcopy=False)
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        raise RuntimeError("Optuna completed no trials; no model can be exported")
    total_trial_compute_seconds = sum(
        trial.duration.total_seconds()
        for trial in completed_trials
        if trial.duration is not None
    )

    best_params = _lgbm_params(study.best_trial.params, seed=seed)
    rows = []
    baseline_row, _ = _evaluate_configuration(
        "lgbm_baseline",
        _lgbm_params(
            {
                key: value
                for key, value in BASELINE_PARAMS.items()
                if key not in {"random_state", "verbosity"}
            },
            seed=seed,
        ),
        df_train,
        df_student,
        FEATURE_COLS,
        n_splits,
    )
    rows.append(baseline_row)
    tuned_row, tuned_model = _evaluate_configuration(
        "lgbm_tuned",
        best_params,
        df_train,
        df_student,
        FEATURE_COLS,
        n_splits,
    )
    rows.append(tuned_row)

    comparison_path = Path(comparison_path)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(comparison_path, index=False)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(LightGBMBundle(tuned_model), model_path)

    config = {
        "model": "lightgbm",
        "feature_cols": list(FEATURE_COLS),
        "seed": seed,
        "cv": {"method": "GroupKFold", "n_splits": n_splits, "group": "trip_id"},
        "objective": "mean GroupKFold candidate-row log loss",
        "study_name": study_name,
        "study_database": str(database_path),
        "trial_log": str(trial_log_path),
        "completed_trials_total": len(completed_trials),
        "best_cv_log_loss": study.best_value,
        "total_trial_compute_time_s": total_trial_compute_seconds,
        "baseline_params": BASELINE_PARAMS,
        "best_params": best_params,
        "comparison_csv": str(comparison_path),
        "model_bundle": str(model_path),
        "comparison": rows,
    }
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    summary = {
        "study_name": study_name,
        "completed_trials_total": len(completed_trials),
        "best_cv_log_loss": study.best_value,
        "best_params": study.best_trial.params,
        "total_trial_compute_time_s": total_trial_compute_seconds,
        "last_run_compute_time_s": elapsed,
    }
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"Completed {len(completed_trials)} Optuna trials; "
        f"best CV log loss={study.best_value:.6f}; "
        f"total trial compute time={total_trial_compute_seconds:.1f}s"
    )
    print(pd.DataFrame(rows).to_string(index=False))
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune PathSnap's LightGBM model")
    parser.add_argument("--cand-train", default="results/candidates_train.parquet")
    parser.add_argument("--cand-stu", default="results/candidates_student.parquet")
    parser.add_argument("--lab-train", default="results/labels_train.parquet")
    parser.add_argument("--lab-stu", default="results/labels_student.parquet")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_SPLITS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--study-name", default="lgbm_logloss")
    args = parser.parse_args()

    df_train, df_student = load_real(
        args.cand_train, args.cand_stu, args.lab_train, args.lab_stu
    )
    print(
        f"Using {df_train['trip_id'].nunique()} training trips "
        f"and {df_student['trip_id'].nunique()} student-test trips; "
        "student-test is reserved for final comparison."
    )
    run_tuning(
        df_train,
        df_student,
        n_trials=args.n_trials,
        n_splits=args.n_splits,
        seed=args.seed,
        study_name=args.study_name,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()