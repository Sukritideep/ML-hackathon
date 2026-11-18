# Owner: B. Group-aware CV runner that logs one row per model to a CSV.
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from .evaluate import evaluate


def run_experiment(name, family, pipe, config, df_train, df_stu, feature_cols,
                   n_splits=3, max_train_rows=None, seed=42,
                   out_csv="results/experiments.csv"):
    """
    df_train / df_stu: candidate tables with trip_id, obs_idx, label and feature columns.
    pipe: an sklearn Pipeline or estimator with predict_proba.
    max_train_rows: subsample size for slow models (e.g. KNN).
    """
    df_train = df_train.reset_index(drop=True)
    df_stu = df_stu.reset_index(drop=True)
    X = df_train[feature_cols]
    y = df_train["label"].values
    groups = df_train["trip_id"].values
    rng = np.random.default_rng(seed)

    def subsample(idx):
        if max_train_rows and len(idx) > max_train_rows:
            return rng.choice(idx, max_train_rows, replace=False)
        return idx

    # 1) Group-aware cross-validation on the training trips
    oof = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = clone(pipe).fit(X.iloc[subsample(tr)], y[subsample(tr)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    cv = evaluate(df_train, oof)

    # 2) Final fit on all training trips
    idx = subsample(np.arange(len(y)))
    t0 = time.perf_counter()
    model = clone(pipe).fit(X.iloc[idx], y[idx])
    train_time = time.perf_counter() - t0

    train_eval = evaluate(df_train.iloc[idx].reset_index(drop=True),
                          model.predict_proba(X.iloc[idx])[:, 1])

    # 3) Student-test evaluation (never used for fitting)
    t0 = time.perf_counter()
    p_stu = model.predict_proba(df_stu[feature_cols])[:, 1]
    infer_time = time.perf_counter() - t0
    stu = evaluate(df_stu, p_stu)

    row = dict(
        model=name, family=family, config=json.dumps(config, default=str), seed=seed,
        n_splits=n_splits, train_rows_used=len(idx),
        cv_macro_f1=cv["macro_f1"], cv_logloss=cv["log_loss"], cv_top1=cv["top1"],
        train_macro_f1=train_eval["macro_f1"], train_top1=train_eval["top1"],
        stu_macro_f1=stu["macro_f1"], stu_top1=stu["top1"], stu_logloss=stu["log_loss"],
        stu_precision=stu["precision"], stu_recall=stu["recall"],
        train_time_s=train_time, infer_time_s=infer_time,
        infer_us_per_row=1e6 * infer_time / len(df_stu),
    )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv),
                               index=False)
    return row, model, p_stu