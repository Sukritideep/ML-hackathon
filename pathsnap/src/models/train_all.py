# training the 5 models with group CV, logs results, builds comparison table and plots.
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from .cv import run_experiment
from .data import CANDIDATE_COLS, FEATURE_COLS, load_candidate_tables

SEED = 42
DEFAULT_FEATURES = list(FEATURE_COLS)


def get_models():
    """name -> (family, estimator, config dict, max_train_rows)"""
    return {
        "logreg": ("Linear",
                   make_pipeline(StandardScaler(),
                                 LogisticRegression(C=1.0, max_iter=500, random_state=SEED)),
                   {"C": 1.0, "max_iter": 500}, None),
        "gnb": ("Probabilistic",
                make_pipeline(StandardScaler(), GaussianNB()),
                {"var_smoothing": 1e-9}, None),
        "knn": ("Instance-based",
                make_pipeline(StandardScaler(),
                              KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)),
                {"k": 15, "weights": "distance"}, 150_000),
        "rf": ("Ensemble",
               RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=5,
                                      n_jobs=-1, random_state=SEED),
               {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 5}, None),
        "lgbm": ("Boosting",
                 LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                random_state=SEED, verbose=-1),
                 {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31}, None),
    }


def load_fake():
    df = pd.read_parquet("results/fake_table.parquet")
    trips = sorted(df["trip_id"].unique())
    cut = int(len(trips) * 0.8)
    return (df[df["trip_id"].isin(trips[:cut])].reset_index(drop=True),
            df[df["trip_id"].isin(trips[cut:])].reset_index(drop=True))


def load_real(cand_train, cand_stu, lab_train, lab_stu):
    """Merge A's candidates with C's labels: label = (arc_id == true_arc_id)."""
    key_cols = ["trip_id", "obs_idx"]
    candidate_key_cols = [*key_cols, "arc_id"]
    label_cols = [*key_cols, "true_arc_id"]

    def merge(candidates_path, labels_path, split_name):
        candidates = pd.read_parquet(candidates_path)
        labels = pd.read_parquet(labels_path)
        if list(candidates.columns) != list(CANDIDATE_COLS):
            raise ValueError(
                f"{split_name} candidates must have columns {list(CANDIDATE_COLS)}"
            )
        if list(labels.columns) != label_cols:
            raise ValueError(f"{split_name} labels must have columns {label_cols}")
        if candidates.duplicated(candidate_key_cols).any():
            raise ValueError(f"{split_name} candidates contain duplicate keys")
        if labels.duplicated(key_cols).any():
            raise ValueError(f"{split_name} labels contain duplicate keys")
        for frame_name, frame in (("candidates", candidates), ("labels", labels)):
            ids_are_padded = frame["trip_id"].map(
                lambda value: isinstance(value, str)
                and len(value) == 8
                and value.isdecimal()
            )
            if frame["trip_id"].isna().any() or not ids_are_padded.all():
                raise ValueError(f"{split_name} {frame_name} IDs must be 8-digit strings")
            if not pd.api.types.is_integer_dtype(frame["obs_idx"]):
                raise ValueError(f"{split_name} {frame_name} obs_idx must be integers")
        if set(candidates["trip_id"].unique()) != set(labels["trip_id"].unique()):
            raise ValueError(f"{split_name} candidate and label trip IDs do not match")
        if not pd.api.types.is_integer_dtype(candidates["arc_id"]):
            raise ValueError(f"{split_name} candidate arc_id values must be integers")
        if not np.isfinite(candidates.loc[:, FEATURE_COLS].to_numpy(dtype=float)).all():
            raise ValueError(f"{split_name} candidate features contain non-finite values")
        ordered_keys = candidates[candidate_key_cols].sort_values(
            candidate_key_cols, kind="mergesort"
        ).reset_index(drop=True)
        if not candidates[candidate_key_cols].reset_index(drop=True).equals(ordered_keys):
            raise ValueError(f"{split_name} candidate keys are not sorted")

        numeric_arc_ids = pd.to_numeric(labels["true_arc_id"], errors="raise")
        known_arc_ids = numeric_arc_ids.dropna().to_numpy(dtype=float)
        if (
            not np.isfinite(known_arc_ids).all()
            or (known_arc_ids % 1 != 0).any()
        ):
            raise ValueError(f"{split_name} true_arc_id values must be integers or null")
        labels["true_arc_id"] = pd.Series(
            pd.array(numeric_arc_ids, dtype="Int64"), index=labels.index
        )
        aligned_labels = labels[labels["true_arc_id"].notna()].copy()
        if aligned_labels.empty:
            raise ValueError(f"{split_name} has no aligned labels")

        df = candidates.merge(
            aligned_labels, on=key_cols, how="inner", validate="many_to_one"
        )
        df["label"] = df["arc_id"].eq(df["true_arc_id"]).astype("int8")
        positive_keys = df.loc[df["label"].eq(1), key_cols].drop_duplicates()
        candidate_misses = len(aligned_labels) - len(positive_keys)
        if positive_keys.empty:
            raise ValueError(f"{split_name} has no labels matching a candidate arc")

        # Candidate misses are not valid all-negative training/evaluation groups.
        df = df.merge(positive_keys, on=key_cols, how="inner", validate="many_to_one")
        df = df.sort_values(candidate_key_cols, kind="mergesort").reset_index(drop=True)
        print(
            f"[{split_name}] aligned={len(aligned_labels):,}/{len(labels):,}; "
            f"unmatched={len(labels) - len(aligned_labels):,}; "
            f"candidate_misses_excluded={candidate_misses:,}; "
            f"candidate_rows_kept={len(df):,}"
        )
        return df

    return (
        merge(cand_train, lab_train, "train"),
        merge(cand_stu, lab_stu, "student_test"),
    )


def make_plots(res, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    names = res["model"].tolist()
    for cols, title, fname in [
        (["cv_macro_f1"], "Cross-validation macro F1 (train trips, GroupKFold)", "cv_macro_f1"),
        (["stu_top1", "stu_macro_f1"], "Student-test top-1 accuracy and macro F1", "student_top1_f1"),
        (["cv_logloss", "stu_logloss"], "Log loss (lower is better)", "log_loss"),
    ]:
        ax = res.set_index("model")[cols].plot(kind="bar", figsize=(8, 4), rot=20)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{tag}_{fname}.png"), dpi=150)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="use results/fake_table.parquet")
    ap.add_argument("--cand-train", default="results/candidates_train.parquet")
    ap.add_argument("--cand-stu", default="results/candidates_student.parquet")
    ap.add_argument("--lab-train", default="results/labels_train.parquet")
    ap.add_argument("--lab-stu", default="results/labels_student.parquet")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--only", nargs="*", help="run only these model names")
    ap.add_argument(
        "--prepare-candidates",
        action="store_true",
        help="convert split-listed Member 1 feature CSVs to candidate Parquet tables",
    )
    ap.add_argument(
        "--split-file",
        type=Path,
        default=Path("splits.json"),
        help="train/student split file used by --prepare-candidates",
    )
    ap.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("..") / "member1_preprocessing" / "outputs" / "features",
        help="directory containing Member 1's per-trip feature CSVs",
    )
    ap.add_argument(
        "--overwrite-candidates",
        action="store_true",
        help="allow replacing existing candidate Parquet files",
    )
    args = ap.parse_args()

    if args.prepare_candidates:
        if args.fake:
            ap.error("--prepare-candidates cannot be combined with --fake")
        output_paths = [Path(args.cand_train), Path(args.cand_stu)]
        existing = [path for path in output_paths if path.exists()]
        if existing and not args.overwrite_candidates:
            ap.error(
                "Candidate outputs already exist; pass --overwrite-candidates to "
                "replace them: " + ", ".join(str(path) for path in existing)
            )

        candidates_train, candidates_stu, imputed = load_candidate_tables(
            args.split_file, args.feature_dir
        )
        for path, table in zip(output_paths, (candidates_train, candidates_stu)):
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(path, index=False)
            print(f"Wrote {len(table)} rows to {path}")
        for name, count in sorted(imputed.items()):
            print(f"Filled {count} missing {name} values with 0.0")
        return

    if args.fake:
        df_train, df_stu = load_fake()
        out_csv, tag = "results/experiments_fake.csv", "fake"
        feats = DEFAULT_FEATURES
    else:
        df_train, df_stu = load_real(args.cand_train, args.cand_stu, args.lab_train, args.lab_stu)
        out_csv, tag = "results/experiments.csv", "real"
        feats = DEFAULT_FEATURES
    print(f"[{tag}] train rows={len(df_train)}, student rows={len(df_stu)}, features={feats}")

    if os.path.exists(out_csv):
        os.remove(out_csv)  # fresh log for this comparison run

    rows = []
    for name, (family, est, cfg, max_rows) in get_models().items():
        if args.only and name not in args.only:
            continue
        print(f"--- {name} ---")
        row, _, _ = run_experiment(name, family, est, cfg, df_train, df_stu, feats,
                                   n_splits=args.splits, max_train_rows=max_rows,
                                   seed=SEED, out_csv=out_csv)
        rows.append(row)
        print({k: round(v, 4) for k, v in row.items() if isinstance(v, (float, np.floating))})

    res = pd.DataFrame(rows)
    table_cols = ["model", "family", "config", "cv_macro_f1", "cv_logloss", "cv_top1",
                  "stu_top1", "stu_macro_f1", "stu_logloss", "train_macro_f1",
                  "train_time_s", "infer_time_s"]
    res[table_cols].to_csv(f"results/comparison_{tag}.csv", index=False)
    make_plots(res, "results/plots", tag)
    print(res[["model", "cv_macro_f1", "cv_logloss", "stu_top1", "stu_macro_f1"]])


if __name__ == "__main__":
    main()