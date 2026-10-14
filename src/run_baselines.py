"""
Baseline pipeline (Oct milestone per the Fall plan): establish the
feature-based floor-to-beat before any semantic-bottleneck / GRPO work,
using safety-weighted metrics, with subject-level splits so a subject's
repeat weekly windows never cross train/test.

Also runs three diagnostic experiments specifically aimed at surfacing
flaws in the current data batch:
  1. hrs-leakage demo      -- what happens if the near-label-collinear
                               `hrs` field is (wrongly) used as a feature.
  2. source-shortcut ablation -- how much the model leans on knowing
                               source=globem/hrv rather than behavior itself.
  3. leave-one-source-out  -- train on one modality, test on the other, to
                               probe the plan's "cross-modality generalization"
                               claim directly.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from metrics import full_report

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "samples.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

N_FOLDS = 5
N_REPEATS = 5
RANDOM_SEEDS = list(range(N_REPEATS))

NON_FEATURE_COLS = {
    "sample_id", "subject_id", "source", "tier", "window",
    "risk_category", "risk_int", "phq4_score", "label_source_scale", "hrs",
}


def load():
    df = pd.read_csv(SAMPLES)
    return df


def feature_columns(df, include_source=True, include_hrs=False):
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    if include_hrs:
        cols = cols + ["hrs"]
    X = df[cols].copy()
    if include_source:
        X["source_is_hrv"] = (df["source"] == "hrv").astype(int)
    return X


def majority_baseline(y_train, y_test):
    maj = np.bincount(y_train, minlength=4).argmax()
    y_pred = np.full_like(y_test, maj)
    # fabricate a degenerate one-hot proba for ECE's sake
    proba = np.zeros((len(y_test), 4))
    proba[:, maj] = 1.0
    return y_pred, proba


def logreg_fit_predict(X_train, y_train, X_test):
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)
    return pipe.predict(X_test), pipe.predict_proba(X_test)


def xgb_fit_predict(X_train, y_train, X_test):
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        missing=np.nan,
        verbosity=0,
    )
    clf.fit(X_train, y_train)
    return clf.predict(X_test), clf.predict_proba(X_test)


def run_pooled_cv(df, include_source=True, include_hrs=False, tag="", collect_predictions=False):
    X = feature_columns(df, include_source=include_source, include_hrs=include_hrs)
    y = df["risk_int"].values
    groups = df["subject_id"].values

    fold_reports = {"majority": [], "logreg": [], "xgboost": []}
    fold_fold_severity_counts = []
    all_preds = {"majority": {"y_true": [], "y_pred": []},
                 "logreg": {"y_true": [], "y_pred": []},
                 "xgboost": {"y_true": [], "y_pred": []}}

    for seed in RANDOM_SEEDS:
        skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            fold_fold_severity_counts.append(int((y_test == 3).sum()))

            yp, pp = majority_baseline(y_train, y_test)
            fold_reports["majority"].append(full_report(y_test, yp, pp))
            if collect_predictions:
                all_preds["majority"]["y_true"].extend(y_test.tolist())
                all_preds["majority"]["y_pred"].extend(yp.tolist())

            yp, pp = logreg_fit_predict(X_train, y_train, X_test)
            fold_reports["logreg"].append(full_report(y_test, yp, pp))
            if collect_predictions:
                all_preds["logreg"]["y_true"].extend(y_test.tolist())
                all_preds["logreg"]["y_pred"].extend(yp.tolist())

            yp, pp = xgb_fit_predict(X_train, y_train, X_test)
            fold_reports["xgboost"].append(full_report(y_test, yp, pp))
            if collect_predictions:
                all_preds["xgboost"]["y_true"].extend(y_test.tolist())
                all_preds["xgboost"]["y_pred"].extend(yp.tolist())

    summary = {}
    for model, reports in fold_reports.items():
        agg = {}
        for metric in ["severe_recall", "macro_f1", "critical_failure_rate", "accuracy", "ece"]:
            vals = [r[metric] for r in reports if metric in r and not np.isnan(r[metric])]
            if vals:
                agg[metric] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "p2.5": float(np.percentile(vals, 2.5)),
                    "p97.5": float(np.percentile(vals, 97.5)),
                    "n_folds_with_severe": len(vals) if metric in ("severe_recall", "critical_failure_rate") else None,
                }
        summary[model] = agg

    summary["_meta"] = {
        "tag": tag,
        "n_folds_total": N_FOLDS * N_REPEATS,
        "folds_with_zero_severe_in_test": int(sum(1 for c in fold_fold_severity_counts if c == 0)),
        "min_severe_in_a_test_fold": int(min(fold_fold_severity_counts)),
        "max_severe_in_a_test_fold": int(max(fold_fold_severity_counts)),
    }
    if collect_predictions:
        return summary, all_preds
    return summary


def run_leave_one_source_out(df):
    results = {}
    for train_source, test_source in [("globem", "hrv"), ("hrv", "globem")]:
        train_df = df[df["source"] == train_source]
        test_df = df[df["source"] == test_source]

        # Union feature space: columns the test source doesn't have become
        # all-NaN -- this is the realistic cross-modality scenario, not a bug.
        X_train = feature_columns(train_df, include_source=False)
        X_test = feature_columns(test_df, include_source=False)
        X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=np.nan)

        y_train = train_df["risk_int"].values
        y_test = test_df["risk_int"].values

        classes_in_train = sorted(set(y_train))
        yp, pp_raw = xgb_fit_predict(X_train, y_train, X_test)
        # pad proba columns back out to 4 classes for ECE/report consistency
        pp = np.zeros((len(y_test), 4))
        for i, c in enumerate(classes_in_train):
            pp[:, c] = pp_raw[:, i]

        report = full_report(y_test, yp, pp)
        report["classes_available_in_training"] = classes_in_train
        results[f"train={train_source}_test={test_source}"] = report
    return results


def main():
    df = load()

    print("=== Experiment 1: pooled baseline (default feature set, hrs excluded) ===")
    pooled, pooled_preds = run_pooled_cv(
        df, include_source=True, include_hrs=False, tag="pooled_default", collect_predictions=True
    )
    print(json.dumps(pooled["_meta"], indent=2))
    with open(RESULTS / "pooled_default_predictions.json", "w") as f:
        json.dump(pooled_preds, f)

    print("=== Experiment 2: source-shortcut ablation (no source feature) ===")
    no_source = run_pooled_cv(df, include_source=False, include_hrs=False, tag="pooled_no_source")

    print("=== Experiment 3: hrs-leakage demonstration (hrs included as a feature) ===")
    with_hrs = run_pooled_cv(df, include_source=True, include_hrs=True, tag="pooled_with_hrs_leakage")

    print("=== Experiment 4: leave-one-source-out cross-modality generalization ===")
    loso = run_leave_one_source_out(df)

    all_results = {
        "pooled_default": pooled,
        "pooled_no_source_ablation": no_source,
        "pooled_with_hrs_leakage_demo": with_hrs,
        "leave_one_source_out": loso,
    }
    out_path = RESULTS / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote full results -> {out_path}")


if __name__ == "__main__":
    main()
