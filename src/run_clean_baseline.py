"""
The clean pipeline, step by step.

This file is meant to be read top to bottom as a walkthrough, not just run.
It trains/evaluates on data/samples_clean.csv (480 tier-1 GLOBEM rows only
-- see build_clean_dataset.py for why) using three models of increasing
complexity, so you can see how much a given step actually buys you.

Run:
    python build_dataset.py         # raw json -> data/samples.csv
    python build_clean_dataset.py   # -> data/samples_clean.csv (tier 1 only)
    python run_clean_baseline.py    # this file
"""
import json
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

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "samples_clean.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# STEP 1: load the clean table and separate identity/label columns from
# the actual model inputs.
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA)

NON_FEATURE_COLS = {
    "sample_id", "subject_id", "source", "tier", "window",
    "risk_category", "risk_int", "phq4_score", "label_source_scale",
}
feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

X = df[feature_cols]
y = df["risk_int"].values          # 0=minimal, 1=mild, 2=moderate, 3=severe
groups = df["subject_id"].values   # who the row came from -- for splitting

print(f"Loaded {len(df)} rows, {len(feature_cols)} feature columns, "
      f"{df['subject_id'].nunique()} unique subjects.")
print(f"Class balance: {df['risk_category'].value_counts().to_dict()}")
print("Note: tier-1 GLOBEM is EXACTLY balanced (120/class). That's almost "
      "certainly a curated/subsampled set, not natural PHQ-9 prevalence -- "
      "see results/clean_findings.md for what that does and doesn't mean "
      "for these numbers.\n")

# ---------------------------------------------------------------------------
# STEP 2: why subject-grouped splitting matters here.
#
# A plain KFold/StratifiedKFold splits rows at random. But 130+ subjects in
# this batch show up in more than one weekly window, and the same person's
# two different weeks are NOT independent samples -- if one lands in train
# and the other in test, the model can partly "recognize the person" rather
# than generalize from behavior to risk. StratifiedGroupKFold keeps every
# row for a given subject_id on the same side of the split, while still
# trying to balance the 4 classes across folds.
# ---------------------------------------------------------------------------
N_FOLDS = 5
N_REPEATS = 5  # repeat with different shuffles so Severe Recall (small n)
               # isn't reported off a single lucky/unlucky split


def majority_baseline(y_train, y_test):
    """Floor: always predict the most common class seen in training."""
    maj = np.bincount(y_train, minlength=4).argmax()
    y_pred = np.full_like(y_test, maj)
    proba = np.zeros((len(y_test), 4))
    proba[:, maj] = 1.0
    return y_pred, proba


def logreg_fit_predict(X_train, y_train, X_test):
    """Linear baseline. Needs imputed, scaled input (no native NaN support)."""
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)
    return pipe.predict(X_test), pipe.predict_proba(X_test)


def xgb_fit_predict(X_train, y_train, X_test):
    """Tree-ensemble baseline -- the plan's designated floor-to-beat.
    Handles NaN natively, no imputation needed."""
    clf = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=4,
        eval_metric="mlogloss", missing=np.nan, verbosity=0,
    )
    clf.fit(X_train, y_train)
    return clf.predict(X_test), clf.predict_proba(X_test)


# ---------------------------------------------------------------------------
# STEP 3: run the CV loop. For every fold: fit each model on the train
# split, predict on the held-out split, score with the plan's safety
# metrics (metrics.py), and keep every fold's scores so we can report a
# mean +/- spread rather than one number from one split.
# ---------------------------------------------------------------------------
fold_reports = {"majority": [], "logreg": [], "xgboost": []}
raw_preds = {"xgboost": {"y_true": [], "y_pred": []}}  # for the confusion-matrix chart

for seed in range(N_REPEATS):
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for name, fit_predict in [
            ("majority", majority_baseline),
            ("logreg", logreg_fit_predict),
            ("xgboost", xgb_fit_predict),
        ]:
            if name == "majority":
                y_pred, y_proba = fit_predict(y_train, y_test)
            else:
                y_pred, y_proba = fit_predict(X_train, y_train, X_test)
            fold_reports[name].append(full_report(y_test, y_pred, y_proba))
            if name == "xgboost":
                raw_preds["xgboost"]["y_true"].extend(y_test.tolist())
                raw_preds["xgboost"]["y_pred"].extend(y_pred.tolist())

with open(RESULTS / "clean_predictions.json", "w") as f:
    json.dump(raw_preds, f)

# ---------------------------------------------------------------------------
# STEP 4: aggregate fold-level scores into mean/std/95% range per metric.
# ---------------------------------------------------------------------------
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
            }
    summary[model] = agg

out_path = RESULTS / "clean_baseline_results.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'Model':<12} {'Severe Recall':<16} {'Macro F1':<12} {'Crit.Fail.Rate':<16} {'Accuracy':<10} {'ECE':<8}")
for model in ["majority", "logreg", "xgboost"]:
    m = summary[model]
    def fmt(k):
        return f"{m[k]['mean']:.3f}" if k in m else "n/a"
    print(f"{model:<12} {fmt('severe_recall'):<16} {fmt('macro_f1'):<12} "
          f"{fmt('critical_failure_rate'):<16} {fmt('accuracy'):<10} {fmt('ece'):<8}")

print(f"\nFull results -> {out_path}")
