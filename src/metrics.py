"""
Safety-weighted evaluation metrics, per the plan's §5 (Severe Recall, Macro
F1, Critical Failure Rate, ECE).
"""
import numpy as np
from sklearn.metrics import f1_score, recall_score

SEVERE = 3  # risk_int for "severe"
SAFE_LEVELS = {0, 1}  # minimal, mild -- a Severe sample landing here is the
                       # specific failure mode the plan's reward function
                       # is designed to punish hardest.


def severe_recall(y_true, y_pred):
    mask = y_true == SEVERE
    if mask.sum() == 0:
        return np.nan
    return recall_score(y_true == SEVERE, y_pred == SEVERE, pos_label=True, zero_division=0)


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2, 3], zero_division=0)


def critical_failure_rate(y_true, y_pred):
    """Fraction of true-Severe samples predicted as Minimal or Mild
    (a >=2-level downgrade) -- the clinically dangerous miss."""
    mask = y_true == SEVERE
    if mask.sum() == 0:
        return np.nan
    preds = y_pred[mask]
    return np.mean([p in SAFE_LEVELS for p in preds])


def expected_calibration_error(y_true, y_pred, y_proba, n_bins=10):
    """Standard ECE on the top-1 predicted class confidence."""
    confidences = y_proba.max(axis=1)
    correct = (y_pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() == 0:
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_acc - bin_conf)
    return ece


def full_report(y_true, y_pred, y_proba=None):
    out = {
        "severe_recall": severe_recall(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "critical_failure_rate": critical_failure_rate(y_true, y_pred),
        "accuracy": float(np.mean(y_true == y_pred)),
        "n": int(len(y_true)),
        "n_severe": int((y_true == SEVERE).sum()),
    }
    if y_proba is not None:
        out["ece"] = expected_calibration_error(y_true, y_pred, y_proba)
    return out
