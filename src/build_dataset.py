"""
Parse anchored_globem_hrv.json into a flat, modeling-ready feature table.

Output: data/samples.csv  -- one row per sample_id with:
  - identity/label columns (sample_id, subject_id, source, tier, window,
    risk_category, risk_int, label_source_scale)
  - engineered features, namespaced by source: globem__<feat>_<stat> /
    hrv__<feat>_<stat>
  - hrs (kept, but excluded from the default training feature set -- see
    schema.py)

Also writes data/feature_manifest.json documenting what every engineered
column means and which raw source it came from, so this isn't a black box.
"""
import json
import statistics as stats
from pathlib import Path

import pandas as pd

from schema import GLOBEM_FEATURES, HRV_FEATURES, RISK_TO_INT

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "anchored_globem_hrv.json"
OUT_SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples.csv"
OUT_MANIFEST = Path(__file__).resolve().parent.parent / "data" / "feature_manifest.json"


def _daily_stats(day_dict):
    """mean / std / n_observed / missing_rate over the D1..D7 dict."""
    if not isinstance(day_dict, dict):
        return dict(mean=None, std=None, n_observed=0, missing_rate=1.0)
    vals = [v for v in day_dict.values() if v is not None]
    n = len(day_dict)
    n_obs = len(vals)
    mean = stats.mean(vals) if vals else None
    std = stats.pstdev(vals) if len(vals) > 1 else (0.0 if vals else None)
    missing_rate = 1 - (n_obs / n) if n else 1.0
    return dict(mean=mean, std=std, n_observed=n_obs, missing_rate=missing_rate)


def _range_day_fracs(range_days):
    """within/below/above/unknown counts -> fractions of the 7-day window."""
    if not isinstance(range_days, dict):
        return dict(within_frac=None, below_frac=None, above_frac=None)
    total = sum(range_days.values()) or 1
    return dict(
        within_frac=range_days.get("within", 0) / total,
        below_frac=range_days.get("below", 0) / total,
        above_frac=range_days.get("above", 0) / total,
    )


def _flag_indicators(flag):
    flag = flag or ""
    return dict(
        flag_sparse=1 if "sparse_window" in flag else 0,
        flag_no_observed=1 if "no_observed_days" in flag else 0,
        flag_descriptive_only=1 if "descriptive_only" in flag else 0,
    )


def _engineer_feature(record, feat, namespace, manifest):
    out = {}
    day_dict = record.get(feat)
    for stat_name, val in _daily_stats(day_dict).items():
        col = f"{namespace}__{feat}_{stat_name}"
        out[col] = val
        manifest.setdefault(col, f"{stat_name} of raw daily '{feat}' values ({namespace} source)")

    pctile = record.get(f"{feat}__pctile")
    col = f"{namespace}__{feat}_pctile"
    out[col] = pctile
    manifest.setdefault(col, f"cohort percentile rank of '{feat}' for this window ({namespace} source)")

    for stat_name, val in _range_day_fracs(record.get(f"{feat}__range_days")).items():
        col = f"{namespace}__{feat}_{stat_name}"
        out[col] = val
        manifest.setdefault(
            col, f"fraction of observed days {stat_name.split('_')[0]} the reference range for '{feat}'"
        )

    for stat_name, val in _flag_indicators(record.get(f"{feat}__flag")).items():
        col = f"{namespace}__{feat}_{stat_name}"
        out[col] = val
        manifest.setdefault(col, f"data-quality flag indicator derived from '{feat}__flag'")

    return out


def build():
    with open(RAW_PATH) as f:
        raw = json.load(f)

    manifest = {}
    rows = []
    for rec in raw:
        row = {
            "sample_id": rec["sample_id"],
            "subject_id": rec["subject_id"],
            "source": rec["source"],
            "tier": rec["tier"],
            "window": rec["window"],
            "risk_category": rec["risk_category"],
            "risk_int": RISK_TO_INT[rec["risk_category"]],
            "phq4_score": rec["phq4_score"],
            "hrs": rec["hrs"],
            "label_source_scale": (
                "globem_phq9_mapped" if rec["source"] == "globem" else "hrv_derived_phq9_gad7"
            ),
        }
        feats = GLOBEM_FEATURES if rec["source"] == "globem" else HRV_FEATURES
        namespace = rec["source"]
        for feat in feats:
            row.update(_engineer_feature(rec, feat, namespace, manifest))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_SAMPLES, index=False)
    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"Wrote {len(df)} rows x {df.shape[1]} columns -> {OUT_SAMPLES}")
    print(f"Feature manifest ({len(manifest)} engineered columns) -> {OUT_MANIFEST}")
    return df


if __name__ == "__main__":
    build()
