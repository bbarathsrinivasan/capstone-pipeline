"""
Build the "clean" training set: tier-1 GLOBEM only.

Why tier 1 only (480 of the 598 rows in the batch):
  - It is the one slice whose provenance is fully known: it matches the Fall
    plan's GLOBEM row (N=480) exactly.
  - Tier 3 (78 rows, also source="globem") has zero subject overlap with
    tier 1 and no documented origin -- pooling it in silently would let an
    unexplained batch shape the model.
  - The hrv subset (40 rows) has zero Severe-class examples and, per
    results/findings.md Experiment 4, a model trained on it cannot predict
    Severe at all when applied elsewhere. It stays out of the *training* set
    for the same reason the plan's own dataset table marks it
    "enrichment only," not a primary source.
  - `hrs` is excluded everywhere (see schema.py) -- it's demonstrated label
    leakage, not a feature.

Run after build_dataset.py (this reads its output, data/samples.csv).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "samples.csv"
OUT_PATH = ROOT / "data" / "samples_clean.csv"


def build():
    df = pd.read_csv(IN_PATH)

    clean = df[df["tier"] == 1].copy()

    # Drop columns that only ever apply to the hrv namespace -- for tier-1
    # rows these are 100% NaN by construction, so they're dead weight, not
    # missing data to impute.
    hrv_cols = [c for c in clean.columns if c.startswith("hrv__")]
    clean = clean.drop(columns=hrv_cols)

    # hrs stays out of the modeling table entirely in the clean pass (not
    # even kept-but-unused) -- see docstring.
    clean = clean.drop(columns=["hrs"])

    # Some engineered columns (the *_within_frac / *_below_frac / *_above_frac
    # trio) only exist for raw features that carry a defined "normal range"
    # in the source data. bedtime_min, home_time_min, bt_unique_devices and
    # the screen_* features never have one, so those columns are 100% NaN
    # for every tier-1 row -- not missing data to impute, just dead columns.
    all_nan_cols = [c for c in clean.columns if clean[c].isna().all()]
    clean = clean.drop(columns=all_nan_cols)

    clean.to_csv(OUT_PATH, index=False)

    print(f"Clean set: {len(clean)} rows x {clean.shape[1]} columns -> {OUT_PATH}")
    print(f"Dropped {len(df) - len(clean)} non-tier-1 rows "
          f"({(df['tier'] != 1).sum()} rows: tier 3 globem + hrv)")
    print(f"Dropped {len(hrv_cols)} hrv-only columns, 'hrs', and "
          f"{len(all_nan_cols)} always-empty range columns "
          f"({clean.shape[1]} feature/id columns remain)")
    print(f"risk_category distribution:\n{clean['risk_category'].value_counts()}")
    return clean


if __name__ == "__main__":
    build()
