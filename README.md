# Capstone baseline pipeline — anchored_globem_hrv batch

Baseline pipeline for the Clinically Grounded LLMs capstone, run against the
first data batch (`data/anchored_globem_hrv.json`, 598 samples: 480 globem
"tier 1" + 78 globem "tier 3" + 40 hrv). Purpose: establish the Oct-milestone
feature-based floor (per the Fall implementation plan) and surface concrete
flaws before the semantic-bottleneck + GRPO work starts. This is **not** the
plan's full two-stage method.

There are two passes through this data, run in sequence:

1. **Diagnostic pass** (`run_baselines.py`) — uses everything in the batch,
   on purpose, specifically to go looking for problems: a label-leakage
   check on the `hrs` field, a source-shortcut ablation, and a
   leave-one-source-out cross-modality test. Results: `results/findings.md`.
2. **Clean pass** (`build_clean_dataset.py` + `run_clean_baseline.py`) —
   once the diagnostic pass identified what's untrustworthy (`hrs`,
   tier 3's undocumented provenance, hrv's zero-Severe-class problem), this
   trains only on the one fully-documented slice: tier-1 GLOBEM, 480 rows.
   Written to be read top-to-bottom as a walkthrough of how the pipeline
   works, not just run. Results: `results/clean_findings.md`.

Read `results/clean_findings.md` first if you just want the current
floor-to-beat number; read `results/findings.md` for why tier 3/hrv/`hrs`
were excluded in the first place.

## Setup

```
pip install -r requirements.txt
```

## Run

```
cd src
python build_dataset.py        # raw json -> data/samples.csv + data/feature_manifest.json
python run_baselines.py        # diagnostic pass -> results/baseline_results.json
python build_clean_dataset.py  # filter to tier-1 GLOBEM only -> data/samples_clean.csv
python run_clean_baseline.py   # clean pass -> results/clean_baseline_results.json
python make_plots.py           # -> results/chart_*.png (both passes)
```

## Layout

- `src/schema.py` — which raw fields belong to which source's feature
  namespace, plus the fields excluded from modeling and why.
- `src/build_dataset.py` — parses the raw per-sample JSON (daily dicts +
  percentile/range/flag metadata) into one flat row per sample with
  engineered features, namespaced `globem__*` / `hrv__*` so the two
  modalities' differing units are never silently merged.
- `src/metrics.py` — Severe Recall, Macro F1, Critical Failure Rate
  (true Severe → predicted Minimal/Mild), ECE — per the plan's §5.
- `src/run_baselines.py` — the diagnostic pass: subject-grouped, stratified,
  repeated CV harness; runs the default baseline plus three diagnostic
  experiments (hrs-leakage demo, source-shortcut ablation,
  leave-one-source-out).
- `src/build_clean_dataset.py` — filters to tier-1 GLOBEM only and drops
  dead/leaky columns; explains why in its docstring.
- `src/run_clean_baseline.py` — the clean pass, written as an annotated
  walkthrough (load → why subject-grouped splitting → train → aggregate).
  Read this file if you want to understand how the pipeline works, not just
  its output.
- `src/make_plots.py` — the charts referenced in both findings docs.
- `results/findings.md` / `results/clean_findings.md` — the write-ups.
- `data/feature_manifest.json` — human-readable definition of every
  engineered column, for anyone who didn't write this code.

## Headline results

**Diagnostic pass (all 598 rows):** XGBoost Severe Recall 0.21, logistic
regression 0.38 — both below the plan's 0.40–0.55 honest floor. Including
`hrs` as a feature pushes Severe Recall to 0.99–1.00 — that's leakage, not
signal; `hrs` is excluded from both passes' default feature sets.

**Clean pass (tier-1 GLOBEM only, 480 rows):** XGBoost Severe Recall 0.27,
logistic regression 0.35 — same conclusion on cleaner data. Also surfaces
that tier-1 GLOBEM is *exactly* class-balanced (120/class), almost
certainly curated rather than natural prevalence, and that once that
imbalance is removed, XGBoost's confusion matrix is close to flat — the
current simple features (daily mean/std/percentile) carry little separable
signal on their own. See `results/clean_findings.md` for what that does and
doesn't imply.
