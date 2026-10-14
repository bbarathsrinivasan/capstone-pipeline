# Clean pipeline results — tier-1 GLOBEM only (480 rows)

Run: `build_dataset.py` → `build_clean_dataset.py` → `run_clean_baseline.py`
→ `make_plots.py`. Scope, by design (see `build_clean_dataset.py`
docstring): tier-1 GLOBEM only — the one slice of the first data batch with
fully known provenance, matching the plan's GLOBEM row (N=480) exactly.
Tier 3 (undocumented origin), hrv (zero Severe examples), and `hrs`
(demonstrated label leakage) are all excluded, not just deprioritized.

## Headline numbers

| Model | Severe Recall | Macro F1 | Critical Failure Rate | Accuracy | ECE |
|---|---|---|---|---|---|
| Majority class | 0.12 | 0.10 | 0.88 | 0.25 | 0.75 |
| Logistic regression | 0.35 | 0.27 | 0.40 | 0.27 | 0.21 |
| XGBoost | 0.27 | 0.24 | 0.47 | 0.25 | 0.26 |

(5×5 repeated, subject-grouped, class-stratified CV — see
`run_clean_baseline.py` for exactly how.)

Same conclusion as the full-batch run, now on a cleaner, better-understood
slice: both real models sit below the plan's own "honest floor" of
0.40–0.55 Severe Recall, and accuracy is barely above the 0.25 you'd get by
guessing one of 4 classes at random with no model at all.

## One thing this clean cut surfaces that the pooled run masked

**Tier-1 GLOBEM is *exactly* balanced: 120/120/120/120 across the four
classes.** That's not a coincidence of pooling — it's the data as
delivered. Two consequences worth flagging to the team:

1. This is almost certainly a **curated or subsampled** set, not natural
   PHQ-9 survey prevalence (real depression-risk prevalence in a study
   population is essentially never a perfect 25/25/25/25 split). Worth
   confirming with whoever assembled this batch — if it was downsampled
   from a larger, naturally-skewed pool, the *rate* numbers here (e.g.
   accuracy, PPV) won't transfer to deployment, where Minimal/Mild are
   almost certainly more common than Severe. Severe Recall and Critical
   Failure Rate are less sensitive to this (they're computed within the
   true-Severe group only), so they're the more trustworthy numbers to
   carry forward.
2. See `chart_clean_confusion_matrix.png`: with class imbalance removed,
   XGBoost's predictions are close to flat across all four true classes
   (roughly 0.2–0.3 everywhere, no real diagonal). On the full pooled batch
   the confusion matrix looked a little more structured, but that was
   largely the imbalanced class prior doing the work (predict "Minimal"
   often, be right somewhat often). On balanced data, the current features
   are close to uninformative. That's a more honest read of where things
   actually stand than the pooled numbers gave.

## What this does and doesn't tell you

It tells you: the daily-mean/std/percentile features currently engineered
from this tier-1 batch don't carry much separable signal for 4-class risk,
independent of any class-imbalance effects. It does not yet tell you
whether richer features (temporal trends across the week, cross-week
deltas, interactions) would do better, or whether the ceiling is set by the
behavioral data itself — that's the next thing worth testing before
concluding the semantic-bottleneck approach is the only way to beat this
floor.
