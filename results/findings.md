# Baseline pipeline results — first data batch (anchored_globem_hrv.json)

Run: `src/build_dataset.py` → `src/run_baselines.py` → `src/make_plots.py`.
598 samples (558 globem, 40 hrv), subject-grouped `StratifiedGroupKFold`
(5 folds × 5 repeats = 25 CV runs), models: majority-class floor, logistic
regression, XGBoost. Metrics per the plan's §5: Severe Recall, Macro F1,
Critical Failure Rate, ECE.

This is the Oct-milestone baseline ("advanced ML baselines as the floor to
beat"), run specifically to surface flaws before any semantic-bottleneck /
GRPO work starts on this batch.

## 1. The floor is barely above chance, and the plan's "near random" problem is already visible here

| Model | Severe Recall | Macro F1 | Critical Failure Rate | Accuracy |
|---|---|---|---|---|
| Majority class | 0.00 | 0.13 | 1.00 | 0.34 |
| Logistic regression | 0.38 | 0.27 | 0.36 | 0.28 |
| XGBoost | 0.21 | 0.26 | 0.56 | 0.30 |

Both real models score **below the majority-class baseline on accuracy**,
and XGBoost is well under the plan's own "honest floor" of 0.40–0.55 Severe
Recall for December — on this batch it's at 0.21, with 56% of true-Severe
samples landing on Minimal/Mild. Logistic regression (with balanced class
weights) does better on Severe Recall but is still short of the floor. This
matches the plan's existing worry that raw-feature classification "sits near
random" — now with concrete numbers from real data rather than an assumption.
See `chart_confusion_matrix.png`: every true class's modal prediction is
"Minimal."

Likely contributors, in rough order of suspected impact: the engineered
features are coarse (daily mean/std/percentile, no temporal dynamics within
the week), the 480+78 "globem" rows pool two batches of unknown relationship
(see §4), and 598 samples split across 4 classes with subject-grouping is a
small-data regime for gradient boosting.

## 2. `hrs` is not safe to use as a feature — demonstrated, not just suspected

Re-running the identical pipeline with `hrs` added back in:

| Model | Severe Recall (no hrs → with hrs) | Critical Failure Rate | Accuracy |
|---|---|---|---|
| Logistic regression | 0.38 → **0.99** | 0.36 → 0.00 | 0.28 → 0.97 |
| XGBoost | 0.21 → **1.00** | 0.56 → 0.00 | 0.30 → 0.99 |

See `chart_hrs_leakage.png`. This isn't "a strong feature" — it's a model
reconstructing the label almost perfectly from a single column (ECE also
collapses to ~0.01–0.13, i.e. suspiciously perfect calibration). `hrs` must
be either dropped or have its actual derivation documented and confirmed
independent of `risk_category` before it's anywhere near a feature table.

## 3. Source (globem vs. hrv) isn't being used as a shortcut, for what that's worth

Removing the one-hot "is this an hrv sample" indicator changed every metric
by <0.01 across all three models. So the weak baseline performance in §1
isn't coming from the model leaning on a source shortcut — it's failing to
extract signal from the behavioral features themselves, which is the more
fundamental problem.

## 4. Cross-modality transfer fails in the direction that matters most

| Train → Test | Severe Recall | Critical Failure Rate | Macro F1 | Accuracy | Classes seen in training |
|---|---|---|---|---|---|
| globem → hrv | n/a (0 severe in hrv) | n/a | 0.21 | 0.73 | all 4 |
| **hrv → globem** | **0.00** | **1.00** | 0.09 | 0.23 | **only 3 (no Severe)** |

The hrv batch contains zero Severe-labeled samples, so a model trained only
on it structurally cannot predict Severe at all — every true-Severe globem
sample gets misclassified when transferred. This is a direct, concrete
instance of the cross-modality generalization question the plan lists as a
contribution (§4) failing in one direction. It's also a strong argument for
keeping HRV in the "enrichment only" role the plan's dataset table already
assigns it, rather than treating it as interchangeable with GLOBEM in
pooled training — and for not evaluating Severe Recall on any split that
could end up hrv-only.

## 5. Open items this run couldn't resolve (carried over / new)

- **GLOBEM batch composition**: 558 "globem" rows here split into tier 1
  (n=480, matches the plan's GLOBEM row) and tier 3 (n=78, same schema,
  zero subject overlap with tier 1, undocumented origin) — see the earlier
  data audit (`data-audits/anchored_globem_hrv-batch1.md`). Not re-litigated
  here, but it directly affects whether "pooled globem" in this report means
  what the plan's table says it means.
- **PHQ-9→PHQ-4 mapping** still can't be audited from this file (no raw PHQ-9
  retained for globem rows) — the plan's own "key open item."
- **`label_source_scale`** still doesn't exist as a real field; this report's
  per-source split stands in for it, but the reward function will need the
  real thing.
- Feature engineering here is intentionally simple (daily mean/std/percentile
  only) — worth a pass with richer temporal features before concluding the
  behavioral signal itself is weak, rather than just the current
  featurization.
