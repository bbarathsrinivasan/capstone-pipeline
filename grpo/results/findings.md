# GRPO pipeline — first run, mechanics validation

Scope: a GRPO (Group Relative Policy Optimization) implementation grounded
in the TimeSRL paper (arXiv:2605.21295, the plan's own closest precedent)
and the original GRPO paper (DeepSeekMath, arXiv:2402.03300), run against
the clean tier-1 GLOBEM slice (480 samples, same scope as
`src/build_clean_dataset.py`). This is **not** a real MedGemma training
run — see "Infrastructure constraint" below for why, and what to do
differently with real GPU/HF access.

## Infrastructure constraint (found before any data issue)

This sandbox's network policy allows PyPI/npm/crates/the Go module proxy,
but not `huggingface.co` or the PyTorch CDN (`download.pytorch.org`) —
confirmed via the proxy status endpoint, both return 403. Vertex AI /
Google Cloud Storage are also unreachable from here and no `gcloud` auth
is configured in this session, so the GPU/Vertex setup mentioned isn't
reachable from this sandbox. A quick check of the connected-computer bridge
turned up a transient mount/connection error on retry, so that path wasn't
available during this run either.

**Consequence:** no pretrained model or tokenizer could be downloaded.
`grpo/tokenizer_build.py` trains a small byte-level BPE tokenizer from
scratch (fully offline, via the `tokenizers` package) on the prompt corpus
itself, and `grpo/model.py` builds a randomly-initialized ~1.2M-parameter
GPT-2-architecture model — a real transformer, just with no language
knowledge at all. **Everything below tests whether the GRPO training loop
is implemented correctly, not whether this approach can produce clinically
useful summaries** — that second question needs a real pretrained model
(MedGemma or otherwise), which needs either network access to a model hub
from this sandbox or a working connection to your own GPU/Vertex setup.

## What was implemented, grounded in the papers

- **Reward** (`reward.py`): TimeSRL's actual formula, confirmed from the
  paper — `r(p,y) = exp(-(p-y)^2 / (2*sigma^2))`, sigma=1.2 — on the 0-3
  risk scale, with 0 reward for any unparseable/out-of-range prediction.
  Plus this project's own safety-shaped variant (the plan's stated
  contribution over TimeSRL): the same curve, with an extra penalty
  multiplier (0.15x) specifically when the true label is Severe and the
  predicted score is <=1.5 (a Minimal/Mild call). See
  `chart_reward_shape.png` — the two curves are identical everywhere
  except that one failure mode. **Caveat:** TimeSRL's own paper has no
  asymmetric safety cost, so there's no published formula to match here —
  this specific penalty (0.15x) is this pipeline's own construction and
  should be reviewed by whoever owns reward-function design (Rishi, per
  the Sep 20 sync) before it's treated as final.
- **Freeze one layer, update the other**: LoRA (Hu et al. 2021,
  arXiv:2106.09685), rank 16 / alpha 32 — the exact setting reported in the
  TimeSRL paper. `model.py` freezes every base-model weight and wraps only
  the attention projections with trainable low-rank adapters: 32,768
  trainable parameters out of 1,186,304 total (2.76%). The base model and
  a frozen reference copy (for the KL penalty) share the same starting
  weights; only the policy's LoRA adapters ever get a gradient update.
- **GRPO mechanics** (`train.py`): group sampling (G=4 completions per
  prompt), reward computed per completion, advantage = reward normalized
  within its own group (`(r - group_mean) / (group_std + eps)`) — no
  learned value/critic network, exactly GRPO's point. Policy-gradient loss
  weights each completion's length-normalized log-probability by its
  group-relative advantage, plus a KL penalty against the frozen reference
  model. **One deliberate simplification**: a single gradient step per
  group, no PPO-style clipped multi-epoch replay over the same rollouts —
  reasonable for a mechanics check, not for a real run (see "Next steps").
- **Prompt format** (`prompts.py`): matches the paper's actual input style
  — raw per-day feature dicts with descriptive labels and units (e.g.
  "Sleep Duration (hours): {D1: 6.5, D2: ...}"), not pre-aggregated stats —
  built directly from the raw daily values in `anchored_globem_hrv.json`
  for tier-1 rows, asking for a short summary followed by a parseable
  `Risk score: N` line.

## Main run result: zero reward, every step

12 GRPO steps, 6 prompts/step, group size 4 (288 total rollouts). Mean
reward and parseable-output rate were **exactly 0.000 on every single
step** (`chart_main_run.png`) — in 288 free-form generations, the
untrained policy never once produced a parseable `Risk score: N` line
within the 90-token budget.

This matters for two reasons, not just one:

1. **No learning ever happens.** With every rollout in a group scoring the
   same reward (0), the group-relative advantage is `(0-0)/(0+eps) = 0`
   for every sample, every step. The loss and KL numbers move slightly
   (visible in `grpo_run.json`), but that's LoRA-dropout noise, not a real
   policy update — there is no gradient signal to learn from until at
   least some rollouts in a group succeed.
2. **This is a cold-start problem, not (necessarily) a bug** — verified
   below.

## Sanity check: the training loop itself works

Before trusting #2, I swapped in an artificially easy, continuous reward
(shorter completions score higher — `reward = -length/100`, no exact
string match required) and reused the identical GRPO update code
(`sanity_check.py`). Result (`chart_sanity_check.png`): real, nonzero
loss every step, and the shortest completion in a batch repeatedly spikes
down to 18-27 tokens (vs. a ~100-token ceiling) on steps where the
sampled group happened to contain a lucky short completion — exactly the
pattern you'd expect from a working group-relative-advantage update
reinforcing favorable samples. **This confirms the GRPO implementation
itself is functioning correctly** — the flat-zero result in the main run
is a genuine finding about the task, not an implementation bug: a
structured free-text-plus-exact-format task is not learnable by GRPO from
a cold, randomly-initialized policy within this budget.

This directly argues for the plan's own original Oct-then-Nov ordering
(cold-start SFT, then GRPO) — contrary to what the TimeSRL paper excerpt
available to me suggested (it doesn't describe an SFT stage, "training
appears to begin directly with GRPO"). That may hold for a *pretrained*
4B-parameter LLM, which already knows how to follow a "respond in this
exact format" instruction out of the box; it visibly does not hold for a
policy with zero language pretraining. Worth flagging: even with a real
pretrained MedGemma checkpoint, confirm it reliably hits the target format
zero-shot before assuming GRPO alone will bootstrap it — if it doesn't, a
short cold-start/format-SFT pass first is cheap insurance.

## New data issues found while building prompts from raw daily values

Building prompts directly from the per-day dicts (rather than the
pre-aggregated stats in `src/build_dataset.py`) surfaced two things not
visible in the earlier aggregate-level audit:

- **`bt_unique_devices` values are implausible as a literal device count**
  — daily values in the hundreds to low thousands (e.g. 358-3576 in one
  sample's week). A personal phone does not typically see that many
  distinct Bluetooth devices in a day; this is more likely a cumulative
  scan-event count or similar GLOBEM-internal artifact than literally
  "unique nearby devices." Worth getting the real definition from
  GLOBEM's feature docs or whoever ingested this batch before it's shown
  to a model (or a clinician) under a literal "devices nearby" label.
- **`bedtime_min` can exceed 1440** (minutes in a day) — one sample shows
  1581, i.e. "26.35 hours after midnight." That's presumably a
  day-boundary convention (a 2am bedtime stored as 1560 rather than
  wrapping to 120) rather than an error, but it needs confirming before
  converting these to human-readable clock times in a later prompt
  version — converting naively (1581/60 = 26.35h) would show a
  physically-impossible time to the model.
- **Confirms, at the row level now, the missingness already flagged in
  the earlier audit**: the example used throughout this doc
  (`tier1_globem_0000`) has sleep-related readings on only 2 of 7 days.
  Feeding that directly as a daily dict (rather than a pre-smoothed
  weekly stat) makes the sparsity visible to the model as `null` values
  it has to reason around — which is realistic, but means prompt design
  needs an explicit policy for how a model should talk about a week with
  mostly-missing sleep data, not left implicit.

## Next steps, in priority order

1. **Get real model access working** — either network access to a model
   hub from a compute environment, or a working connection to your
   GPU/Vertex setup — before any of these numbers (reward, parse rate)
   are meaningful signals about the actual approach rather than this
   sandbox's constraints.
2. **Add a cold-start/format-SFT pass** before GRPO, even a short one — a
   few hundred (stats -> correctly-formatted output) pairs, even
   teacher-generated, so GRPO has *some* rollouts clearing the
   parse-success bar to learn from. The sanity check above shows the GRPO
   code is ready for this the moment reward stops being uniformly zero.
3. Resolve the `bt_unique_devices` and `bedtime_min` questions above
   before scaling prompt generation to the full batch.
4. Once a real model is in the loop, re-run the safety-shaped vs. plain
   TimeSRL reward comparison for real — the shape in `chart_reward_shape.png`
   is confirmed correct, but its effect on training has only been
   exercised in isolation (the reward-shape demo), not inside an actual
   non-degenerate training run yet.
