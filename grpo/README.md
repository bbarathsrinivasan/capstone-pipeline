# GRPO pipeline (mechanics validation)

A GRPO implementation grounded in the TimeSRL paper (arXiv:2605.21295) and
the GRPO paper (arXiv:2402.03300), run on the clean tier-1 GLOBEM data.
**Read `results/findings.md` first** — it explains an important constraint:
no pretrained model was reachable from this sandbox (no `huggingface.co` /
PyTorch CDN access, no Vertex AI credentials configured), so this run uses
a small from-scratch tokenizer + randomly-initialized model to validate
that the GRPO training loop itself is implemented correctly. It is not a
MedGemma run and the generations are not meaningful text — see the
findings doc for what it does show, and what to change for a real run.

## Run order

```
cd grpo
python tokenizer_build.py   # train a from-scratch BPE tokenizer (offline)
python train.py             # main GRPO run -> results/grpo_run.json
python sanity_check.py      # verification: proves the GRPO update code
                             # works given a learnable reward (ruling out
                             # a training-loop bug) -> results/sanity_check_run.json
python make_plots.py        # -> results/chart_*.png
```

## Files

- `prompts.py` — builds TimeSRL-style prompts from the RAW daily dicts in
  `data/anchored_globem_hrv.json` (tier-1 only), not the pre-aggregated
  stats used by the baseline pipeline in `../src/`.
- `reward.py` — TimeSRL's Gaussian-decay reward and this project's
  safety-shaped variant; run directly (`python reward.py`) to see the
  reward table for a few (predicted, true) pairs.
- `tokenizer_build.py` — trains the offline BPE tokenizer.
- `model.py` — builds the small GPT-2-architecture base model, the frozen
  reference copy, and the LoRA-wrapped trainable policy (base frozen,
  only rank-16 adapters trainable — the "freeze one layer, update the
  other" piece).
- `train.py` — the GRPO training loop itself, written to be read top to
  bottom: generate a group of rollouts, score them, compute group-relative
  advantage, compute the policy-gradient + KL loss, update. This is the
  file to read to understand how GRPO works mechanically.
- `sanity_check.py` — a verification run with an easy synthetic reward,
  to confirm the main run's zero-reward result is a cold-start finding,
  not a bug in the training code.
- `make_plots.py` — the three charts referenced in `results/findings.md`.

## Headline result

Zero reward and zero parseable predictions across all 12 steps of the main
run — expected from a randomly-initialized policy on a hard structured
task, and confirmed (via the sanity check) to be a cold-start problem, not
a training-loop bug. Strongest actionable takeaway: a cold-start/format-SFT
pass before GRPO isn't optional for this kind of policy, and should be
re-verified against a real pretrained model once one is reachable.

## Real-model track (requires real network access — NOT this sandbox)

The four files below replace the from-scratch stand-in with an actual
pretrained model, and add the cold-start fix (SFT) the findings doc called
for. This sandbox's proxy blocks huggingface.co (confirmed 403), so these
must be run from a machine with normal internet access — e.g. this
project's own Mac, in a regular Terminal, from inside `capstone_pipeline/`:

```
cd grpo
pip install -r ../requirements.txt
python zero_shot_check.py   # how much of the cold-start gap a real small
                             # instruct model already closes, zero-shot
python sft.py                # the cold-start fix: short supervised warm-up
                              # teaching the output FORMAT -> checkpoints/sft_lora/
python train_real.py         # GRPO on the real model, starting from the
                              # SFT checkpoint -> results/grpo_real_run.json
```

- `model_real.py` — loads a real pretrained model (default: `Qwen/Qwen3-0.6B`,
  the smallest member of the same family TimeSRL used at 4B — see its
  `SCALE_UP_PATH` for the 0.6B -> 1.7B -> 4B progression) and wraps it with
  the same LoRA freeze/update setup as `model.py`, just with the real
  architecture's attention-projection module names.
- `zero_shot_check.py` — measures the chosen model's parse-success rate on
  this project's actual prompts with ZERO training, to see how big a gap
  SFT actually needs to close before assuming either way.
- `sft.py` — the cold-start fix itself: supervised fine-tuning on
  (prompt -> correctly-formatted completion) pairs built from the real
  tier-1 records, teaching the output format before GRPO has to. Targets
  are deliberately only ~80% label-accurate (see its docstring) so GRPO
  still has real accuracy left to learn, not just a memorized shortcut.
- `train_real.py` — the same GRPO algorithm as `train.py`, pointed at the
  real model and (if present) the SFT checkpoint, with group/batch/step
  sizes that scale automatically with `GPU_SCALE` (see below).

Why SFT is needed at all (the cold-start problem, restated): GRPO only
learns from the REWARD DIFFERENCE between rollouts in a group. If every
rollout in a group scores the same reward — 0, because none of them
produced a parseable `Risk score: N` line — the group-relative advantage
is `(0-0)/(0+eps) = 0` for all of them, so there is no gradient signal at
all, no matter how many steps run. A policy has to already clear a minimum
"produces something rewardable" bar before GRPO has anything to refine.
`sanity_check.py` already proved the GRPO update code itself is correct
given ANY learnable signal — SFT is what supplies that minimum bar for
this specific, format-constrained task.

**Confirmed on the Mac, Qwen3-0.6B:** zero-shot parse rate 0.05 -> after
SFT, 1.00 on held-out subjects. The cold-start fix works. The one real bug
hit along the way: the SFT token-length budget was first guessed from
character counts, which badly underestimates Qwen's actual BPE token count
on this dense numeric text — it silently truncated the training target off
of a large fraction of batches before being fixed to measure real
tokenized length instead (see `sft.py`'s `compute_max_length`).

## Running on a real GPU (AWS)

Everything above also runs unchanged on a CUDA GPU — `model_real.py`
auto-detects `cuda` and:
- uses bf16 instead of fp32 for the base model (standard LoRA recipe on a
  real GPU; fp32 was specifically needed to avoid a `nan`-loss dtype bug on
  the Mac's MPS backend, which doesn't apply on CUDA),
- scales `BATCH_SIZE` / `GROUP_SIZE` / `PROMPTS_PER_STEP` / `N_STEPS` up via
  `GPU_SCALE` (4x by default — edit the constant in `model_real.py` to
  change it) instead of staying tuned to laptop-safe values.

Setup on a fresh instance (a `g5.xlarge`/`g5.2xlarge`, A10G 24GB, is enough
for 0.6B-4B LoRA fine-tuning; use an AWS Deep Learning AMI to skip CUDA setup):

```
git clone <your repo URL>
cd capstone_pipeline
pip install -r requirements.txt
bash grpo/run_gpu.sh                     # runs all three steps, default model
bash grpo/run_gpu.sh Qwen/Qwen3-1.7B      # or scale up the model directly
```

`run_gpu.sh` just calls `zero_shot_check.py` -> `sft.py` -> `train_real.py`
in order (same as the manual commands above) and prints the detected GPU
first. The `GRPO_MODEL_NAME` env var it sets is read by `model_real.py` --
set it yourself if you call the python files directly instead of the script.
Stop the instance when you're not actively running a job.
