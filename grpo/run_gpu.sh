#!/usr/bin/env bash
# Convenience runner for the real-model GRPO track on a fresh AWS GPU instance.
# Runs the full sequence in order: zero-shot check -> SFT (cold-start fix) ->
# GRPO -> before/after eval -> plots. Each step reads/writes grpo/results/
# and grpo/checkpoints/, so it's safe to re-run individual steps by calling
# the python files directly instead of this script once you're past the
# first full pass.
#
# Usage:
#   bash grpo/run_gpu.sh                      # default model, now Qwen3-4B --
#                                              # every validated result so far is on this size;
#                                              # don't drop to 0.6B/1.7B except for a quick smoke
#                                              # test of new CODE (SFT reliably fails to even teach
#                                              # the output format on 0.6B -- not a meaningful
#                                              # accuracy signal, see model_real.py)
#   bash grpo/run_gpu.sh Qwen/Qwen3-0.6B       # smoke-test override only -- see above
#   GRPO_N_STEPS=80 bash grpo/run_gpu.sh       # override GRPO step count (default 40)
#   GRPO_GROUP_SIZE=8 bash grpo/run_gpu.sh     # override rollout group size
#   GRPO_NOISE_RATE=0.0 bash grpo/run_gpu.sh   # SFT label-noise ablation
#
# Expects to be run from the repo root (or anywhere -- it cds to the repo
# root itself) on a machine with: a CUDA GPU, Python 3.10+, and
# `pip install -r requirements.txt` already run (see README.md).
#
# W&B logging: run `wandb login` once per machine (or export WANDB_API_KEY)
# before this script, so sft.py/train_real.py/evaluate_before_after.py log
# runs to the "Capstone Implementation" project under the
# bbasavar-carnegie-mellon-university entity (override with WANDB_PROJECT /
# WANDB_ENTITY). Not logged in and don't want to be? Set WANDB_DISABLED=1
# and these scripts skip W&B entirely and just print to stdout/results/ as
# before.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "${1:-}" != "" ]; then
    export GRPO_MODEL_NAME="$1"
fi

# One shared group id for this entire pipeline run, so the sft/grpo/eval
# runs below show up grouped together in the W&B UI instead of as three
# unrelated runs -- set WANDB_RUN_GROUP yourself before calling this script
# to reuse a specific group name instead of a fresh timestamp each time.
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-run-$(date +%Y%m%d-%H%M%S)}"
echo "W&B run group: $WANDB_RUN_GROUP"

python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('Memory (GB):', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
else:
    print('WARNING: no CUDA GPU detected -- this will run on CPU and be very slow. '
          'Check your instance type and that the CUDA build of torch installed correctly.')
"

echo
echo "== 1/5: zero-shot format check =="
python3 grpo/zero_shot_check.py

echo
echo "== 2/5: SFT (cold-start fix) =="
python3 grpo/sft.py

echo
echo "== 3/5: GRPO on the real model =="
python3 grpo/train_real.py

echo
echo "== 4/5: before/after evaluation (held-out subjects) =="
python3 grpo/evaluate_before_after.py

echo
echo "== 5/5: plots =="
python3 grpo/make_plots_real.py

echo
echo "Done. Results in grpo/results/, checkpoints in grpo/checkpoints/{sft_lora,grpo_lora}."
