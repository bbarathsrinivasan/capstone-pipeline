#!/usr/bin/env bash
# Convenience runner for the real-model GRPO track on a fresh AWS GPU instance.
# Runs the full sequence in order: zero-shot check -> SFT (cold-start fix) ->
# GRPO -> before/after eval -> plots. Each step reads/writes grpo/results/
# and grpo/checkpoints/, so it's safe to re-run individual steps by calling
# the python files directly instead of this script once you're past the
# first full pass.
#
# Usage:
#   bash grpo/run_gpu.sh                      # default model (Qwen3-0.6B)
#   bash grpo/run_gpu.sh Qwen/Qwen3-1.7B       # scale-up: override the model
#   GRPO_N_STEPS=80 bash grpo/run_gpu.sh       # override GRPO step count (default 40)
#   GRPO_GROUP_SIZE=8 bash grpo/run_gpu.sh     # override rollout group size
#   GRPO_NOISE_RATE=0.0 bash grpo/run_gpu.sh   # SFT label-noise ablation
#
# Expects to be run from the repo root (or anywhere -- it cds to the repo
# root itself) on a machine with: a CUDA GPU, Python 3.10+, and
# `pip install -r requirements.txt` already run (see README.md).
#
# W&B logging: run `wandb login` once per machine (or export WANDB_API_KEY)
# before this script, so sft.py/train_real.py/evaluate_before_after.py can
# log runs to the "capstone-grpo" project (override with WANDB_PROJECT).
# Not logged in and don't want to be? Set WANDB_DISABLED=1 and these scripts
# skip W&B entirely and just print to stdout/results/ as before.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "${1:-}" != "" ]; then
    export GRPO_MODEL_NAME="$1"
fi

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
