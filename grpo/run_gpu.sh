#!/usr/bin/env bash
# Convenience runner for the real-model GRPO track on a fresh AWS GPU instance.
# Runs the full sequence in order: zero-shot check -> SFT (cold-start fix) ->
# GRPO. Each step reads/writes grpo/results/ and grpo/checkpoints/, so it's
# safe to re-run individual steps by calling the python files directly
# instead of this script once you're past the first full pass.
#
# Usage:
#   bash grpo/run_gpu.sh                      # default model (Qwen3-0.6B)
#   bash grpo/run_gpu.sh Qwen/Qwen3-1.7B       # scale-up: override the model
#
# Expects to be run from the repo root (or anywhere -- it cds to the repo
# root itself) on a machine with: a CUDA GPU, Python 3.10+, and
# `pip install -r requirements.txt` already run (see README.md).
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
echo "== 1/3: zero-shot format check =="
python3 grpo/zero_shot_check.py

echo
echo "== 2/3: SFT (cold-start fix) =="
python3 grpo/sft.py

echo
echo "== 3/3: GRPO on the real model =="
python3 grpo/train_real.py

echo
echo "Done. Results in grpo/results/, checkpoint in grpo/checkpoints/sft_lora."
