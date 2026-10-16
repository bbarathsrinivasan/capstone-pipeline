"""
Real pretrained model loading for the GRPO pipeline -- the replacement for
the from-scratch tokenizer_build.py/model.py stand-in used in the mechanics
validation pass (see results/findings.md, "Infrastructure constraint").

This needs real network access to huggingface.co. This sandbox's proxy
blocks it (confirmed: 403 on huggingface.co and download.pytorch.org), so
this file is meant to be RUN from a machine with normal internet access --
see grpo/README.md "Real-model track" for exact commands to run yourself.

Default model: Qwen3-0.6B -- the smallest member of the same model family
TimeSRL itself used (Qwen3-4B), chosen specifically so the scale-up path
(0.6B -> 1.7B -> 4B) is a one-line MODEL_NAME change, never a re-tune of
the reward function or LoRA config below (both already match what the
paper reports for the 4B model).
"""
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

# Device selection -- the FIRST real speed fix. Nothing in the original
# version of this file ever moved the model off CPU, which is almost
# certainly why a laptop run looked "crashing"-slow (tens of seconds per
# step): every forward/backward pass ran single-threaded CPU tensor ops.
# `accelerate`/`transformers` don't do this for you automatically outside
# a Trainer -- a manual loop (like sft.py's and train_real.py's) has to
# move the model and every batch tensor itself.
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# bf16 on CUDA: a real GPU's tensor cores support bf16 natively and this is
# the standard LoRA recipe (bf16 base + fp32 adapters, e.g. QLoRA-style) --
# it roughly halves memory and meaningfully speeds up both the Mac-sized
# and the scaled-up (1.7B/4B) runs. This is DELIBERATELY NOT used on
# CPU/MPS: forcing fp32 everywhere there was the actual fix for the
# `loss=nan` bug (see load_base_model below) -- bf16 is only safe here
# because CUDA's bf16 kernels are mature in a way MPS's aren't yet.
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Scale-up path, in order -- override via the GRPO_MODEL_NAME env var (see
# grpo/run_gpu.sh) rather than editing this file, e.g.:
#   GRPO_MODEL_NAME="Qwen/Qwen3-1.7B" python3 grpo/sft.py
MODEL_NAME = os.environ.get("GRPO_MODEL_NAME", "Qwen/Qwen3-0.6B")
SCALE_UP_PATH = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]

# A real GPU has far more memory and throughput than the Mac's MPS backend
# -- scale batch/group sizes up automatically rather than hand-tuning a
# separate config per machine. sft.py and train_real.py read this.
GPU_SCALE = 1 if DEVICE == "cuda" else 1

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"

# LoRA target modules for a Qwen3-style decoder's attention projections --
# the "freeze one layer, update the other" piece, same idea as model.py's
# GPT-2 c_attn target, just the real architecture's module names. Rank
# 16 / alpha 32 matches the LoRA setting TimeSRL reports for Qwen3-4B.
LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


def free_memory():
    """MPS-specific: PyTorch's MPS caching allocator doesn't release memory
    back to the OS between calls the way CUDA's does. Repeated
    policy.generate() calls in particular (zero_shot_check.py,
    sft.py's eval_parse_rate, train_real.py's rollout sampling) can ratchet
    the reserved pool up until it hits the device's watermark ceiling --
    this is what a 'MPS backend out of memory ... max allowed: X GiB' crash
    actually means on a 0.6B model that has no business needing that much
    memory. Call this after every generate() call, and periodically during
    training. No-op on CUDA/CPU."""
    import gc
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()


def load_tokenizer(model_name=MODEL_NAME):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(model_name=MODEL_NAME):
    from transformers import AutoModelForCausalLM

    # Request DTYPE explicitly, two ways: the loader kwarg (new transformers
    # versions renamed `torch_dtype` to `dtype` -- try both so this works
    # either way). On CPU/MPS, DTYPE is fp32 and we ALSO call `.float()`
    # afterward as a belt-and-suspenders guarantee -- this is the actual
    # fix for the `loss=nan` bug hit on the Mac: Qwen3 checkpoints ship in
    # bfloat16, and if the dtype request silently didn't take (exactly what
    # the "torch_dtype is deprecated" warning was flagging), the base model
    # stayed in bf16 while PEFT's LoRA A/B matrices were initialized in
    # fp32 -- that mismatch, run through MPS's/CPU's less mature bf16
    # support, is the leading suspect for the nan (see sft.py's NaN guard
    # for the runtime check). On CUDA, DTYPE is bf16 on purpose (see the
    # module-level comment) and we do NOT force `.float()` afterward, since
    # that would just undo the memory/speed win -- bf16-base + fp32-adapter
    # is the standard, well-supported LoRA setup on real GPU kernels.
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=DTYPE)
    if DEVICE != "cuda":
        model = model.float()
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {model_name}: {n_params / 1e6:.1f}M parameters (pretrained weights, "
          f"dtype={DTYPE}, device={DEVICE}).")
    return model


def build_policy_and_reference(model_name=MODEL_NAME, sft_adapter_path=None):
    """Mirrors model.py's build_policy_and_reference, but on a real
    pretrained model.

    If sft_adapter_path is given, the policy starts from the SFT-trained
    LoRA adapter (see sft.py) instead of a fresh LoRA init -- this is the
    fix for the cold-start problem documented in results/findings.md: a
    policy with no adapter training has near-zero chance of emitting a
    parseable `Risk score: N` line, so GRPO never sees a nonzero reward to
    learn from. Starting GRPO from the SFT checkpoint instead means most
    rollouts already clear the format bar, so GRPO has something to refine.

    Reference model (for the KL penalty) is the pretrained base, UNADAPTED
    -- not the SFT checkpoint. This is a deliberate choice, not the only
    valid one: using the plain pretrained base as the KL anchor keeps the
    policy from drifting arbitrarily far from "a normal language model",
    while anchoring to the SFT checkpoint instead would keep it closer to
    "the format-tuned model" specifically. Worth revisiting once a real run
    is in hand -- flag it the same way as the safety-shaped reward penalty.

    NOTE on how the reference is obtained: there is no separate reference
    model object anymore. An earlier version built one with
    `copy.deepcopy(base)`, which keeps a full second copy of the base
    model's weights resident in GPU memory for the entire run -- on a 4B
    model in bf16 that's ~8GB permanently spent on something only needed
    for one KL term per step. train_real.py instead computes reference
    logprobs on THIS SAME policy object with its LoRA adapter temporarily
    disabled (`with policy.disable_adapter(): ...`), which is mathematically
    identical (disabling the adapter makes the model behave exactly like
    the unmodified pretrained base) without the duplicate memory. This is
    what let GRPO fit in the budget SFT already proved fits.
    """
    base = load_base_model(model_name)

    if sft_adapter_path:
        from peft import PeftModel

        policy = PeftModel.from_pretrained(base, sft_adapter_path, is_trainable=True).to(DEVICE)
        print(f"Policy initialized from SFT checkpoint: {sft_adapter_path}")
    else:
        policy = get_peft_model(base, LORA_CONFIG).to(DEVICE)
        print("Policy initialized from a FRESH LoRA adapter (no SFT) -- "
              "expect the cold-start problem from results/findings.md to repeat, "
              "just less severely than the from-scratch model (this one at "
              "least knows English).")

    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    print(f"LoRA policy: {trainable:,} trainable / {total:,} total "
          f"({100 * trainable / total:.2f}%) -- base weights frozen, only "
          f"rank-16 adapters update.")
    return policy


if __name__ == "__main__":
    tok = load_tokenizer()
    policy = build_policy_and_reference()
