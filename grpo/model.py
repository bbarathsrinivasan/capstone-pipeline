"""
A small, randomly-initialized GPT-2-architecture causal LM + the tier-1
GLOBEM byte-level BPE tokenizer from tokenizer_build.py.

Freeze-one-layer / update-other-layer, concretely: LoRA (Hu et al. 2021,
arXiv:2106.09685 -- in the team's own Sep 20 reference list) wraps the
attention projections, freezes every pretrained-equivalent base weight, and
leaves only the small rank-16 adapter matrices trainable. Rank 16 / alpha 32
matches the LoRA setting reported in the TimeSRL paper.
"""
from pathlib import Path

from peft import LoraConfig, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent
TOKENIZER_DIR = ROOT / "tokenizer"

# Small enough to train on 2 CPU cores in a few minutes: ~6M base params.
MODEL_CONFIG_KWARGS = dict(
    n_layer=4,
    n_head=4,
    n_embd=128,
    n_positions=768,
    n_ctx=768,
)

LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn"],  # GPT-2's fused qkv projection
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


def load_tokenizer():
    # PreTrainedTokenizerFast needs a tokenizer.json, not raw vocab/merges --
    # build one directly from the trained ByteLevelBPETokenizer.
    from tokenizers import ByteLevelBPETokenizer

    raw = ByteLevelBPETokenizer(
        str(TOKENIZER_DIR / "vocab.json"),
        str(TOKENIZER_DIR / "merges.txt"),
    )
    raw.save(str(TOKENIZER_DIR / "tokenizer.json"))
    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(TOKENIZER_DIR / "tokenizer.json"),
        pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", unk_token="<unk>",
    )
    return tok


def build_base_model(vocab_size, pad_token_id=0, bos_token_id=1, eos_token_id=2):
    config = GPT2Config(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id,
        **MODEL_CONFIG_KWARGS,
    )
    model = GPT2LMHeadModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Base model: {n_params / 1e6:.1f}M parameters (randomly initialized -- "
          f"see tokenizer_build.py docstring for why no pretrained weights).")
    return model


def build_policy_and_reference(tokenizer):
    """Returns (policy, reference). `policy` is LoRA-wrapped with the base
    frozen and only adapters trainable; `reference` is a separate, fully
    frozen copy of the SAME initial weights, used for the GRPO KL penalty
    so the policy can't drift arbitrarily far from its starting point."""
    import copy

    base = build_base_model(
        tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    reference = copy.deepcopy(base)
    for p in reference.parameters():
        p.requires_grad = False

    policy = get_peft_model(base, LORA_CONFIG)
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    print(f"LoRA policy: {trainable:,} trainable params / {total:,} total "
          f"({100 * trainable / total:.2f}%) -- base weights frozen, only "
          f"rank-16 adapters update.")
    return policy, reference


if __name__ == "__main__":
    tok = load_tokenizer()
    print(f"Tokenizer vocab size: {tok.vocab_size}")
    policy, reference = build_policy_and_reference(tok)
