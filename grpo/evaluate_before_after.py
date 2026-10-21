"""
Before/after evaluation: did GRPO actually improve classification accuracy,
on subjects NEITHER training stage ever saw?

"Before" loads the base model with only the SFT LoRA adapter (checkpoints/
sft_lora). "After" loads it with the GRPO-trained adapter instead
(checkpoints/grpo_lora, saved by train_real.py's main() after this run's
fixes). Both are evaluated, with plain greedy decoding (no sampling noise,
unlike GRPO's own stochastic rollouts -- this is meant to be a clean,
reproducible comparison of the two checkpoints themselves), on the exact
same held-out val-subject set sft.py's split produces: prompts.py's
split_examples_by_subject with the same VAL_FRACTION/SEED train_real.py
uses for its own training pool. Neither sft.py nor train_real.py ever
trains on these subjects -- see both files' main() for where they're
excluded.

reward_mean here is NOT directly comparable to train_real.py's training-time
reward_mean: that one is averaged over temperature=1.0 (or 0.7, depending on
which run) sampled rollouts; this one is a single greedy decode per example.

Run after both sft.py and train_real.py have produced their checkpoints:
    python3 grpo/evaluate_before_after.py
-> results/before_after_eval.json
"""
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_real import CHECKPOINT_DIR, DEVICE, load_tokenizer, load_base_model, free_memory
from prompts import build_examples, split_examples_by_subject
from reward import parse_predicted_score, safety_shaped_reward

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 0
VAL_FRACTION = 0.15  # must match sft.py / train_real.py
MAX_NEW_TOKENS = 160


def compute_prompt_max_length(tokenizer, examples, margin=20):
    lens = [len(tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]) for ex in examples]
    return max(lens) + margin


@torch.no_grad()
def evaluate(policy, tokenizer, examples, prompt_max_length, label):
    """Greedy-decode every held-out example once, score it the same way
    reward.py scores a GRPO rollout, and report both parse rate and actual
    classification accuracy -- the thing the whole pipeline is meant to
    improve, which training-time reward_mean alone doesn't directly show."""
    policy.eval()
    n = len(examples)
    n_parsed = 0
    n_exact = 0
    abs_errors = []
    rewards = []
    samples = []
    for ex in examples:
        inputs = tokenizer(ex["prompt"], return_tensors="pt", truncation=True, max_length=prompt_max_length)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        out = policy.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        p = parse_predicted_score(text)
        y = ex["risk_int"]
        r = safety_shaped_reward(p, y)
        rewards.append(r)
        if p is not None:
            n_parsed += 1
            abs_errors.append(abs(p - y))
            if round(p) == y:
                n_exact += 1
        samples.append({
            "sample_id": ex["sample_id"], "true_risk": ex["risk_category"],
            "generation": text, "parsed_score": p, "reward": r,
        })
        del out, inputs
        free_memory()

    result = {
        "label": label,
        "n": n,
        "parse_rate": n_parsed / n,
        # Two accuracy numbers on purpose: "_of_all" penalizes a format
        # failure as wrong (the fair end-to-end number); "_of_parsed" isolates
        # classification skill from format reliability, in case the two
        # checkpoints differ a lot on one but not the other.
        "exact_match_accuracy_of_all": n_exact / n,
        "exact_match_accuracy_of_parsed": (n_exact / n_parsed) if n_parsed else None,
        "mean_abs_error_of_parsed": (sum(abs_errors) / len(abs_errors)) if abs_errors else None,
        "mean_reward": sum(rewards) / n,
        "samples": samples,
    }
    print(f"[{label}] n={n} parse_rate={result['parse_rate']:.2f} "
          f"exact_match_acc(of all)={result['exact_match_accuracy_of_all']:.2f} "
          f"mean_abs_error(parsed)={result['mean_abs_error_of_parsed']} "
          f"mean_reward={result['mean_reward']:.3f}")
    return result


def main():
    tokenizer = load_tokenizer()
    examples = build_examples()
    _, val_examples = split_examples_by_subject(examples, VAL_FRACTION, SEED)
    prompt_max_length = compute_prompt_max_length(tokenizer, examples)
    print(f"Held-out eval set: {len(val_examples)} examples from "
          f"{len({e['subject_id'] for e in val_examples})} subjects -- the same "
          f"split sft.py and train_real.py both exclude from training.")

    sft_path = CHECKPOINT_DIR / "sft_lora"
    grpo_path = CHECKPOINT_DIR / "grpo_lora"
    if not sft_path.exists():
        sys.exit(f"Missing {sft_path} -- run sft.py first.")
    if not grpo_path.exists():
        sys.exit(f"Missing {grpo_path} -- run train_real.py first (this run's "
                  f"version saves it; older results dirs won't have it).")

    print("\n--- BEFORE (SFT only, no GRPO) ---")
    base = load_base_model()
    policy = PeftModel.from_pretrained(base, str(sft_path)).to(DEVICE)
    before = evaluate(policy, tokenizer, val_examples, prompt_max_length, "before_sft_only")
    del policy, base
    free_memory()

    print("\n--- AFTER (SFT + GRPO) ---")
    base2 = load_base_model()
    policy2 = PeftModel.from_pretrained(base2, str(grpo_path)).to(DEVICE)
    after = evaluate(policy2, tokenizer, val_examples, prompt_max_length, "after_grpo")

    out = {"before": before, "after": after}
    with open(RESULTS / "before_after_eval.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {RESULTS / 'before_after_eval.json'}")
    print(
        f"\nSummary: exact-match accuracy {before['exact_match_accuracy_of_all']:.2f} -> "
        f"{after['exact_match_accuracy_of_all']:.2f}, mean reward "
        f"{before['mean_reward']:.3f} -> {after['mean_reward']:.3f}"
    )


if __name__ == "__main__":
    main()
