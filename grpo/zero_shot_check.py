"""
Before spending any time on SFT: how much of the cold-start problem does a
REAL pretrained instruct model already solve, with zero training?

results/findings.md's main run used a randomly-initialized, from-scratch
model (0% parse rate -- it had never seen English, let alone the target
format). A real small model has presumably already learned to follow a
"respond in exactly this format" instruction from its own instruction
tuning. This script measures that directly, on this project's actual
prompts, before deciding whether SFT is even necessary or how much of a
gap it needs to close.

Run this FIRST, before sft.py. Needs real network access to huggingface.co
-- see README.md "Real-model track" for where to run this.
"""
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_real import MODEL_NAME, DEVICE, load_tokenizer, load_base_model, free_memory
from prompts import build_examples
from reward import parse_predicted_score, reward_batch

N_SAMPLES = 20
MAX_NEW_TOKENS = 200  # real instruct models reason more verbosely than the tiny stand-in did
SEED = 0


@torch.no_grad()
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    tokenizer = load_tokenizer()
    model = load_base_model()
    model.eval()

    examples = build_examples()
    sample = random.sample(examples, min(N_SAMPLES, len(examples)))

    # Measure the real tokenized prompt length instead of guessing -- same
    # fix as sft.py/train_real.py after the MAX_LENGTH=640 bug in sft.py
    # turned out to silently truncate completions off of most SFT batches.
    # Here it would silently truncate the END of the clinical feature block
    # itself, with no warning, biasing every generation below.
    lens = [len(tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]) for ex in sample]
    prompt_max_length = max(lens) + 20
    print(f"Measured prompt token length: min={min(lens)} max={max(lens)} mean={sum(lens) / len(lens):.0f} "
          f"-> prompt truncation max_length set to {prompt_max_length}.")

    n_parsed = 0
    rows = []
    for ex in sample:
        inputs = tokenizer(ex["prompt"], return_tensors="pt", truncation=True, max_length=prompt_max_length)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = parse_predicted_score(text)
        reward = reward_batch([text], [ex["risk_int"]])[0][0]
        if parsed is not None:
            n_parsed += 1
        rows.append({
            "sample_id": ex["sample_id"], "true_risk": ex["risk_category"],
            "parsed_score": parsed, "reward": reward, "generation_preview": text[:200],
        })
        print(f"[{ex['sample_id']}] true={ex['risk_category']:<8} parsed={parsed} reward={reward:.3f}")
        del out, inputs
        free_memory()

    parse_rate = n_parsed / len(sample)
    print(f"\n{MODEL_NAME} zero-shot parse-success rate: {parse_rate:.2f} "
          f"({n_parsed}/{len(sample)}), vs. 0.00 for the from-scratch model in the main run.")
    if parse_rate < 0.5:
        print("Still well below reliable -- SFT (sft.py) is worth doing before GRPO.")
    else:
        print("Already fairly reliable zero-shot -- SFT may only need to be short, "
              "or GRPO might even be tried directly first as a comparison point.")
    return rows


if __name__ == "__main__":
    main()
