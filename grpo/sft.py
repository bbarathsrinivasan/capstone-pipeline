"""
Format-SFT ("cold-start fix") for the GRPO pipeline.

Why this exists -- the cold-start problem, concretely:
GRPO only produces a learning signal when at least SOME rollouts in a group
earn a nonzero reward. In the main run (results/grpo_run.json), every single
rollout scored reward 0 because the policy never once produced a parseable
`Risk score: N` line -- so the group-relative advantage, (reward -
group_mean) / group_std, was (0-0)/(0+eps) = 0 for every sample, every step.
There was nothing to push the policy toward or away from: no gradient
signal, no learning, regardless of how many steps were run. That's the cold
start problem -- it's not that GRPO's math is broken (the sanity check in
sanity_check.py proves the update code works given ANY learnable signal),
it's that a policy has to already clear a minimum "produces something
rewardable" bar before GRPO has anything to refine.

The fix: a short supervised warm-up pass BEFORE GRPO, teaching the policy
the output FORMAT (and a reasonable summary register) via ordinary
next-token prediction on (prompt -> correctly-formatted completion) pairs.
GRPO then takes over to refine ACCURACY via the reward signal, starting
from a policy that already clears the format bar most of the time.

Design choice, flagged for review (same spirit as reward.py's flagged
0.15x penalty -- this is this project's own construction, not from the
paper): SFT targets are correctly FORMATTED but only (1 - NOISE_RATE)
label-accurate. If every SFT example ended in the exactly correct score,
the model could learn a shortcut from surface features straight to "the
right answer" well enough that GRPO would have no remaining errors to
correct -- which defeats the point of doing RL afterward. NOISE_RATE is a
knob, not a settled value; revisit after the first real GRPO run shows
whether GRPO is doing any real work on top of SFT or just inheriting it.

Run this on a machine with real network access to huggingface.co (not this
sandbox -- see README.md "Real-model track"). Run zero_shot_check.py first
to see how big a gap this actually needs to close for your chosen model.
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_real import MODEL_NAME, CHECKPOINT_DIR, DEVICE, GPU_SCALE, load_tokenizer, load_base_model, LORA_CONFIG, free_memory
from prompts import build_examples, split_examples_by_subject
from reward import parse_predicted_score
from peft import get_peft_model

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 0
NOISE_RATE = 0.2          # fraction of SFT targets with a deliberately off-by-one stated score
VAL_FRACTION = 0.15       # held out BY SUBJECT, same discipline as the baseline pipeline's grouped CV
N_EPOCHS = 3
BATCH_SIZE = 2 * GPU_SCALE  # 2 on the Mac (MPS OOM'd above this -- attention memory scales with
                            # batch_size * seq_length^2, and gradient checkpointing below was the
                            # other lever); GPU_SCALE=4 on a real CUDA GPU gives 8, which a single
                            # 24GB GPU (e.g. g5.xlarge's A10G) handles comfortably for 0.6B-4B LoRA.
                            # Raise GPU_SCALE in model_real.py if you have more headroom to spend.
LEARNING_RATE = 1e-4
COMPLETION_TOKEN_BUDGET = 150  # headroom reserved for the completion on top of the real
                               # measured prompt length (see compute_max_length) -- our
                               # completions run well under 100 tokens, so this is generous
# NOTE: there used to be a fixed `MAX_LENGTH = 640` constant here, guessed from CHARACTER
# counts (prompts.py's char-length stats: min=1407 max=1452 across all 480 prompts). That
# guess was badly wrong for Qwen's actual BPE tokenizer -- this text is dense with numbers
# and punctuation ("D1: null, D2: 4.5, ..."), which BPE tokenizes far less efficiently than
# prose (closer to 1.5-2 chars/token than the ~3.3 assumed). Because all 480 prompts are
# nearly the same character length, once the real token length crossed 640, essentially ALL
# of them did -- not a few long outliers, nearly the whole dataset. That's why a large
# fraction of training batches were silently skipped (see the SKIPPED guard in main()) even
# though enough real batches got through to still train the model successfully. It's now
# replaced by compute_max_length() below, called once in main() and passed explicitly
# everywhere a length is needed -- no fixed guess, no module-level constant to go stale.
EVAL_N = 20               # trimmed from 30 on the Mac -- fewer free-generation calls before training
                          # starts, since those were the first thing to balloon MPS's reserved memory
                          # pool. Fine to raise back up on a GPU where this isn't a concern.


def compute_max_length(tokenizer, examples, buffer=COMPLETION_TOKEN_BUDGET):
    """Measure the REAL tokenized prompt length with the model's actual
    tokenizer, instead of guessing from character counts -- exactly the
    mistake that caused nearly every SFT batch to get silently skipped.
    Tokenization alone is cheap (no model forward pass), so this runs in
    well under a second for 480 prompts."""
    lens = [len(tokenizer(ex["prompt"] + "\n\n", add_special_tokens=False)["input_ids"])
            for ex in examples]
    max_len = max(lens)
    computed = max_len + buffer
    print(f"Measured prompt token length: min={min(lens)} max={max_len} mean={sum(lens) / len(lens):.0f} "
          f"-> MAX_LENGTH set to {computed} (max prompt + {buffer}-token completion budget).")
    return computed

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# A few interchangeable phrasings per risk level, so SFT teaches the FORMAT
# and a plausible clinical-summary register, not one memorized string.
SUMMARY_TEMPLATES = {
    0: [
        "Sleep duration and efficiency stay within a typical range across the week, with home time and screen use showing no unusual pattern.",
        "This week shows consistent sleep timing, normal mobility (time spent away from home), and no notable spike in nighttime phone use.",
    ],
    1: [
        "Sleep is slightly shorter or more irregular than typical on a couple of days, with a modest increase in nighttime screen time.",
        "A few nights show reduced sleep efficiency and slightly later bedtimes, alongside a small uptick in phone use before sleep.",
    ],
    2: [
        "Several days show shortened sleep duration and lower sleep efficiency, paired with reduced time spent away from home and elevated nighttime screen use.",
        "The week shows a recurring pattern of late bedtimes, fragmented sleep (high wake-after-sleep-onset), and increased isolation at home.",
    ],
    3: [
        "Sleep is substantially disrupted across most of the week -- short duration, poor efficiency, and very late bedtimes -- alongside minimal time away from home and heavy nighttime phone use.",
        "Multiple days show severely reduced sleep, high wake-after-sleep-onset, little time outside the home, and a marked increase in screen use overnight -- a pattern consistent with significant withdrawal.",
    ],
}


def build_sft_target(true_risk_int, rng):
    stated_risk = true_risk_int
    if rng.random() < NOISE_RATE:
        delta = rng.choice([-1, 1])
        stated_risk = min(3, max(0, true_risk_int + delta))
    # Bug fix: this used to be SUMMARY_TEMPLATES[true_risk_int] -- the
    # summary described the TRUE severity while the stated score (on the
    # NOISE_RATE fraction of rows) was independently shifted off of it. That
    # taught the model, by example, that the final score doesn't have to
    # agree with the summary it just wrote -- the opposite of what you'd
    # want if the score is supposed to be DECIDED FROM the summary rather
    # than independently re-read off the raw numbers. Keying the template to
    # stated_risk instead makes every SFT target internally consistent: the
    # summary always matches whatever score actually gets stated, noised or
    # not. (The label noise relative to the TRUE risk is still there and
    # still intentional -- see NOISE_RATE's docstring above -- this only
    # fixes which text the stated score is paired with.)
    summary = rng.choice(SUMMARY_TEMPLATES[stated_risk])
    return f"{summary}\n\nRisk score: {stated_risk}"


def build_sft_examples():
    """Subject-grouped split via prompts.py's split_examples_by_subject,
    shared with train_real.py (GRPO's training pool) and
    evaluate_before_after.py (the before/after holdout) so all three agree
    on exactly the same held-out subjects -- SFT-val, GRPO's training data,
    and the before/after eval set are never the same subjects by accident."""
    examples = build_examples()
    train_examples, val_examples = split_examples_by_subject(examples, VAL_FRACTION, SEED)
    rng = random.Random(SEED)

    def to_rows(exs):
        return [
            {"subject_id": e["subject_id"], "prompt": e["prompt"],
             "completion": build_sft_target(e["risk_int"], rng),
             "risk_int": e["risk_int"], "risk_category": e["risk_category"]}
            for e in exs
        ]

    train_rows, val_rows = to_rows(train_examples), to_rows(val_examples)
    n_val_subjects = len({e["subject_id"] for e in val_examples})
    n_subjects = len({e["subject_id"] for e in examples})
    print(f"SFT dataset: {len(train_rows)} train / {len(val_rows)} val rows "
          f"({n_val_subjects}/{n_subjects} subjects held out of train).")
    return train_rows, val_rows


def tokenize_row(tokenizer, row, max_length):
    """Prompt tokens are masked out of the loss (-100) -- SFT should only be
    graded on producing the completion, not on 'predicting' the prompt it
    was handed.

    max_length has NO default on purpose: Python binds a default argument's
    value once, at function-DEFINITION time, not at call time. An earlier
    version of this function (`max_length=MAX_LENGTH`, with a module-level
    `MAX_LENGTH = 640`) looked like it would pick up a later
    `MAX_LENGTH = compute_max_length(...)` reassignment in main() -- it
    would not have; every call still would have used the module-load-time
    value of 640 regardless. Requiring callers to pass max_length
    explicitly closes that trap instead of relying on global mutation."""
    prompt_ids = tokenizer(row["prompt"] + "\n\n", add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    if eos is not None:
        completion_ids = completion_ids + [eos]
    input_ids = (prompt_ids + completion_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + completion_ids)[:max_length]
    if all(l == -100 for l in labels):
        # The prompt alone ate the whole max_length budget, truncating the
        # ENTIRE completion away -- every label in this row is masked, so
        # it contributes nothing to the loss. One row like this in a batch
        # is harmless (ignored); see the all-masked-BATCH guard in main()
        # for why a whole batch of these is what actually produces nan.
        print(f"WARNING: prompt alone reached max_length={max_length} tokens for "
              f"sample -- completion fully truncated. Raise MAX_LENGTH.")
    return input_ids, labels


def make_batch(tokenizer, rows, pad_id, max_length):
    tokenized = [tokenize_row(tokenizer, r, max_length) for r in rows]
    max_len = max(len(ids) for ids, _ in tokenized)
    batch_ids, batch_labels, batch_attn = [], [], []
    for ids, labels in tokenized:
        pad_n = max_len - len(ids)
        batch_ids.append(ids + [pad_id] * pad_n)
        batch_labels.append(labels + [-100] * pad_n)
        batch_attn.append([1] * len(ids) + [0] * pad_n)
    ids_t = torch.tensor(batch_ids, device=DEVICE)
    labels_t = torch.tensor(batch_labels, device=DEVICE)
    attn_t = torch.tensor(batch_attn, device=DEVICE)
    return ids_t, labels_t, attn_t


@torch.no_grad()
def eval_parse_rate(policy, tokenizer, rows, prompt_max_length, n=EVAL_N, max_new_tokens=160):
    """The real readiness gate before resuming GRPO: how often does the
    model's own FREE-RUNNING generation (not the fed target) produce a
    parseable `Risk score: N` line? This is what GRPO needs to be nonzero,
    not the training loss.

    prompt_max_length is the same real-tokenizer-measured value used for
    training (minus the completion budget doesn't matter here -- there's no
    completion to fit, just the prompt -- but it must still be >= the real
    max prompt length, or this truncates the prompt itself and the model
    is evaluated on a chopped-off input)."""
    policy.eval()
    sample = random.sample(rows, min(n, len(rows)))
    n_parsed = 0
    for row in sample:
        inputs = tokenizer(row["prompt"], return_tensors="pt", truncation=True, max_length=prompt_max_length)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        out = policy.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if parse_predicted_score(text) is not None:
            n_parsed += 1
        del out, inputs
        free_memory()
    return n_parsed / len(sample)


def main():
    tokenizer = load_tokenizer()
    base = load_base_model()
    policy = get_peft_model(base, LORA_CONFIG).to(DEVICE)

    # Gradient checkpointing: the real fix for the OOM, not just a
    # mitigation. Without it, autograd retains every layer's intermediate
    # activations (attention matrices included) for the whole forward pass
    # so backward can use them -- that's what actually consumed the ~18GB,
    # not leftover cache from eval. With it, activations are recomputed
    # during backward instead of stored, trading some compute time for a
    # large, roughly linear-in-depth memory reduction. `enable_input_require_grads`
    # is needed alongside it specifically because the base model is frozen
    # (LoRA-only training) -- without it, checkpointing can silently break
    # the gradient path into the LoRA adapters.
    policy.gradient_checkpointing_enable()
    policy.enable_input_require_grads()

    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    print(f"SFT policy: {trainable:,}/{total:,} trainable ({100 * trainable / total:.2f}%), "
          f"device={DEVICE}, gradient_checkpointing=on.")

    train_rows, val_rows = build_sft_examples()

    # The actual fix: measure the real tokenized prompt length with THIS
    # tokenizer on THESE prompts, instead of the character-count guess that
    # made MAX_LENGTH=640 and silently truncated the completion off of
    # nearly every batch (see the module-level comment above and the
    # run below -- epoch 3 alone logged dozens of SKIPPED batches). Passed
    # explicitly everywhere below rather than reassigning the module-level
    # MAX_LENGTH, so there's no stale-default-argument trap left anywhere.
    max_length = compute_max_length(tokenizer, train_rows + val_rows)

    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=LEARNING_RATE)

    parse_rate_before = eval_parse_rate(policy, tokenizer, val_rows, max_length)
    print(f"\nParse-success rate BEFORE SFT (val, greedy decode): {parse_rate_before:.2f}")
    free_memory()  # clear the reserved pool the generation calls above built up, before training starts

    history = []
    t0 = time.time()
    n_steps_per_epoch = max(1, len(train_rows) // BATCH_SIZE)
    for epoch in range(N_EPOCHS):
        random.shuffle(train_rows)
        for step in range(n_steps_per_epoch):
            batch_rows = train_rows[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
            if not batch_rows:
                continue
            input_ids, labels, attn = make_batch(tokenizer, batch_rows, tokenizer.pad_token_id, max_length)

            # Guard against the exact failure mode seen in the first real
            # run: if EVERY row in this batch got its completion fully
            # truncated away (see tokenize_row's warning above), every
            # label is -100 and the batch-mean cross-entropy is 0/0 = nan.
            # Skip rather than silently training on a nan loss for hours.
            # With max_length now measured from the real tokenizer instead
            # of a character-count guess, this should fire rarely to never
            # -- if it still fires a lot, something else (e.g. a handful of
            # genuinely longer prompts) is going on and is worth a second look.
            if (labels != -100).sum().item() == 0:
                print(f"[epoch {epoch + 1} step {step}] SKIPPED: all labels masked "
                      f"in this batch (every prompt hit max_length={max_length}).")
                continue

            policy.train()
            out = policy(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss

            if not torch.isfinite(loss):
                print(f"[epoch {epoch + 1} step {step}] loss is {loss.item()} (non-finite) -- "
                      f"skipping this update rather than corrupting the weights with a nan "
                      f"gradient. If this keeps happening, see model_real.py's load_base_model "
                      f"docstring for the dtype-mismatch hypothesis, and consider "
                      f"attn_implementation='eager' or a lower LEARNING_RATE.")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            history.append({"epoch": epoch, "step": step, "loss": loss.item()})
            if step % 10 == 0:
                print(f"[epoch {epoch + 1}/{N_EPOCHS} step {step}/{n_steps_per_epoch}] "
                      f"loss={loss.item():.4f} ({time.time() - t0:.0f}s elapsed)")
            if step % 20 == 0:
                free_memory()

    parse_rate_after = eval_parse_rate(policy, tokenizer, val_rows, max_length)
    print(f"\nParse-success rate AFTER SFT (val, greedy decode): {parse_rate_after:.2f} "
          f"(was {parse_rate_before:.2f})")
    if parse_rate_after < 0.7:
        print("Still below a reliable floor -- consider more epochs, more SFT examples, "
              "or a less aggressive NOISE_RATE before moving on to train_real.py.")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    save_path = CHECKPOINT_DIR / "sft_lora"
    policy.save_pretrained(str(save_path))
    print(f"Saved SFT LoRA adapter to {save_path}")

    with open(RESULTS / "sft_run.json", "w") as f:
        json.dump({
            "config": dict(MODEL_NAME=MODEL_NAME, NOISE_RATE=NOISE_RATE, N_EPOCHS=N_EPOCHS,
                            BATCH_SIZE=BATCH_SIZE, LEARNING_RATE=LEARNING_RATE, max_length=max_length),
            "n_train": len(train_rows), "n_val": len(val_rows),
            "parse_rate_before_sft": parse_rate_before, "parse_rate_after_sft": parse_rate_after,
            "loss_history": history, "wall_clock_seconds": time.time() - t0,
        }, f, indent=2)
    print(f"-> {RESULTS / 'sft_run.json'}")


if __name__ == "__main__":
    main()
