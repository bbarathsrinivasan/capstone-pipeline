"""
GRPO on a real pretrained model, starting from the SFT checkpoint in
sft.py. This is the "real" counterpart to train.py, which validated the
GRPO mechanics on a from-scratch, randomly-initialized model (see
results/findings.md) -- same algorithm, same reward, same group-relative
advantage update; only the model and the starting point change. Read
train.py's own docstring/comments first if the mechanics themselves
aren't already clear -- this file repeats them only where something
actually differs.

Run order: zero_shot_check.py -> sft.py -> train_real.py (this file).
Needs real network access to huggingface.co -- see README.md "Real-model
track" for where to run this (not this sandbox).
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_real import MODEL_NAME, CHECKPOINT_DIR, DEVICE, GPU_SCALE, load_tokenizer, build_policy_and_reference, free_memory
from prompts import build_examples
from reward import reward_batch

# sft.py's run exposed a real bug: a fixed, char-count-guessed max_length
# (640) silently truncated the completion off of most SFT batches, because
# Qwen's BPE tokenizer is far less token-efficient on this dense numeric/
# punctuation-heavy text than a character-count guess assumes (see sft.py's
# compute_max_length docstring for the full story -- short version: ~1.5-2
# chars/token here, not the ~3.3 a prose-based guess assumes). This file's
# `max_length=900` truncation calls below have the same root risk, just a
# quieter failure mode: instead of masking a completion to nothing, it
# would silently chop off the END of the clinical feature block fed to the
# model, which could bias generation without ever printing a warning. Fixed
# the same way -- measure it from the real tokenizer once, up front.
def compute_prompt_max_length(tokenizer, examples, margin=20):
    lens = [len(tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]) for ex in examples]
    max_len = max(lens)
    computed = max_len + margin
    print(f"Measured prompt token length: min={min(lens)} max={max_len} mean={sum(lens) / len(lens):.0f} "
          f"-> prompt truncation max_length set to {computed} (max prompt + {margin}-token margin).")
    return computed

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# --- what differs from train.py, and why -----------------------------------
# - GROUP_SIZE / PROMPTS_PER_STEP / N_STEPS are all smaller: a real model's
#   generate() + forward passes are far slower than the 1.2M-param stand-in's.
#   These scale with GPU_SCALE (from model_real.py: 4 on CUDA, 1 on Mac/CPU)
#   so a real GPU gets a bigger group/batch and more steps automatically,
#   instead of this staying tuned to "still runs on a laptop" after you move
#   to AWS. Raise N_STEPS by hand once a short run confirms things look
#   healthy -- it's a flat multiplier on wall-clock time, so it's the
#   cheapest knob to turn once you trust the rest of the setup.
# - LEARNING_RATE drops back to the paper's own 5e-5 (train.py used an
#   inflated 1e-3 specifically to get visible movement on CPU in 12 steps
#   on a model with no SFT at all -- that reasoning no longer applies here).
# - SFT_ADAPTER_PATH: starting point for the policy. Set to None to
#   reproduce the cold-start finding on a REAL model (expect a low but
#   nonzero parse rate -- better than 0.00, likely still not reliable --
#   see zero_shot_check.py's result for this specific number before
#   assuming either way).
GROUP_SIZE = 4 * GPU_SCALE
PROMPTS_PER_STEP = 4 * GPU_SCALE
N_STEPS = 8 * GPU_SCALE
MAX_NEW_TOKENS = 160
TEMPERATURE = 0.7  # was 1.0 -- three different code fixes (KL estimator, no
                   # dropout, 10x lower LR) all produced the SAME collapse
                   # with near-identical gibberish content, which argues
                   # against "destructive gradient step" as the mechanism (a
                   # 10x smaller LR step should have looked at least somewhat
                   # different and didn't). Working hypothesis instead: at
                   # temperature=1.0/top_p=0.95, sampling is exploratory
                   # enough that a near-tied early token decision can flip
                   # from a tiny/any weight change, and once that happens the
                   # autoregressive rollout drifts into a strong attractor
                   # in the BASE model's own output distribution (consistent
                   # math/Chinese-text content every time) rather than
                   # recovering. Lower temperature keeps sampling closer to
                   # the model's peak, making that tip-over less likely.
LEARNING_RATE = 5e-6  # was 5e-5 -- every run so far (3/3) collapsed the SFT
                      # checkpoint's format-adherence to 0 by step 2 regardless
                      # of which loss term produced the step-1 gradient (same
                      # outcome whether that gradient was negative- or
                      # positive-signed -- see results/grpo_real_run.json
                      # history across runs). 10x smaller as a direct test of
                      # whether the step size itself is what's knocking a
                      # narrow SFT optimum off its format cliff.
KL_COEF = 0.02
REWARD_VARIANT = "safety_shaped"
SEED = 0
SFT_ADAPTER_PATH = str(CHECKPOINT_DIR / "sft_lora")  # run sft.py first; or None for the cold-start comparison

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def sample_eval_set(examples, n_per_class=6):
    by_class = {0: [], 1: [], 2: [], 3: []}
    for e in examples:
        by_class[e["risk_int"]].append(e)
    for k in by_class:
        random.shuffle(by_class[k])
    chosen = []
    for k in range(4):
        chosen.extend(by_class[k][:n_per_class])
    random.shuffle(chosen)
    return chosen


@torch.no_grad()
def generate_group(policy, tokenizer, prompt, group_size, max_new_tokens, prompt_max_length):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=prompt_max_length)
    input_ids = inputs["input_ids"].repeat(group_size, 1).to(DEVICE)
    attention_mask = inputs["attention_mask"].repeat(group_size, 1).to(DEVICE)
    out = policy.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
    )
    completions = out[:, input_ids.shape[1]:]
    texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
    free_memory()  # see model_real.py's free_memory docstring -- MPS's caching
                   # allocator can ratchet up over repeated generate() calls
                   # otherwise, the same failure mode sft.py hit
    return completions, texts, inputs["input_ids"].shape[1]


def sequence_logprobs(model, full_ids, prompt_len, attention_mask):
    out = model(input_ids=full_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    targets = full_ids[:, 1:]
    logprobs = F.log_softmax(logits, dim=-1)
    token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = torch.zeros_like(targets, dtype=torch.float)
    mask[:, prompt_len - 1:] = 1.0
    mask = mask * (attention_mask[:, 1:].float())
    seq_logprob = (token_logprobs * mask).sum(dim=1)
    n_tokens = mask.sum(dim=1).clamp(min=1)
    return seq_logprob, n_tokens


def grpo_step(policy, tokenizer, optimizer, batch_examples, prompt_max_length):
    all_rewards, all_parsed, all_texts = [], [], []
    group_advantages = []
    full_ids_list, attn_list, prompt_lens = [], [], []

    for ex in batch_examples:
        completions, texts, prompt_len = generate_group(
            policy, tokenizer, ex["prompt"], GROUP_SIZE, MAX_NEW_TOKENS, prompt_max_length
        )
        rewards, parsed = reward_batch(texts, [ex["risk_int"]] * GROUP_SIZE, variant=REWARD_VARIANT)
        rewards = np.array(rewards)
        mean, std = rewards.mean(), rewards.std()
        advantages = (rewards - mean) / (std + 1e-4)

        prompt_inputs = tokenizer(ex["prompt"], return_tensors="pt", truncation=True, max_length=prompt_max_length)
        prompt_inputs = {k: v.to(DEVICE) for k, v in prompt_inputs.items()}
        for i in range(GROUP_SIZE):
            full = torch.cat([prompt_inputs["input_ids"][0], completions[i]]).unsqueeze(0)
            attn = torch.ones_like(full)
            full_ids_list.append(full)
            attn_list.append(attn)
            prompt_lens.append(prompt_len)
            group_advantages.append(advantages[i])

        all_rewards.extend(rewards.tolist())
        all_parsed.extend(parsed)
        all_texts.extend(texts)

    max_len = max(f.shape[1] for f in full_ids_list)
    pad_id = tokenizer.pad_token_id
    padded_ids, padded_attn = [], []
    for f, a in zip(full_ids_list, attn_list):
        pad_n = max_len - f.shape[1]
        padded_ids.append(F.pad(f, (0, pad_n), value=pad_id))
        padded_attn.append(F.pad(a, (0, pad_n), value=0))
    batch_ids = torch.cat(padded_ids, dim=0)
    batch_attn = torch.cat(padded_attn, dim=0)
    advantages_t = torch.tensor(group_advantages, dtype=torch.float32, device=DEVICE)

    policy.train()
    policy_logprob_sum = torch.zeros(len(full_ids_list), device=DEVICE)
    n_tok_all = torch.zeros(len(full_ids_list), device=DEVICE)
    for i in range(len(batch_examples)):
        pl = prompt_lens[i * GROUP_SIZE]
        idx = slice(i * GROUP_SIZE, (i + 1) * GROUP_SIZE)
        seq_lp, n_tok = sequence_logprobs(policy, batch_ids[idx], pl, batch_attn[idx])
        policy_logprob_sum[idx] = seq_lp
        n_tok_all[idx] = n_tok
    mean_logprob = policy_logprob_sum / n_tok_all

    with torch.no_grad():
        ref_logprob_sum = torch.zeros(len(full_ids_list), device=DEVICE)
        # Reference logprobs on the SAME model with its LoRA adapter
        # temporarily disabled, instead of a separate deepcopy()'d reference
        # model -- that deepcopy kept a full second 4B-parameter base model
        # resident in GPU memory (~8GB in bf16) for the whole run just for
        # this KL term, which is exactly what pushed GRPO's memory past what
        # SFT needed on the same GPU (SFT only ever has ONE model resident).
        # disable_adapter() makes `policy` behave as the plain pretrained
        # base -- mathematically identical to the deepcopy reference, same
        # weights, just without the duplicate copy.
        with policy.disable_adapter():
            for i in range(len(batch_examples)):
                pl = prompt_lens[i * GROUP_SIZE]
                idx = slice(i * GROUP_SIZE, (i + 1) * GROUP_SIZE)
                seq_lp, n_tok = sequence_logprobs(policy, batch_ids[idx], pl, batch_attn[idx])
                ref_logprob_sum[idx] = seq_lp
        ref_mean_logprob = ref_logprob_sum / n_tok_all

    pg_loss = -(advantages_t.detach() * mean_logprob).mean()
    # KL estimator: Schulman's "k3" estimator (http://joschu.net/blog/kl-approx.html),
    # not the raw signed difference this used to be. The raw difference
    # (mean_logprob - ref_mean_logprob) is unbiased in EXPECTATION over
    # x~policy, but for any single sampled x it can be negative -- and once
    # pg_loss has zero gradient (every rollout in a batch gets reward 0, so
    # advantage=0 for all of them -- exactly what parse_success_rate=0.0
    # means in grpo_real_run.json), that raw difference becomes the ONLY
    # live gradient. Minimizing loss then means minimizing it: pushing the
    # policy to assign LOWER probability to the very sequence it just
    # sampled, with nothing to anchor it. That is the mechanism behind the
    # observed collapse (kl drifting from -0.002 to -0.46 over steps 2-8,
    # final generations total gibberish) -- the "KL penalty" was, in that
    # regime, an incentive to become incoherent, not a containment term.
    # The k3 estimator has the same expectation but is >=0 for every
    # sample (e^r - 1 - r >= 0 for all real r, equality only at r=0), so
    # this exploit isn't reachable: drifting from the reference can only
    # ever raise the loss, never lower it, even when pg_loss is silent.
    log_ratio = ref_mean_logprob.detach() - mean_logprob  # r = log pi_ref(x) - log pi(x)
    kl = (torch.exp(log_ratio) - 1 - log_ratio).mean()
    loss = pg_loss + KL_COEF * kl

    # Same dtype-mismatch nan risk as sft.py (see model_real.py's
    # load_base_model docstring) -- skip the update rather than take a step
    # on a corrupted gradient.
    optimizer.zero_grad()
    if not torch.isfinite(loss):
        print(f"WARNING: loss is {loss.item()} (non-finite) -- skipping this step's update.")
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
        optimizer.step()

    return dict(
        loss=loss.item(), pg_loss=pg_loss.item(), kl=kl.item(),
        reward_mean=float(np.mean(all_rewards)), reward_std=float(np.std(all_rewards)),
        parse_success_rate=float(np.mean([p is not None for p in all_parsed])),
        rewards=all_rewards, parsed=all_parsed, texts=all_texts,
    )


def main():
    tokenizer = load_tokenizer()
    sft_path = SFT_ADAPTER_PATH if Path(SFT_ADAPTER_PATH).exists() else None
    if SFT_ADAPTER_PATH and sft_path is None:
        print(f"WARNING: {SFT_ADAPTER_PATH} not found -- run sft.py first. "
              f"Continuing with a FRESH LoRA init (reproduces the cold-start finding on a real model).")
    policy = build_policy_and_reference(sft_adapter_path=sft_path)
    # Disable dropout on the policy before any RL training. LORA_CONFIG
    # (model_real.py) sets lora_dropout=0.05 -- harmless, even useful, as
    # ordinary regularization during sft.py's supervised training, but
    # actively wrong here: nothing in this file ever puts the model in
    # eval() mode, so that 5% dropout is stochastically active in BOTH
    # generate_group()'s forward pass (which produces the sampled tokens)
    # and sequence_logprobs()'s separate forward pass a few lines below
    # (which computes log pi_theta(x) for those same tokens, for the
    # gradient). Those are two different calls with two different random
    # dropout masks over the SAME weights -- so "log pi_theta(x)" is not
    # actually well-defined for a fixed theta and x the way a policy
    # gradient update requires; the x being scored was never really drawn
    # from the exact distribution being differentiated. TRL's PPO/GRPO
    # trainers call the equivalent of this for exactly this reason before
    # RL (never before SFT). Candidate explanation for why a single,
    # correctly-clipped step 1 update was enough to collapse the SFT
    # checkpoint's format-adherence onto pure gibberish in
    # grpo_real_run.json: this fix makes pi_theta deterministic given
    # theta, closing that gap, but hasn't been isolated/confirmed yet --
    # rerun and check whether parse_rate now survives past step 1.
    for module in policy.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    # Same OOM fix as sft.py: see its main() comment for why this (not the
    # cache-clearing in free_memory) is what actually controls memory for
    # the training forward/backward pass.
    policy.gradient_checkpointing_enable()
    policy.enable_input_require_grads()
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=LEARNING_RATE)

    examples = build_examples()
    prompt_max_length = compute_prompt_max_length(tokenizer, examples)
    eval_set = sample_eval_set(examples, n_per_class=6)
    print(f"Training pool: {len(eval_set)} class-balanced tier-1 prompts "
          f"(6 per class), model='{MODEL_NAME}', sft_checkpoint={'yes' if sft_path else 'NO'}, "
          f"reward variant='{REWARD_VARIANT}'")

    history = []
    t0 = time.time()
    for step in range(N_STEPS):
        batch = random.sample(eval_set, PROMPTS_PER_STEP)
        step_log = grpo_step(policy, tokenizer, optimizer, batch, prompt_max_length)
        # Keep texts/rewards/parsed per step now (previously stripped out) --
        # until now we could only see the final_samples at the very end of
        # the whole run, with no visibility into WHEN or HOW a step's
        # rollouts actually drift. This is exactly the data needed to tell
        # apart "garbled from the first token" vs "starts coherent, drifts
        # mid-generation" for the temperature-based fix above, instead of
        # inferring blindly from the aggregate numbers alone.
        history.append(step_log)
        print(f"[step {step + 1:>2}/{N_STEPS}] "
              f"reward_mean={step_log['reward_mean']:.3f} "
              f"parse_rate={step_log['parse_success_rate']:.2f} "
              f"loss={step_log['loss']:.4f} kl={step_log['kl']:.4f} "
              f"({time.time() - t0:.0f}s elapsed)")

    final_examples = random.sample(eval_set, 4)
    final_samples = []
    for ex in final_examples:
        _, texts, _ = generate_group(policy, tokenizer, ex["prompt"], 1, MAX_NEW_TOKENS, prompt_max_length)
        rewards, parsed = reward_batch(texts, [ex["risk_int"]], variant=REWARD_VARIANT)
        final_samples.append({
            "sample_id": ex["sample_id"], "true_risk": ex["risk_category"],
            "generation": texts[0], "parsed_score": parsed[0], "reward": rewards[0],
        })

    out = {
        "config": dict(
            MODEL_NAME=MODEL_NAME, sft_checkpoint_used=bool(sft_path),
            GROUP_SIZE=GROUP_SIZE, PROMPTS_PER_STEP=PROMPTS_PER_STEP, N_STEPS=N_STEPS,
            MAX_NEW_TOKENS=MAX_NEW_TOKENS, LEARNING_RATE=LEARNING_RATE, KL_COEF=KL_COEF,
            REWARD_VARIANT=REWARD_VARIANT,
        ),
        "history": history,
        "final_samples": final_samples,
        "wall_clock_seconds": time.time() - t0,
    }
    with open(RESULTS / "grpo_real_run.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {out['wall_clock_seconds']:.0f}s. -> {RESULTS / 'grpo_real_run.json'}")


if __name__ == "__main__":
    main()
