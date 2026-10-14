"""
GRPO training loop, implemented manually (not via TRL's GRPOTrainer) so
every step is visible and runs fully offline. Mirrors the algorithm in the
TimeSRL paper (arXiv:2605.21295, Sec. on GRPO+RLVR) and the original GRPO
paper (DeepSeekMath, arXiv:2402.03300): group-relative advantage, no
learned value/critic network, KL penalty against a frozen reference model.

Read top to bottom -- this file IS the "how does the pipeline work" answer.

Known, deliberate simplification vs. the papers: this runs ONE gradient
step per group of rollouts (no PPO-style multi-epoch clipped updates over
replayed rollouts). That's a real simplification, not an oversight -- see
results/grpo_findings.md for why it doesn't matter for what this run is
testing (pipeline mechanics on a from-scratch model), and why it would
matter for a real training run.
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
from model import build_policy_and_reference, load_tokenizer
from prompts import build_examples
from reward import reward_batch

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# --- run-size knobs: kept deliberately tiny for a CPU mechanics demo -------
GROUP_SIZE = 4          # G rollouts per prompt (paper uses K=8 on 4 H100s)
PROMPTS_PER_STEP = 6     # prompts per GRPO update
N_STEPS = 12             # total GRPO updates
MAX_NEW_TOKENS = 90
TEMPERATURE = 1.0
LEARNING_RATE = 1e-3     # higher than the paper's 5e-5 -- small model, few steps
KL_COEF = 0.02
REWARD_VARIANT = "safety_shaped"  # or "timesrl"
SEED = 0

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def sample_eval_set(examples, n_per_class=6):
    """Small, class-balanced subset so a 12-step CPU run actually sees all
    four risk levels, including Severe, every step."""
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
def generate_group(policy, tokenizer, prompt, group_size, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=600)
    input_ids = inputs["input_ids"].repeat(group_size, 1)
    attention_mask = inputs["attention_mask"].repeat(group_size, 1)
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
    return completions, texts, inputs["input_ids"].shape[1]


def sequence_logprobs(model, full_ids, prompt_len, attention_mask):
    """Sum of log p(token_t | tokens_<t) over the COMPLETION tokens only
    (everything after prompt_len), for each sequence in the batch."""
    out = model(input_ids=full_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    targets = full_ids[:, 1:]
    logprobs = F.log_softmax(logits, dim=-1)
    token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # mask: only completion tokens (index >= prompt_len - 1 in the shifted frame)
    mask = torch.zeros_like(targets, dtype=torch.float)
    mask[:, prompt_len - 1:] = 1.0
    mask = mask * (attention_mask[:, 1:].float())
    seq_logprob = (token_logprobs * mask).sum(dim=1)
    n_tokens = mask.sum(dim=1).clamp(min=1)
    return seq_logprob, n_tokens


def grpo_step(policy, reference, tokenizer, optimizer, batch_examples):
    all_rewards, all_parsed, all_texts = [], [], []
    group_advantages = []
    full_ids_list, attn_list, prompt_lens = [], [], []

    for ex in batch_examples:
        completions, texts, prompt_len = generate_group(
            policy, tokenizer, ex["prompt"], GROUP_SIZE, MAX_NEW_TOKENS
        )
        rewards, parsed = reward_batch(texts, [ex["risk_int"]] * GROUP_SIZE, variant=REWARD_VARIANT)
        rewards = np.array(rewards)
        # GRPO's core move: normalize reward WITHIN the group, no value network.
        mean, std = rewards.mean(), rewards.std()
        advantages = (rewards - mean) / (std + 1e-4)

        prompt_inputs = tokenizer(ex["prompt"], return_tensors="pt", truncation=True, max_length=600)
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

    # pad to a common length for one batched forward pass
    max_len = max(f.shape[1] for f in full_ids_list)
    pad_id = tokenizer.pad_token_id
    padded_ids, padded_attn = [], []
    for f, a in zip(full_ids_list, attn_list):
        pad_n = max_len - f.shape[1]
        padded_ids.append(F.pad(f, (0, pad_n), value=pad_id))
        padded_attn.append(F.pad(a, (0, pad_n), value=0))
    batch_ids = torch.cat(padded_ids, dim=0)
    batch_attn = torch.cat(padded_attn, dim=0)
    advantages_t = torch.tensor(group_advantages, dtype=torch.float32)

    # policy term: per-sequence mean log-prob (length-normalized), weighted
    # by the group-relative advantage -- this is the GRPO policy-gradient term.
    policy.train()
    policy_logprob_sum = torch.zeros(len(full_ids_list))
    n_tok_all = torch.zeros(len(full_ids_list))
    # loop per-example prompt_len since they can differ slightly by padding
    offset = 0
    for i in range(len(batch_examples)):
        pl = prompt_lens[i * GROUP_SIZE]
        idx = slice(i * GROUP_SIZE, (i + 1) * GROUP_SIZE)
        seq_lp, n_tok = sequence_logprobs(policy, batch_ids[idx], pl, batch_attn[idx])
        policy_logprob_sum[idx] = seq_lp
        n_tok_all[idx] = n_tok
    mean_logprob = policy_logprob_sum / n_tok_all

    with torch.no_grad():
        ref_logprob_sum = torch.zeros(len(full_ids_list))
        for i in range(len(batch_examples)):
            pl = prompt_lens[i * GROUP_SIZE]
            idx = slice(i * GROUP_SIZE, (i + 1) * GROUP_SIZE)
            seq_lp, n_tok = sequence_logprobs(reference, batch_ids[idx], pl, batch_attn[idx])
            ref_logprob_sum[idx] = seq_lp
        ref_mean_logprob = ref_logprob_sum / n_tok_all

    pg_loss = -(advantages_t.detach() * mean_logprob).mean()
    kl = (mean_logprob - ref_mean_logprob.detach()).mean()  # approx per-token KL direction
    loss = pg_loss + KL_COEF * kl

    optimizer.zero_grad()
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
    policy, reference = build_policy_and_reference(tokenizer)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=LEARNING_RATE)

    examples = build_examples()
    eval_set = sample_eval_set(examples, n_per_class=6)
    print(f"Training pool: {len(eval_set)} class-balanced tier-1 prompts "
          f"(6 per class), reward variant='{REWARD_VARIANT}'")

    history = []
    t0 = time.time()
    for step in range(N_STEPS):
        batch = random.sample(eval_set, PROMPTS_PER_STEP)
        step_log = grpo_step(policy, reference, tokenizer, optimizer, batch)
        history.append({k: v for k, v in step_log.items() if k not in ("rewards", "parsed", "texts")})
        print(f"[step {step+1:>2}/{N_STEPS}] "
              f"reward_mean={step_log['reward_mean']:.3f} "
              f"parse_rate={step_log['parse_success_rate']:.2f} "
              f"loss={step_log['loss']:.4f} kl={step_log['kl']:.4f} "
              f"({time.time() - t0:.0f}s elapsed)")

    # one final held-out-style pass to capture example generations before/after
    final_examples = random.sample(eval_set, 4)
    final_samples = []
    for ex in final_examples:
        _, texts, _ = generate_group(policy, tokenizer, ex["prompt"], 1, MAX_NEW_TOKENS)
        rewards, parsed = reward_batch(texts, [ex["risk_int"]], variant=REWARD_VARIANT)
        final_samples.append({
            "sample_id": ex["sample_id"], "true_risk": ex["risk_category"],
            "generation": texts[0], "parsed_score": parsed[0], "reward": rewards[0],
        })

    out = {
        "config": dict(
            GROUP_SIZE=GROUP_SIZE, PROMPTS_PER_STEP=PROMPTS_PER_STEP, N_STEPS=N_STEPS,
            MAX_NEW_TOKENS=MAX_NEW_TOKENS, LEARNING_RATE=LEARNING_RATE, KL_COEF=KL_COEF,
            REWARD_VARIANT=REWARD_VARIANT, model_params=sum(p.numel() for p in policy.parameters()),
        ),
        "history": history,
        "final_samples": final_samples,
        "wall_clock_seconds": time.time() - t0,
    }
    with open(RESULTS / "grpo_run.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {out['wall_clock_seconds']:.0f}s. -> {RESULTS / 'grpo_run.json'}")


if __name__ == "__main__":
    main()
