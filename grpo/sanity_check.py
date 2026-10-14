"""
Verification step, not part of the real pipeline: before trusting the
"reward stayed at 0.000 for all 12 steps" result in train.py, rule out a
training-loop bug by swapping in an easy, learnable synthetic reward and
checking GRPO actually moves the policy.

Reward here: 1.0 if the completion is short (<= SHORT_THRESHOLD tokens),
0.0 otherwise. This is learnable from a random init because sampling
naturally produces *some* variance in when <eos> gets emitted -- unlike
"produce the exact substring 'Risk score: N'", which a random policy
essentially never hits, so the real reward signal in train.py never turns
on. If avg completion length drops over steps here, the GRPO update code
itself is doing its job; the flat-zero result in the real run is then a
cold-start/task-difficulty finding, not a bug.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import build_policy_and_reference, load_tokenizer
from prompts import build_examples
import train as T  # reuse generate_group / sequence_logprobs

SHORT_THRESHOLD = 15
N_STEPS = 15
GROUP_SIZE = 6
PROMPTS_PER_STEP = 4

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)


def short_completion_reward(texts, tokenizer):
    """Continuous reward (not a rare binary event): shorter completion =
    higher reward. There's already natural length variance across sampled
    completions (even near a model's max_new_tokens cap, *when* <eos>/a
    repeated pattern triggers the generation loop to stop varies a bit per
    sample), which is enough for a group-relative advantage to be nonzero
    every step -- unlike a binary threshold, which needs a rare exact event
    to ever fire from a random policy."""
    lens = [len(tokenizer(t)["input_ids"]) for t in texts]
    rewards = [-l / 100.0 for l in lens]
    return rewards, lens


def main():
    tokenizer = load_tokenizer()
    policy, reference = build_policy_and_reference(tokenizer)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=T.LEARNING_RATE)
    examples = build_examples()[:PROMPTS_PER_STEP * 3]

    print(f"Sanity check: reward=1.0 iff completion <= {SHORT_THRESHOLD} tokens. "
          f"If GRPO works, mean completion length should trend down.\n")

    history = []
    for step in range(N_STEPS):
        batch = random.sample(examples, PROMPTS_PER_STEP)
        all_lens, group_advantages = [], []
        full_ids_list, attn_list, prompt_lens = [], [], []

        for ex in batch:
            completions, texts, prompt_len = T.generate_group(policy, tokenizer, ex["prompt"], GROUP_SIZE, T.MAX_NEW_TOKENS)
            rewards, lens = short_completion_reward(texts, tokenizer)
            rewards = np.array(rewards)
            mean, std = rewards.mean(), rewards.std()
            advantages = (rewards - mean) / (std + 1e-4)
            all_lens.extend(lens)

            prompt_inputs = tokenizer(ex["prompt"], return_tensors="pt", truncation=True, max_length=600)
            for i in range(GROUP_SIZE):
                full = torch.cat([prompt_inputs["input_ids"][0], completions[i]]).unsqueeze(0)
                full_ids_list.append(full)
                attn_list.append(torch.ones_like(full))
                prompt_lens.append(prompt_len)
                group_advantages.append(advantages[i])

        import torch.nn.functional as F
        max_len = max(f.shape[1] for f in full_ids_list)
        pad_id = tokenizer.pad_token_id
        batch_ids = torch.cat([F.pad(f, (0, max_len - f.shape[1]), value=pad_id) for f in full_ids_list], dim=0)
        batch_attn = torch.cat([F.pad(a, (0, max_len - a.shape[1]), value=0) for a in attn_list], dim=0)
        advantages_t = torch.tensor(group_advantages, dtype=torch.float32)

        policy.train()
        n = len(full_ids_list)
        mean_logprob = torch.zeros(n)
        for i in range(len(batch)):
            idx = slice(i * GROUP_SIZE, (i + 1) * GROUP_SIZE)
            seq_lp, n_tok = T.sequence_logprobs(policy, batch_ids[idx], prompt_lens[i * GROUP_SIZE], batch_attn[idx])
            mean_logprob[idx] = seq_lp / n_tok

        loss = -(advantages_t.detach() * mean_logprob).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        print(f"[step {step+1:>2}/{N_STEPS}] mean_completion_len={np.mean(all_lens):.1f} "
              f"min={min(all_lens)} max={max(all_lens)} loss={loss.item():.5f}")
        history.append(dict(step=step + 1, mean_len=float(np.mean(all_lens)),
                             min_len=int(min(all_lens)), max_len=int(max(all_lens)),
                             loss=loss.item()))

    with open(Path(__file__).resolve().parent / "results" / "sanity_check_run.json", "w") as f:
        json.dump({"history": history, "short_threshold": SHORT_THRESHOLD}, f, indent=2)


if __name__ == "__main__":
    main()
