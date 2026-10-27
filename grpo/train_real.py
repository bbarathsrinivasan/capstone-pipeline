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
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_real import MODEL_NAME, CHECKPOINT_DIR, DEVICE, GPU_SCALE, load_tokenizer, build_policy_and_reference, free_memory
from prompts import build_examples, split_examples_by_subject
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
# Overridable via GRPO_GROUP_SIZE now, same pattern as GRPO_N_STEPS below --
# lets a scale-up/ablation run tune rollout-group size without editing this
# file (bigger groups give a less noisy group-mean/std for the advantage
# estimate, at a roughly linear generate() time cost).
GROUP_SIZE = int(os.environ.get("GRPO_GROUP_SIZE", 4 * GPU_SCALE))
PROMPTS_PER_STEP = 4 * GPU_SCALE
# How many of a GROUP_SIZE batch's sequences sequence_logprobs() forwards
# through the model at once. Added 2026-10-27 after TWO prior OOM fixes in
# that function (log_softmax->gather, then logsumexp) both still OOM'd at
# virtually the same memory point -- neither actually bounds peak memory,
# since the forward pass's own activations (even with grad checkpointing)
# and any full-vocab reduction's internal temporaries both still scale
# with however many sequences go through `model(...)` in one call. This
# caps that at a fixed, small number regardless of GROUP_SIZE/GPU_SCALE --
# a real ceiling instead of another guess at what fits.
LOGPROB_MICRO_BATCH = int(os.environ.get("GRPO_LOGPROB_MICRO_BATCH", 2))
# N_STEPS used to be a flat 8*GPU_SCALE with no way to change it short of
# editing this file -- fine while every run was dying by step 2 regardless,
# not fine now that the real bug (policy left in train() mode during
# generation, see generate_group()'s docstring) is fixed and steps actually
# work. Overridable via GRPO_N_STEPS so scaling up doesn't need a new
# round-trip through this file. Default raised 8 -> 40 (5x) now that a
# longer run is worth the wall-clock/GPU cost -- each step was ~65-70s on a
# g5.xlarge for Qwen3-4B, so 40 steps is roughly 45 minutes.
N_STEPS = int(os.environ.get("GRPO_N_STEPS", 8 * GPU_SCALE * 5))
VAL_FRACTION = 0.15  # must match sft.py's -- see split_examples_by_subject
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

# Periodic held-out probing during training (limitation fix: previously the
# only held-out signal was the ONE evaluate_before_after.py pass at the very
# end -- no visibility into whether the best checkpoint was actually the
# LAST step, or whether training had already started overfitting/drifting
# by then). GRPO_EVAL_EVERY=0 disables probing entirely (falls back to
# saving only the final-step weights).
GRPO_EVAL_EVERY = int(os.environ.get("GRPO_EVAL_EVERY", 10))
GRPO_EVAL_N = int(os.environ.get("GRPO_EVAL_N", 20))

# Points at the W&B project you created at https://wandb.ai/authorize
# ("Capstone Implementation" under the bbasavar-carnegie-mellon-university
# org/entity). Override either via env var if you ever log to a different
# project/entity without editing this file.
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "bbasavar-carnegie-mellon-university")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "Capstone Implementation")
# Set once (e.g. by run_gpu.sh, which exports a timestamp-based value) so
# the sft/grpo/eval runs from one full pipeline run show up grouped
# together in the W&B UI instead of as unrelated runs.
WANDB_RUN_GROUP = os.environ.get("WANDB_RUN_GROUP")
WANDB_ENABLED = WANDB_AVAILABLE and os.environ.get("WANDB_DISABLED", "").lower() not in ("1", "true")

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


@torch.no_grad()
def generate_group(policy, tokenizer, prompt, group_size, max_new_tokens, prompt_max_length):
    # This is the actual root cause of the collapse, not the KL term, not
    # dropout, not LR, not temperature (all tried, none worked -- see
    # results/grpo_real_run.json history across runs, and the EC2 log's
    # "Caching is incompatible with gradient checkpointing in
    # Qwen3DecoderLayer. Setting past_key_values=None." warning, which only
    # ever appears starting at step 2). grpo_step() calls policy.train()
    # right before computing the gradient, and nothing ever calls
    # policy.eval() back afterward. HF's gradient-checkpointing path only
    # activates when self.training=True, and Qwen3's decoder layer disables
    # the KV cache whenever that path is active -- so from step 2 onward,
    # every generate() call here was silently decoding WITHOUT a working
    # cache, with gradient checkpointing's activation-recomputation
    # machinery (built for one full-sequence training forward pass) running
    # underneath an incremental, one-token-at-a-time generation loop it was
    # never designed for. Step 1 was never exposed to this because
    # policy.train() hadn't been called yet the first time through. Forcing
    # eval() here (restoring it afterward is grpo_step()'s job, which it
    # already does via its own policy.train() call) makes every generate()
    # call -- every step, not just step 1 -- use the normal, cached,
    # correct decode path regardless of what happened in a previous step's
    # training forward pass.
    policy.eval()
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
        # repetition_penalty, NOT no_repeat_ngram_size -- a real bug found
        # 2026-10-23: no_repeat_ngram_size=3 (this file's previous fix for a
        # degenerate repeat-loop on a true-Severe miss) blocks any 3-gram
        # that already occurred ANYWHERE in the full sequence, prompt
        # included -- and prompts.py's own instruction text contains the
        # literal phrase "Risk score: <0, 1, 2, or 3>". That permanently
        # banned the model from ever generating "Risk score:" again,
        # causing a deterministic 0.00 parse rate (not a training problem --
        # SFT's loss curve was fine; eval_parse_rate's greedy decode was
        # structurally blocked from the correct phrase). repetition_penalty
        # softly discourages repeating tokens already seen (prompt or
        # generated) without ever fully banning a phrase, so it still damps
        # a degenerate loop while leaving "Risk score: N" reachable.
        repetition_penalty=1.3,
        pad_token_id=tokenizer.pad_token_id,
    )
    completions = out[:, input_ids.shape[1]:]
    texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
    free_memory()  # see model_real.py's free_memory docstring -- MPS's caching
                   # allocator can ratchet up over repeated generate() calls
                   # otherwise, the same failure mode sft.py hit
    return completions, texts, inputs["input_ids"].shape[1]


def sequence_logprobs(model, full_ids, prompt_len, attention_mask):
    # Third fix for the same OOM (2026-10-27), and the first that actually
    # bounds peak memory instead of just changing how a vocab-sized
    # reduction is computed. Two earlier attempts:
    #   1. log_softmax(logits) + gather -- materializes a SECOND full
    #      (batch, seq_len, vocab~152k) tensor alongside `logits` itself.
    #   2. gathered_logit - logsumexp(logits) -- mathematically avoids
    #      that second tensor, but logsumexp's OWN internal exp() step
    #      still materializes a same-size temporary to do the reduction,
    #      so it OOM'd at virtually the identical memory point (confirmed:
    #      the 2026-10-27 traceback moved from the log_softmax line to the
    #      logsumexp line, needing almost exactly the same ~2GiB).
    # Neither fix touches the OTHER thing that scales with batch size: the
    # forward pass's own activation memory for however many sequences go
    # through `model(...)` in one call (gradient checkpointing reduces but
    # doesn't eliminate this). So this version does two things instead:
    # (a) F.cross_entropy's fused kernel -- the exact op HF's own
    # `model(..., labels=...)` path uses internally, confirmed NOT to OOM
    # in sft.py at this same model/batch/seq-length scale -- computes the
    # per-token negative log-likelihood via a streaming reduction over the
    # vocab dim, without a second full-size tensor; (b) forwarding only
    # LOGPROB_MICRO_BATCH sequences through the model at a time instead of
    # the whole GROUP_SIZE batch in one call, which caps peak memory at a
    # fixed number regardless of how large GROUP_SIZE/GPU_SCALE get later --
    # a real ceiling, not another guess at what happens to fit today.
    seq_logprobs, n_tokens_list = [], []
    for start in range(0, full_ids.shape[0], LOGPROB_MICRO_BATCH):
        end = start + LOGPROB_MICRO_BATCH
        ids_chunk = full_ids[start:end]
        attn_chunk = attention_mask[start:end]
        out = model(input_ids=ids_chunk, attention_mask=attn_chunk)
        logits = out.logits[:, :-1, :]
        targets = ids_chunk[:, 1:]
        token_logprobs = -F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
        ).view(targets.shape)
        mask = torch.zeros_like(targets, dtype=torch.float)
        mask[:, prompt_len - 1:] = 1.0
        mask = mask * (attn_chunk[:, 1:].float())
        seq_logprobs.append((token_logprobs * mask).sum(dim=1))
        n_tokens_list.append(mask.sum(dim=1).clamp(min=1))
    return torch.cat(seq_logprobs), torch.cat(n_tokens_list)


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


def epoch_batches(pool, batch_size, seed=SEED):
    """Cycling shuffled sampler over the full training pool, replacing the
    old `random.sample(eval_set, PROMPTS_PER_STEP)` per step. Independent
    resampling each step means some of the 411 training examples could go
    the whole run never seen while others get sampled repeatedly (expected
    duplicates in N_STEPS*PROMPTS_PER_STEP draws from 411 with no
    replacement tracking); this instead reshuffles the whole pool and walks
    it in fixed-size chunks, so one full pass ("epoch") touches every
    example about once before any repeats, with a fresh shuffle each pass."""
    pool = list(pool)
    rng = random.Random(seed)
    while True:
        rng.shuffle(pool)
        n_full = len(pool) - (len(pool) % batch_size)
        for i in range(0, n_full, batch_size):
            yield pool[i:i + batch_size]


@torch.no_grad()
def quick_eval(policy, tokenizer, examples, prompt_max_length, n=GRPO_EVAL_N):
    """Cheap mid-training readiness probe on held-out subjects -- greedy
    decode (not GRPO's stochastic rollouts), scored the same way
    evaluate_before_after.py scores a checkpoint, but on a SMALL subsample
    (n, not the full held-out set) so this can run every few steps without
    meaningfully slowing training down. Used for best-checkpoint selection:
    without this, main() had no way to tell whether the step-N weights it
    saved were actually the best the run produced, or whether training had
    already started drifting by step N."""
    policy.eval()
    sample = random.sample(examples, min(n, len(examples)))
    texts, labels = [], []
    for ex in sample:
        inputs = tokenizer(ex["prompt"], return_tensors="pt", truncation=True, max_length=prompt_max_length)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        out = policy.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            repetition_penalty=1.3, pad_token_id=tokenizer.pad_token_id,  # see generate_group's comment
        )
        texts.append(tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
        labels.append(ex["risk_int"])
        del out, inputs
        free_memory()
    rewards, parsed = reward_batch(texts, labels, variant=REWARD_VARIANT)
    n_exact = sum(1 for p, y in zip(parsed, labels) if p is not None and round(p) == y)
    return dict(
        mean_reward=float(np.mean(rewards)),
        parse_rate=float(np.mean([p is not None for p in parsed])),
        exact_match_accuracy=n_exact / len(sample),
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
    # Used to be a hand-picked 24-example stratified subsample of the full
    # 480 (sample_eval_set, n_per_class=6) -- far too little diversity for a
    # real training run, and with no guarantee it avoided whatever subjects
    # sft.py held out for its own validation. Now: the SAME subject-grouped
    # split sft.py uses (prompts.py's split_examples_by_subject, same
    # VAL_FRACTION/SEED), training only on train_examples -- the held-out
    # val_examples subjects are never trained on by either stage, and stay
    # available as the genuine before/after holdout (evaluate_before_after.py
    # reconstructs the identical split independently).
    eval_set, held_out_examples = split_examples_by_subject(examples, VAL_FRACTION, SEED)
    print(f"GRPO training pool: {len(eval_set)} tier-1 prompts from "
          f"{len({e['subject_id'] for e in eval_set})} subjects, model='{MODEL_NAME}', "
          f"sft_checkpoint={'yes' if sft_path else 'NO'}, reward variant='{REWARD_VARIANT}'. "
          f"{len(held_out_examples)} examples from "
          f"{len({e['subject_id'] for e in held_out_examples})} subjects held out "
          f"(never trained on) for evaluate_before_after.py.")

    run = None
    if WANDB_ENABLED:
        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            group=WANDB_RUN_GROUP,
            job_type="grpo",
            name=f"grpo-{MODEL_NAME.split('/')[-1]}-{int(time.time())}",
            tags=["grpo", MODEL_NAME.split("/")[-1], REWARD_VARIANT],
            notes=(
                f"GRPO (RL stage) on {MODEL_NAME} starting from "
                f"{'the SFT LoRA checkpoint' if sft_path else 'a FRESH LoRA init (cold-start run)'}. "
                f"{len(eval_set)} training prompts / {len(held_out_examples)} held-out, "
                f"group_size={GROUP_SIZE}, {N_STEPS} steps, reward='{REWARD_VARIANT}'. "
                f"Held-out probe every {GRPO_EVAL_EVERY} steps selects the best checkpoint "
                f"saved to checkpoints/grpo_lora (checkpoints/grpo_lora_final is always the last step)."
            ),
            config=dict(
                MODEL_NAME=MODEL_NAME, sft_checkpoint_used=bool(sft_path),
                GROUP_SIZE=GROUP_SIZE, PROMPTS_PER_STEP=PROMPTS_PER_STEP, N_STEPS=N_STEPS,
                MAX_NEW_TOKENS=MAX_NEW_TOKENS, LEARNING_RATE=LEARNING_RATE, KL_COEF=KL_COEF,
                TEMPERATURE=TEMPERATURE, REWARD_VARIANT=REWARD_VARIANT,
                GRPO_EVAL_EVERY=GRPO_EVAL_EVERY, GRPO_EVAL_N=GRPO_EVAL_N,
            ),
        )
    elif not WANDB_AVAILABLE:
        print("wandb not installed -- skipping W&B logging (pip install wandb to enable).")

    grpo_save_path = CHECKPOINT_DIR / "grpo_lora"          # best-so-far by held-out probe
    final_save_path = CHECKPOINT_DIR / "grpo_lora_final"   # always written, last-step weights
    best_reward = -float("inf")
    best_step = None

    batch_iter = epoch_batches(eval_set, PROMPTS_PER_STEP)
    history = []
    t0 = time.time()
    for step in range(N_STEPS):
        batch = next(batch_iter)
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
        if run is not None:
            run.log({
                "train/reward_mean": step_log["reward_mean"], "train/reward_std": step_log["reward_std"],
                "train/parse_success_rate": step_log["parse_success_rate"], "train/loss": step_log["loss"],
                "train/pg_loss": step_log["pg_loss"], "train/kl": step_log["kl"],
            }, step=step + 1)

        if GRPO_EVAL_EVERY and ((step + 1) % GRPO_EVAL_EVERY == 0 or step == N_STEPS - 1):
            probe = quick_eval(policy, tokenizer, held_out_examples, prompt_max_length, GRPO_EVAL_N)
            print(f"  [held-out probe @ step {step + 1}] n={min(GRPO_EVAL_N, len(held_out_examples))} "
                  f"mean_reward={probe['mean_reward']:.3f} parse_rate={probe['parse_rate']:.2f} "
                  f"exact_match_accuracy={probe['exact_match_accuracy']:.2f}")
            if run is not None:
                run.log({
                    "eval/mean_reward": probe["mean_reward"], "eval/parse_rate": probe["parse_rate"],
                    "eval/exact_match_accuracy": probe["exact_match_accuracy"],
                }, step=step + 1)
            if probe["mean_reward"] > best_reward:
                best_reward, best_step = probe["mean_reward"], step + 1
                policy.save_pretrained(str(grpo_save_path))
                print(f"  -> new best held-out mean_reward={best_reward:.3f}, saved checkpoints/grpo_lora")

    final_examples = random.sample(eval_set, 4)
    final_samples = []
    for ex in final_examples:
        _, texts, _ = generate_group(policy, tokenizer, ex["prompt"], 1, MAX_NEW_TOKENS, prompt_max_length)
        rewards, parsed = reward_batch(texts, [ex["risk_int"]], variant=REWARD_VARIANT)
        final_samples.append({
            "sample_id": ex["sample_id"], "true_risk": ex["risk_category"],
            "generation": texts[0], "parsed_score": parsed[0], "reward": rewards[0],
        })

    # Always save the last-step weights (previously train_real.py never saved
    # a checkpoint at all, so there was no way to evaluate "after GRPO"
    # separately from "during GRPO training"). Additionally, checkpoints/
    # grpo_lora holds the BEST held-out probe snapshot found above, not just
    # whatever the last step happened to produce -- if no probe ever ran or
    # none ever beat -inf (GRPO_EVAL_EVERY=0, or a very short run), fall back
    # to the final weights so grpo_lora always exists for
    # evaluate_before_after.py to load.
    policy.save_pretrained(str(final_save_path))
    print(f"Saved final-step LoRA adapter to {final_save_path}")
    if best_step is None:
        policy.save_pretrained(str(grpo_save_path))
        print(f"No held-out probe improved during training -- also saved final weights to {grpo_save_path}")
    else:
        print(f"Best held-out mean_reward={best_reward:.3f} at step {best_step} -- {grpo_save_path} holds that checkpoint.")

    out = {
        "config": dict(
            MODEL_NAME=MODEL_NAME, sft_checkpoint_used=bool(sft_path),
            GROUP_SIZE=GROUP_SIZE, PROMPTS_PER_STEP=PROMPTS_PER_STEP, N_STEPS=N_STEPS,
            MAX_NEW_TOKENS=MAX_NEW_TOKENS, LEARNING_RATE=LEARNING_RATE, KL_COEF=KL_COEF,
            REWARD_VARIANT=REWARD_VARIANT,
        ),
        "history": history,
        "final_samples": final_samples,
        "best_checkpoint": dict(step=best_step, mean_reward=None if best_step is None else best_reward),
        "wall_clock_seconds": time.time() - t0,
    }
    with open(RESULTS / "grpo_real_run.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {out['wall_clock_seconds']:.0f}s. -> {RESULTS / 'grpo_real_run.json'}")

    if run is not None:
        run.summary["best_step"] = best_step
        run.summary["best_mean_reward"] = best_reward if best_step is not None else None
        run.log({"final_samples": wandb.Table(
            columns=["sample_id", "true_risk", "generation", "parsed_score", "reward"],
            data=[[s["sample_id"], s["true_risk"], s["generation"], s["parsed_score"], s["reward"]] for s in final_samples],
        )})
        run.finish()


if __name__ == "__main__":
    main()
