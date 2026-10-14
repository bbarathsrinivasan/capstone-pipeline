"""
Reward functions for GRPO, per TimeSRL (arXiv:2605.21295) and the plan's
safety-shaped variant (Fall plan §4, "primary contribution").

TimeSRL's reward (RLVR -- reinforcement learning from verifiable rewards):
    r(p, y) = exp(-(p - y)^2 / (2 * sigma^2)),  sigma = 1.2
A Gaussian centered on the true label, treating the 0-3 risk scale as a
continuous target: closer numeric predictions score higher, with no credit
for an unparseable/missing prediction.

This project's stated contribution over TimeSRL is to weight the clinically
dangerous error (true Severe predicted as Minimal/Mild) harder than generic
distance. TimeSRL's own paper doesn't need this -- it never discusses an
asymmetric safety cost -- so there's no published formula to copy; the
version below is this pipeline's own construction, and should be reviewed
by whoever owns the reward-function design (Rishi, per the plan's Sep 20
action items) rather than taken as settled.
"""
import math
import re

SIGMA = 1.2
SEVERE = 3
SAFE_LEVELS = {0, 1}
CRITICAL_FAILURE_MULTIPLIER = 0.15  # how hard a Severe->Safe miss is punished

SCORE_PATTERN = re.compile(r"risk\s*score\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_predicted_score(generated_text):
    """Pull the numeric risk score out of a generation. None if unparseable
    or out of the valid [0, 3] range -- both count as a format failure,
    not a wrong-but-valid prediction."""
    matches = SCORE_PATTERN.findall(generated_text)
    if not matches:
        return None
    try:
        p = float(matches[-1])  # last match wins if the model repeats itself
    except ValueError:
        return None
    if not (0 <= p <= 3):
        return None
    return p


def timesrl_reward(p, y):
    """Plain TimeSRL Gaussian-decay reward. p=None -> 0 (format failure)."""
    if p is None:
        return 0.0
    return math.exp(-(p - y) ** 2 / (2 * SIGMA ** 2))


def safety_shaped_reward(p, y):
    """TimeSRL reward, with an extra multiplicative penalty specifically for
    the clinically dangerous miss: true Severe (y=3) predicted as Minimal/Mild
    (p<=1.5). Every other error is scored identically to timesrl_reward."""
    if p is None:
        return 0.0
    base = math.exp(-(p - y) ** 2 / (2 * SIGMA ** 2))
    if y == SEVERE and p <= 1.5:
        return base * CRITICAL_FAILURE_MULTIPLIER
    return base


def reward_batch(generated_texts, true_ints, variant="safety_shaped"):
    """variant: 'timesrl' or 'safety_shaped'. Returns (rewards, parsed_scores)."""
    fn = timesrl_reward if variant == "timesrl" else safety_shaped_reward
    parsed = [parse_predicted_score(t) for t in generated_texts]
    rewards = [fn(p, y) for p, y in zip(parsed, true_ints)]
    return rewards, parsed


if __name__ == "__main__":
    # Quick demo of the reward SHAPE across prediction error, for Severe vs
    # non-Severe true labels -- this is the clearest way to see what the
    # safety-shaping actually does before trusting it inside RL training.
    print(f"{'true_y':<8}{'pred_p':<8}{'timesrl':<10}{'safety_shaped':<14}")
    for y in [0, 3]:
        for p in [y, y + 1, y - 1, 0, 3]:
            if not (0 <= p <= 3):
                continue
            print(f"{y:<8}{p:<8}{timesrl_reward(p, y):<10.3f}{safety_shaped_reward(p, y):<14.3f}")
