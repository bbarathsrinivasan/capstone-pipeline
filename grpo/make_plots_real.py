"""
Plots for the real-model GRPO run (results/grpo_real_run.json) -- the
real-model counterpart to make_plots.py (which plots the from-scratch
mechanics-validation run, train.py's results/grpo_run.json).

Colors/styling follow this project's validated chart palette (categorical
slots assigned in fixed order -- blue, orange, aqua, yellow -- never cycled
or picked ad hoc; recessive gridlines; thin 2px lines on a light chart
surface) rather than matplotlib's defaults.

Run after train_real.py:
    python3 grpo/make_plots_real.py
-> results/chart_real_training.png
"""
from pathlib import Path

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# Validated categorical palette, light mode, fixed slot order (slots 1-4).
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SLOT_BLUE = "#2a78d6"
SLOT_ORANGE = "#eb6834"
SLOT_AQUA = "#1baf7a"
SLOT_YELLOW = "#eda100"


def _style_axis(ax, title, color, steps, values):
    marker = "o" if len(steps) <= 20 else None  # markers clutter past ~20 points
    ax.plot(steps, values, color=color, linewidth=2, marker=marker, markersize=5)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left")
    ax.set_xlabel("step", color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.grid(color=GRIDLINE, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(BASELINE)
    ax.axhline(0, color=BASELINE, linewidth=0.8)


def main():
    with open(RESULTS / "grpo_real_run.json") as f:
        data = json.load(f)
    history = data["history"]
    steps = list(range(1, len(history) + 1))
    reward = [h["reward_mean"] for h in history]
    parse_rate = [h["parse_success_rate"] for h in history]
    kl = [h["kl"] for h in history]
    loss = [h["loss"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=SURFACE)
    fig.suptitle(
        f"GRPO on {data['config']['MODEL_NAME']} -- {len(history)} steps, "
        f"lr={data['config']['LEARNING_RATE']}, group_size={data['config']['GROUP_SIZE']}",
        color=INK_PRIMARY, fontsize=12,
    )

    _style_axis(axes[0, 0], "reward_mean", SLOT_BLUE, steps, reward)
    _style_axis(axes[0, 1], "parse_success_rate", SLOT_ORANGE, steps, parse_rate)
    axes[0, 1].set_ylim(-0.05, 1.05)
    _style_axis(axes[1, 0], "kl (k3 estimator, >= 0 by construction)", SLOT_AQUA, steps, kl)
    _style_axis(axes[1, 1], "loss", SLOT_YELLOW, steps, loss)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = RESULTS / "chart_real_training.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
