"""Charts for grpo/results/findings.md, reusing the validated reference
palette (categorical slot 1 blue for the main signal, status colors
reserved for the zero/'format failure' reference line)."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
STATUS_CRITICAL = "#d03b3b"

plt.rcParams.update({
    "font.family": "sans-serif", "axes.edgecolor": GRID, "axes.labelcolor": SECONDARY_INK,
    "text.color": INK, "xtick.color": SECONDARY_INK, "ytick.color": SECONDARY_INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def chart_main_run():
    d = json.load(open(RESULTS / "grpo_run.json"))
    h = d["history"]
    steps = [r["step"] if "step" in r else i + 1 for i, r in enumerate(h)]
    reward = [r["reward_mean"] for r in h]
    parse_rate = [r["parse_success_rate"] for r in h]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, series, title, color in [
        (axes[0], reward, "Mean reward per GRPO step\n(safety-shaped, 0 = format failure)", BLUE),
        (axes[1], parse_rate, "Fraction of rollouts with a parseable\n'Risk score: N' line", BLUE),
    ]:
        ax.plot(steps, series, color=color, marker="o", markersize=4, linewidth=1.8)
        ax.axhline(0, color=GRID, linewidth=1)
        ax.set_ylim(-0.05, max(0.2, max(series) * 1.3 if max(series) > 0 else 0.2))
        ax.set_xlabel("GRPO step", fontsize=9.5)
        ax.set_title(title, fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    fig.suptitle("Main GRPO run: never a single parseable prediction in 12 steps "
                 f"({d['config']['N_STEPS']} steps x {d['config']['PROMPTS_PER_STEP']} prompts x "
                 f"{d['config']['GROUP_SIZE']} rollouts)",
                 fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(RESULTS / "chart_main_run.png", dpi=160)
    plt.close(fig)


def chart_sanity_check():
    d = json.load(open(RESULTS / "sanity_check_run.json"))
    h = d["history"]
    steps = [r["step"] for r in h]
    mean_len = [r["mean_len"] for r in h]
    min_len = [r["min_len"] for r in h]

    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.plot(steps, mean_len, color=BLUE, marker="o", markersize=4, linewidth=1.8, label="mean completion length")
    ax.plot(steps, min_len, color=ORANGE, marker="o", markersize=4, linewidth=1.4, linestyle="--",
             label="shortest in batch")
    ax.set_xlabel("GRPO step", fontsize=9.5)
    ax.set_ylabel("completion length (tokens)", fontsize=9.5)
    ax.set_title("Sanity check: continuous length-based reward\n(confirms the GRPO update code moves the policy)",
                 fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "chart_sanity_check.png", dpi=160)
    plt.close(fig)


def chart_reward_shape():
    import sys
    sys.path.insert(0, str(ROOT))
    from reward import timesrl_reward, safety_shaped_reward

    ps = np.linspace(0, 3, 61)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, y, title in [(axes[0], 0, "True label: Minimal (y=0)"), (axes[1], 3, "True label: Severe (y=3)")]:
        ts = [timesrl_reward(p, y) for p in ps]
        ss = [safety_shaped_reward(p, y) for p in ps]
        ax.plot(ps, ts, color=BLUE, linewidth=2, label="TimeSRL (plain)")
        ax.plot(ps, ss, color=STATUS_CRITICAL, linewidth=2, linestyle="--", label="Safety-shaped")
        ax.axvline(y, color=GRID, linewidth=1)
        ax.set_xlabel("predicted score p", fontsize=9.5)
        ax.set_ylabel("reward", fontsize=9.5)
        ax.set_title(title, fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("Reward shape: safety-shaping only bites when true=Severe and p<=1.5", fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(RESULTS / "chart_reward_shape.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    chart_main_run()
    chart_sanity_check()
    chart_reward_shape()
    print("Wrote chart_main_run.png, chart_sanity_check.png, chart_reward_shape.png to results/")
