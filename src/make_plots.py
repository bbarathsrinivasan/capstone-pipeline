"""Two diagnostic charts for results/findings.md, using the validated
reference palette (categorical slots 1/2 for the two experiment conditions,
status colors reserved for the safety-metric reference bands, sequential
blue for the confusion-matrix heatmap)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"     # categorical slot 1 -- "default" condition
ORANGE = "#eb6834"   # categorical slot 2 -- "with hrs" condition
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": GRID,
    "axes.labelcolor": SECONDARY_INK,
    "text.color": INK,
    "xtick.color": SECONDARY_INK,
    "ytick.color": SECONDARY_INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})


def chart_leakage_demo():
    results = json.load(open(RESULTS / "baseline_results.json"))
    models = ["majority", "logreg", "xgboost"]
    model_labels = ["Majority\nclass", "Logistic\nregression", "XGBoost"]

    default = results["pooled_default"]
    leaked = results["pooled_with_hrs_leakage_demo"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    for ax, metric, title, band in [
        (axes[0], "severe_recall", "Severe Recall", (0.40, 0.55, 0.70)),
        (axes[1], "critical_failure_rate", "Critical Failure Rate\n(true Severe -> predicted Minimal/Mild)", None),
    ]:
        x = np.arange(len(models))
        w = 0.32
        v_default = [default[m][metric]["mean"] for m in models]
        v_leaked = [leaked[m][metric]["mean"] for m in models]

        ax.bar(x - w / 2, v_default, width=w, color=BLUE, label="Default (no hrs)")
        ax.bar(x + w / 2, v_leaked, width=w, color=ORANGE, label="With hrs (leakage)")

        for xi, v in zip(x - w / 2, v_default):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=SECONDARY_INK)
        for xi, v in zip(x + w / 2, v_leaked):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=SECONDARY_INK)

        if band:
            floor_lo, floor_hi, target = band
            ax.axhspan(floor_lo, floor_hi, color=STATUS_GOOD, alpha=0.08)
            ax.axhline(target, color=STATUS_GOOD, linestyle="--", linewidth=1)
            ax.text(len(models) - 0.5, target + 0.02, "plan target >= 0.70", fontsize=8, color=STATUS_GOOD)
            ax.text(len(models) - 0.5, floor_lo + 0.01, "honest floor 0.40-0.55", fontsize=8, color=STATUS_GOOD)

        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=9)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_ylim(0, 1.08)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(-0.02, 0.86))
    fig.suptitle("`hrs` field: feature vs. near-total label leakage (pooled, subject-grouped 5x5 CV)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(RESULTS / "chart_hrs_leakage.png", dpi=160)
    plt.close(fig)


def _confusion_matrix_chart(preds_path, out_name, title):
    preds = json.load(open(RESULTS / preds_path))
    y_true = np.array(preds["xgboost"]["y_true"])
    y_pred = np.array(preds["xgboost"]["y_pred"])
    labels = [0, 1, 2, 3]
    label_names = ["Minimal", "Mild", "Moderate", "Severe"]

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE), vmin=0, vmax=1)

    for i in range(4):
        for j in range(4):
            val = cm[i, j]
            txt_color = "#ffffff" if val > 0.55 else INK
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10, color=txt_color)

    ax.set_xticks(range(4)); ax.set_xticklabels(label_names, fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels(label_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9.5)
    ax.set_ylabel("True", fontsize=9.5)
    ax.set_title(title, fontsize=10, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(RESULTS / out_name, dpi=160)
    plt.close(fig)


def chart_confusion_matrix():
    _confusion_matrix_chart(
        "pooled_default_predictions.json", "chart_confusion_matrix.png",
        "XGBoost, pooled default features (598 rows, all tiers)\nrow-normalized confusion matrix (all 25 CV folds pooled)",
    )


def chart_clean_confusion_matrix():
    _confusion_matrix_chart(
        "clean_predictions.json", "chart_clean_confusion_matrix.png",
        "XGBoost, clean pipeline (tier-1 GLOBEM only, 480 rows)\nrow-normalized confusion matrix (all 25 CV folds pooled)",
    )


if __name__ == "__main__":
    chart_leakage_demo()
    chart_confusion_matrix()
    chart_clean_confusion_matrix()
    print("Wrote chart_hrs_leakage.png, chart_confusion_matrix.png, chart_clean_confusion_matrix.png to results/")
