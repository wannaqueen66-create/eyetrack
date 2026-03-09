from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = {
    "blue": "#6C8EBF",
    "orange": "#E6A96B",
    "green": "#8FB996",
    "purple": "#B8A1C9",
    "gray": "#97A1AA",
    "ink": "#334155",
    "grid": "#D9E2EC",
}

METRIC_LABELS = {
    "FC": "Fixation Count (FC)",
    "TTFF": "Time to First Fixation (TTFF, ms)",
    "TTFF": "Time to First Fixation (TTFF, ms)",
    "FFD": "First Fixation Duration (FFD, ms)",
    "TFD": "Total Fixation Duration (TFD, ms)",
    "MFD": "Mean Fixation Duration (MFD, ms)",
    "RFF": "Re-fixation Frequency (RFF)",
    "MPD": "Mean Pupil Diameter (MPD)",
}


def apply_paper_style():
    sns.set_theme(style="white", context="notebook")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.alpha": 0.55,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.8,
        "lines.markersize": 5.5,
    })


def soften_axes(ax):
    ax.spines["left"].set_color(PALETTE["ink"])
    ax.spines["bottom"].set_color(PALETTE["ink"])
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.grid(axis="y", color=PALETTE["grid"], alpha=0.6, linewidth=0.7)
    ax.grid(axis="x", visible=False)


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)
