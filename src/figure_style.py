from __future__ import annotations

import math

import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = {
    "blue": "#6F97BD",
    "orange": "#E3A86F",
    "green": "#8FB7A1",
    "purple": "#A99AC6",
    "gray": "#A8B2BC",
    "ink": "#243447",
    "grid": "#E5ECF2",
    "muted": "#6B7C8F",
    "light_blue": "#E8F1F8",
    "light_orange": "#FAEBDD",
    "light_green": "#E9F3ED",
    "light_red": "#FBE8E4",
}

METRIC_LABELS = {
    "FC": "Fixation Count (FC)",
    "FC_share": "Fixation-count share within trial (FC_share)",
    "FC_prop": "Fixation-count proportion within trial (FC_prop)",
    "FC_rate": "Fixation-count rate (FC/s)",
    "TTFF": "Time to First Fixation (TTFF, ms)",
    "FFD": "First Fixation Duration (FFD, ms)",
    "TFD": "Total Fixation Duration (TFD, ms)",
    "share": "Attention allocation share (TFD share)",
    "share_pct": "Attention allocation share (%)",
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
        "axes.titlecolor": PALETTE["ink"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.grid": False,
        "grid.color": PALETTE["grid"],
        "grid.alpha": 0.65,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.8,
        "legend.title_fontsize": 9,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "savefig.dpi": 320,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "lines.linewidth": 1.9,
        "lines.markersize": 5.2,
        "patch.linewidth": 0.8,
        "figure.titleweight": "semibold",
    })


def soften_axes(ax, grid_axis: str = "y"):
    ax.spines["left"].set_color(PALETTE["ink"])
    ax.spines["bottom"].set_color(PALETTE["ink"])
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", colors=PALETTE["ink"], length=3)
    ax.grid(axis=grid_axis, color=PALETTE["grid"], alpha=0.7, linewidth=0.7)
    other = "x" if grid_axis == "y" else "y"
    ax.grid(axis=other, visible=False)


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def metric_value_label(metric: str, value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    value = float(value)
    if metric == "share_pct":
        return f"{value:.1f}%"
    if metric in {"TFD", "TTFF", "FFD", "MFD"}:
        return f"{value:.0f}"
    if metric in {"FC", "RFF"}:
        return f"{value:.1f}"
    if metric == "MPD":
        return f"{value:.2f}"
    return f"{value:.2f}"


def reserve_right_text_space(ax, x0: float, x1: float, frac: float = 0.24):
    span = max(float(x1) - float(x0), 1.0)
    ax.set_xlim(float(x0) - 0.05 * span, float(x1) + frac * span)


def choose_sparse_label_indices(values, max_labels: int = 4) -> list[int]:
    vals = [(i, float(v)) for i, v in enumerate(values) if v is not None and math.isfinite(float(v))]
    if not vals:
        return []
    if len(vals) <= max_labels:
        return [i for i, _ in vals]
    ordered = sorted(vals, key=lambda t: t[0])
    idx = {ordered[0][0], ordered[-1][0]}
    vmax = max(vals, key=lambda t: t[1])[0]
    vmin = min(vals, key=lambda t: t[1])[0]
    idx.update([vmax, vmin])
    # prefer peak / trough / endpoints; only add midpoint when we still have budget
    if len(idx) < max_labels and len(ordered) >= 5:
        idx.add(ordered[len(ordered) // 2][0])
    return sorted(idx)[:max_labels]


def annotate_series_smart(ax, xs, ys, metric: str, color: str, max_labels: int = 4):
    pairs = [(i, float(x), float(y)) for i, (x, y) in enumerate(zip(xs, ys)) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if not pairs:
        return
    keep = set(choose_sparse_label_indices([p[2] for p in pairs], max_labels=max_labels))
    ymin, ymax = ax.get_ylim()
    span = max(float(ymax) - float(ymin), 1.0)
    placed = []
    for j, (src_i, x, y) in enumerate(pairs):
        if src_i not in keep:
            continue
        dy = 8 if j % 2 == 0 else -10
        y_try = y + (0.03 * span if dy >= 0 else -0.03 * span)
        for px, py in placed:
            if abs(x - px) < 0.35 and abs(y_try - py) < 0.08 * span:
                dy = dy + 10 if dy >= 0 else dy - 10
                y_try = y + (0.05 * span if dy >= 0 else -0.05 * span)
        ax.annotate(
            metric_value_label(metric, y),
            xy=(x, y),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom" if dy >= 0 else "top",
            fontsize=6.8,
            color=color,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.92),
            zorder=5,
        )
        placed.append((x, y_try))


def annotate_right_ci_labels(ax, y_positions, ci_high, labels, color: str = "#334155", pad_frac: float = 0.03):
    vals = [float(v) for v in ci_high if v is not None and math.isfinite(float(v))]
    if not vals:
        return
    xmin = min(vals)
    xmax = max(vals)
    span = max(xmax - xmin, 1.0)
    x_text = xmax + pad_frac * span
    reserve_right_text_space(ax, xmin, xmax, frac=max(0.22, pad_frac * 8.0))
    for y, lab in zip(y_positions, labels):
        if str(lab).strip():
            ax.text(x_text, y, str(lab), va="center", ha="left", fontsize=7, color=color)
