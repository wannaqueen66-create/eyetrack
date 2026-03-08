from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from .figure_style import apply_paper_style, soften_axes, metric_label, PALETTE


def _pick_metric_cols(df: pd.DataFrame) -> List[str]:
    preferred = ["FC", "TFF", "FFD", "TFD", "MFD", "RFF", "MPD"]
    out = [c for c in preferred if c in df.columns]
    return out


def export_metric_barplots(df: pd.DataFrame, outdir: str, prefix: str = "aoi_class") -> list[str]:
    """Export one PNG per metric (x=class_name, y=metric).

    Designed for per-file class-level AOI results.
    """
    out_paths: list[str] = []
    if df is None or len(df) == 0 or ("class_name" not in df.columns):
        return out_paths

    metric_cols = _pick_metric_cols(df)
    if not metric_cols:
        return out_paths

    out_p = Path(outdir)

    d = df.copy()
    d["class_name"] = d["class_name"].astype(str)

    created_dir = False
    for m in metric_cols:
        y = pd.to_numeric(d[m], errors="coerce")
        if not y.notna().any():
            continue

        if not created_dir:
            out_p.mkdir(parents=True, exist_ok=True)
            created_dir = True

        apply_paper_style()
        fig = plt.figure(figsize=(8.6, 4.8))
        ax = fig.add_subplot(1, 1, 1)
        bars = ax.bar(
            d["class_name"],
            y.fillna(0.0),
            color=PALETTE["blue"],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_title(f"AOI {metric_label(m)} by class", pad=10)
        ax.set_xlabel("AOI class")
        ax.set_ylabel(metric_label(m))
        ax.tick_params(axis='x', labelrotation=24)
        soften_axes(ax)

        ymax = y.max(skipna=True)
        for rect, val in zip(bars, y.tolist()):
            if pd.isna(val):
                continue
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, color=PALETTE["ink"])
        if pd.notna(ymax):
            ax.set_ylim(top=float(ymax) * 1.12 if float(ymax) > 0 else 1.0)

        fig.tight_layout()

        p = out_p / f"{prefix}_{m}.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        out_paths.append(str(p))

    return out_paths
