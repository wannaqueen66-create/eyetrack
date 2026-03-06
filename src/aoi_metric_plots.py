from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def _pick_metric_cols(df: pd.DataFrame) -> List[str]:
    preferred = ["FC", "TTFF", "FFD", "TFD", "MFD", "RF", "MPD"]
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
    out_p.mkdir(parents=True, exist_ok=True)

    d = df.copy()
    d["class_name"] = d["class_name"].astype(str)

    for m in metric_cols:
        y = pd.to_numeric(d[m], errors="coerce")
        if not y.notna().any():
            continue

        fig = plt.figure(figsize=(9, 4.8))
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(d["class_name"], y.fillna(0.0), color="#4C78A8", alpha=0.9)
        ax.set_title(f"AOI {m} by class")
        ax.set_xlabel("AOI class")
        ax.set_ylabel(m)
        ax.tick_params(axis='x', labelrotation=30)
        fig.tight_layout()

        p = out_p / f"{prefix}_{m}.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        out_paths.append(str(p))

    return out_paths
