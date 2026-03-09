#!/usr/bin/env python3
"""AOI distribution diagnostics (paper-friendly)."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _skew(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    return float(pd.Series(x).skew())


def _hist(x: pd.Series, path: str, title: str):
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]
    plt.figure(figsize=(6, 4))
    plt.hist(x, bins=40, color="#4477AA", alpha=0.85)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="AOI distribution diagnostics")
    ap.add_argument("--analysis_csv", default=None)
    ap.add_argument("--aoi_class_csv", default=None)
    ap.add_argument("--outdir", default="outputs_aoi_diagnostics")
    ap.add_argument("--only_class", default=None)
    args = ap.parse_args()
    if not args.analysis_csv and not args.aoi_class_csv:
        raise SystemExit("Provide --analysis_csv or --aoi_class_csv")
    path = args.analysis_csv or args.aoi_class_csv
    df = pd.read_csv(path)
    if args.only_class and "class_name" in df.columns:
        df = df[df["class_name"] == args.only_class].copy()
    os.makedirs(args.outdir, exist_ok=True)
    if "TTFF" not in df.columns:
        raise ValueError("Missing required column: TTFF")
    if "TFD" not in df.columns and "dwell_time_ms" in df.columns:
        df["TFD"] = pd.to_numeric(df["dwell_time_ms"], errors="coerce")
    if "FC" not in df.columns and "fixation_count" in df.columns:
        df["FC"] = pd.to_numeric(df["fixation_count"], errors="coerce")
    rows = []
    for col, visit_col in [("TFD", None), ("TTFF", "visited"), ("FC", None)]:
        if col not in df.columns:
            continue
        sub = df[df[visit_col] == 1] if visit_col and visit_col in df.columns else df
        x = pd.to_numeric(sub[col], errors="coerce")
        x = x[np.isfinite(x)]
        _hist(x, os.path.join(args.outdir, f"hist_{col}_raw.png"), f"{col} (raw)  n={len(x)}")
        _hist(np.log1p(x), os.path.join(args.outdir, f"hist_{col}_log1p.png"), f"log1p({col})  n={len(x)}")
        rows.append({"metric": col, "n": int(len(x)), "mean": float(x.mean()) if len(x) else np.nan, "median": float(np.median(x)) if len(x) else np.nan, "skew": _skew(x), "suggest_transform": "log1p" if _skew(x) > 1.0 else "none"})
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "aoi_distribution_summary.csv"), index=False)
    with open(os.path.join(args.outdir, "NOTE_methods_distribution.md"), "w", encoding="utf-8") as f:
        f.write("# AOI outcome distribution diagnostics\n\n")
        f.write(f"Input: `{path}`\n\n")
        f.write("Generated histograms (raw and log1p) and `aoi_distribution_summary.csv`.\n\n")
        f.write("Typical reporting guidance (align with this repo):\n")
        f.write("- `TTFF` is modeled on trials with `visited==1` (two-part approach).\n")
        f.write("- `TFD` is often right-skewed; log1p transform is commonly appropriate.\n")
        f.write("- `FC` is a count variable; consider Poisson/NB models (overdispersion check).\n")
    print("Saved to", args.outdir)


if __name__ == "__main__":
    main()
