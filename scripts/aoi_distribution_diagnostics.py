#!/usr/bin/env python3
"""AOI distribution diagnostics (paper-friendly).

This script generates quick diagnostics for common AOI outcomes:
- dwell_time_ms (often right-skewed)
- TTFF_ms (defined only when visited==1)
- fixation_count (count; often overdispersed)

It saves histograms (raw + log1p) and a small CSV summary including skewness.

Input can be either:
- --analysis_csv (preferred): merged AOI + predictors table
- or --aoi_class_csv: outputs from run_aoi_metrics/batch_aoi_metrics

"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    ap.add_argument("--analysis_csv", default=None, help="Merged analysis table (AOI + predictors)")
    ap.add_argument("--aoi_class_csv", default=None, help="AOI metrics by class CSV")
    ap.add_argument("--outdir", default="outputs_aoi_diagnostics")
    ap.add_argument("--only_class", default=None, help="Optional AOI class_name filter")
    args = ap.parse_args()

    if not args.analysis_csv and not args.aoi_class_csv:
        raise SystemExit("Provide --analysis_csv or --aoi_class_csv")

    path = args.analysis_csv or args.aoi_class_csv
    df = pd.read_csv(path)

    if args.only_class and "class_name" in df.columns:
        df = df[df["class_name"] == args.only_class].copy()

    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    targets = [
        ("dwell_time_ms", None),
        ("TTFF_ms", "visited"),
        ("fixation_count", None),
    ]

    for col, visit_col in targets:
        if col not in df.columns:
            continue
        sub = df
        if visit_col and visit_col in df.columns:
            sub = df[df[visit_col] == 1]

        x = pd.to_numeric(sub[col], errors="coerce")
        x = x[np.isfinite(x)]

        raw_png = os.path.join(args.outdir, f"hist_{col}_raw.png")
        _hist(x, raw_png, f"{col} (raw)  n={len(x)}")

        log_png = os.path.join(args.outdir, f"hist_{col}_log1p.png")
        _hist(np.log1p(x), log_png, f"log1p({col})  n={len(x)}")

        rows.append(
            {
                "metric": col,
                "n": int(len(x)),
                "mean": float(x.mean()) if len(x) else np.nan,
                "median": float(np.median(x)) if len(x) else np.nan,
                "skew": _skew(x),
                "suggest_transform": "log1p" if _skew(x) > 1.0 else "none",
            }
        )

    out_csv = os.path.join(args.outdir, "aoi_distribution_summary.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # A short markdown note for papers
    md = os.path.join(args.outdir, "NOTE_methods_distribution.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# AOI outcome distribution diagnostics\n\n")
        f.write(f"Input: `{path}`\n\n")
        f.write("Generated histograms (raw and log1p) and `aoi_distribution_summary.csv`.\n\n")
        f.write("Typical reporting guidance (align with this repo):\n")
        f.write("- `TTFF_ms` is modeled on trials with `visited==1` (two-part approach).\n")
        f.write("- `dwell_time_ms` is often right-skewed; log1p transform is commonly appropriate.\n")
        f.write("- `fixation_count` is a count variable; consider Poisson/NB models (overdispersion check).\n")

    print("Saved to", args.outdir)


if __name__ == "__main__":
    main()
