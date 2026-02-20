#!/usr/bin/env python3
"""Batch heatmap generation (no AOI) for multiple eye-tracking CSV files.

Typical Colab usage:
  1) Upload csvs.zip
  2) unzip to batch_csvs/
  3) python scripts/batch_heatmap.py --input_dir batch_csvs --screen_w 1920 --screen_h 1080

Outputs:
  <outdir>/<file_stem>/heatmap.png
  <outdir>/batch_quality_report.csv
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Ensure repo root is on sys.path so `import src.*` works when running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import load_and_clean, plot_heatmap, quality_report


def iter_csv_files(input_dir: Path, pattern: str):
    # Use pathlib glob to support ** patterns
    for p in sorted(input_dir.glob(pattern)):
        if p.is_file() and p.suffix.lower() == ".csv":
            yield p


def safe_stem(p: Path) -> str:
    # Avoid empty stems / weird paths
    stem = p.stem.strip() or "unnamed"
    return stem


def main():
    ap = argparse.ArgumentParser(description="Batch generate gaze heatmaps for many CSVs (no AOI)")
    ap.add_argument("--input_dir", required=True, help="Directory that contains many CSV files (can have subfolders)")
    ap.add_argument("--pattern", default="**/*.csv", help="Glob pattern under input_dir (default: **/*.csv)")
    ap.add_argument("--outdir", default="outputs_batch_heatmap", help="Output directory")

    ap.add_argument("--screen_w", type=int, default=1280)
    ap.add_argument("--screen_h", type=int, default=1440)

    ap.add_argument(
        "--require_validity",
        action="store_true",
        help="If set, enforce Validity Left/Right == 1 (only when those columns exist). Default is OFF for batch robustness.",
    )
    ap.add_argument(
        "--columns_map",
        default=None,
        help="Path to JSON mapping of required columns to candidate names (default: use configs/columns_default.json)",
    )

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"input_dir not found: {input_dir}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    files = list(iter_csv_files(input_dir, args.pattern))
    if not files:
        raise SystemExit(f"No CSV files found under {input_dir} with pattern={args.pattern}")

    for fp in files:
        name = safe_stem(fp)
        one_out = outdir / name
        one_out.mkdir(parents=True, exist_ok=True)

        try:
            df, clean = load_and_clean(
                str(fp),
                screen_w=args.screen_w,
                screen_h=args.screen_h,
                require_validity=args.require_validity,
                columns_map_path=args.columns_map,
            )
            plot_heatmap(clean, str(one_out / "heatmap.png"), args.screen_w, args.screen_h)

            qr = quality_report(df, clean)
            qr.update({"file": str(fp), "outdir": str(one_out)})
            rows.append(qr)
        except Exception as e:
            rows.append({"file": str(fp), "outdir": str(one_out), "error": repr(e)})

    report = pd.DataFrame(rows)
    report_path = outdir / "batch_quality_report.csv"
    report.to_csv(report_path, index=False)

    print("Saved:")
    print(" -", str(report_path))
    print(" - heatmaps under:", str(outdir))


if __name__ == "__main__":
    main()
