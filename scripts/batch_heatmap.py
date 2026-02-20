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

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

import numpy as np
import matplotlib.pyplot as plt


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

    # Background overlay (optional)
    ap.add_argument("--background_img", default=None, help="Optional background image (png/jpg) to overlay the heatmap")
    ap.add_argument("--alpha", type=float, default=0.55, help="Heatmap overlay alpha on background (default 0.55)")
    ap.add_argument("--bins", type=int, default=200, help="Grid bins for density overlay (default 200x200)")
    ap.add_argument("--sigma", type=float, default=2.0, help="Gaussian smoothing sigma for density overlay (default 2.0)")

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

    def density_from_points(xy: np.ndarray):
        if xy.size == 0:
            return np.zeros((args.bins, args.bins), dtype=float)
        x = np.clip(xy[:, 0], 0, args.screen_w)
        y = np.clip(xy[:, 1], 0, args.screen_h)
        H, _, _ = np.histogram2d(
            x,
            y,
            bins=[args.bins, args.bins],
            range=[[0, args.screen_w], [0, args.screen_h]],
        )
        if gaussian_filter is None:
            raise SystemExit("scipy is required for --background_img overlay mode. Please install requirements.txt")
        H = gaussian_filter(H, sigma=args.sigma, mode="nearest")
        s = H.sum()
        if s > 0:
            H = H / s
        return H

    def save_overlay(H: np.ndarray, out_png: Path, title: str):
        bg = plt.imread(args.background_img)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 8))
        # stretch background to screen coordinate extent
        plt.imshow(bg, extent=[0, args.screen_w, args.screen_h, 0], aspect="auto")
        plt.imshow(H.T, origin="upper", extent=[0, args.screen_w, args.screen_h, 0], cmap="inferno", alpha=args.alpha, aspect="auto")
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()

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

            # default heatmap (no background)
            plot_heatmap(clean, str(one_out / "heatmap.png"), args.screen_w, args.screen_h)

            # optional overlay heatmap on background
            if args.background_img:
                xy = clean[["Gaze Point X[px]", "Gaze Point Y[px]"]].dropna().to_numpy(dtype=float)
                H = density_from_points(xy)
                save_overlay(H, one_out / "heatmap_overlay.png", f"Gaze Heatmap (overlay): {name}")

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
