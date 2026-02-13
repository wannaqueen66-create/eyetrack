#!/usr/bin/env python3
import argparse
import os
from src.pipeline import load_and_clean, quality_report, plot_heatmap, plot_scanpath, compute_aoi_metrics
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--screen_w", type=int, default=1280)
    ap.add_argument("--screen_h", type=int, default=1440)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df, clean = load_and_clean(args.input, args.screen_w, args.screen_h)

    qr = quality_report(df, clean)
    pd.DataFrame([qr]).to_csv(os.path.join(args.outdir, "quality_report.csv"), index=False)

    plot_heatmap(clean, os.path.join(args.outdir, "heatmap.png"), args.screen_w, args.screen_h)
    plot_scanpath(clean, os.path.join(args.outdir, "scanpath.png"), args.screen_w, args.screen_h)

    aois = {
        "Left": (0, 0, args.screen_w // 3, args.screen_h),
        "Center": (args.screen_w // 3, 0, 2 * args.screen_w // 3, args.screen_h),
        "Right": (2 * args.screen_w // 3, 0, args.screen_w, args.screen_h),
    }
    aoi = compute_aoi_metrics(clean, aois)
    aoi.to_csv(os.path.join(args.outdir, "aoi_metrics.csv"), index=False)

    print("Done:", args.outdir)


if __name__ == "__main__":
    main()
