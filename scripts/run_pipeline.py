#!/usr/bin/env python3
import argparse
import os
import sys

# Ensure repo root is on sys.path so `import src.*` works when running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import load_and_clean, quality_report, plot_heatmap, plot_scanpath, compute_aoi_metrics
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--screen_w", type=int, default=1280)
    ap.add_argument("--screen_h", type=int, default=1440)
    ap.add_argument("--no_validity", action="store_true", help="Do not require Validity Left/Right == 1 (if those columns exist)")
    ap.add_argument("--columns_map", default=None, help="Path to JSON mapping of required columns to candidate names (default: configs/columns_default.json)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df, clean = load_and_clean(
        args.input,
        args.screen_w,
        args.screen_h,
        require_validity=(not args.no_validity),
        columns_map_path=args.columns_map,
    )

    qr = quality_report(df, clean)
    pd.DataFrame([qr]).to_csv(os.path.join(args.outdir, "quality_report.csv"), index=False)

    plot_heatmap(clean, os.path.join(args.outdir, "heatmap.png"), args.screen_w, args.screen_h)
    plot_scanpath(clean, os.path.join(args.outdir, "scanpath.png"), args.screen_w, args.screen_h)

    # Demo rectangular AOIs for indoor pingpong scene quick check
    aois = {
        "table_zone": (0, 0, args.screen_w // 3, args.screen_h),
        "player_zone": (args.screen_w // 3, 0, 2 * args.screen_w // 3, args.screen_h),
        "background_zone": (2 * args.screen_w // 3, 0, args.screen_w, args.screen_h),
    }
    aoi = compute_aoi_metrics(clean, aois)
    aoi.to_csv(os.path.join(args.outdir, "aoi_metrics.csv"), index=False)

    print("Done:", args.outdir)


if __name__ == "__main__":
    main()
