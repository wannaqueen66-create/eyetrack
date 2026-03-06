#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description="Optimize merged AOI outputs into organized PNG/table views")
    ap.add_argument("--merged_outdir", required=True, help="Path like .../输出结果_AOI_合并")
    ap.add_argument("--group_manifest", required=True)
    ap.add_argument("--group_id_col", default="name")
    ap.add_argument("--outdir", default=None, help="Default: <merged_outdir>/optimized_outputs")
    ap.add_argument("--skip_if_exists", action="store_true")
    args = ap.parse_args()

    merged_outdir = os.path.abspath(args.merged_outdir)
    class_csv = os.path.join(merged_outdir, "batch_aoi_metrics_by_class.csv")
    poly_csv = os.path.join(merged_outdir, "batch_aoi_metrics_by_polygon.csv")
    outdir = os.path.abspath(args.outdir or os.path.join(merged_outdir, "optimized_outputs"))

    if not os.path.exists(class_csv):
        raise SystemExit(f"Missing merged class CSV: {class_csv}")
    if not os.path.exists(args.group_manifest):
        raise SystemExit(f"Missing group_manifest: {args.group_manifest}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(script_dir, "optimize_aoi_outputs.py")
    if not os.path.exists(target):
        raise SystemExit(f"Missing script: {target}")

    cmd = [
        sys.executable,
        target,
        "--aoi_class_csv", class_csv,
        "--group_manifest", args.group_manifest,
        "--group_id_col", args.group_id_col,
        "--outdir", outdir,
    ]
    if os.path.exists(poly_csv):
        cmd += ["--aoi_polygon_csv", poly_csv]
    if args.skip_if_exists:
        cmd += ["--skip_if_exists"]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Saved optimized outputs to:", outdir)


if __name__ == "__main__":
    main()
