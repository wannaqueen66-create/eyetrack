#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime

from colab_scene_scan import print_scan_report


def main():
    ap = argparse.ArgumentParser(description="Colab-friendly research-bundle pipeline for eyetrack")
    ap.add_argument("--scenes_root_orig", default="/content/drive/MyDrive/映射")
    ap.add_argument("--group_manifest", default=None)
    ap.add_argument("--scene_features_csv", default=None, help="Optional; if omitted, run_analysis2.py will auto-generate scene_features.csv from scene folders + AOI JSON")
    ap.add_argument("--repo_dir", default="/content/eyetrack")
    ap.add_argument("--aoi_json_mode", default="image_stem")
    ap.add_argument("--min_valid_ratio", default="0.6")
    args = ap.parse_args()

    group_manifest = args.group_manifest or os.path.join(args.scenes_root_orig, "group_manifest.csv")
    print_scan_report(args.scenes_root_orig, group_manifest)
    if not os.path.exists(group_manifest):
        raise SystemExit(f"Missing group_manifest: {group_manifest}")
    if not os.path.exists(args.repo_dir):
        raise SystemExit(f"Missing repo_dir: {args.repo_dir}")

    run_tag = datetime.now().strftime("研究输出_%Y%m%d_%H%M%S")
    out_root = os.path.join(args.scenes_root_orig, run_tag)
    os.makedirs(out_root, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/run_analysis2.py",
        "--group_manifest", group_manifest,
        "--scenes_root", args.scenes_root_orig,
        "--aoi_json_mode", args.aoi_json_mode,
        "--min_valid_ratio", str(args.min_valid_ratio),
        "--out_root", out_root,
    ]
    if args.scene_features_csv:
        cmd += ["--scene_features_csv", args.scene_features_csv]

    print("即将输出研究总目录 / OUT_ROOT:", out_root)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=args.repo_dir, check=True)
    print("DONE:", out_root)


if __name__ == "__main__":
    main()
