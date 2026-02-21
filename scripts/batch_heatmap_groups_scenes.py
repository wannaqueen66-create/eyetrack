#!/usr/bin/env python3
"""Batch heatmap groups across multiple scene folders.

Use case:
- You have N scene folders (e.g., 12 scenes). Each scene folder contains:
  - A background image (png/jpg/jpeg)
  - Many participant gaze CSVs (one CSV per participant for that scene)
- You want to run `batch_heatmap_groups.py` once per scene, producing
  independent outputs under outdir/<scene_folder_name>/...

This script is a thin wrapper that iterates scene directories and calls
`batch_heatmap_groups.py` for each scene.

Typical Colab usage:
  python scripts/batch_heatmap_groups_scenes.py \
    --manifest /content/group_manifest.csv \
    --scenes_root /content/scenes \
    --screen_w 1748 --screen_h 2064 \
    --outdir /content/outputs_by_scene

Notes:
- CSV matching still follows `batch_heatmap_groups.py` logic (by name in filename
  under --csv_dir). Ensure filenames contain the participant name.
- Background image is auto-detected inside each scene folder.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


IMG_EXTS = (".png", ".jpg", ".jpeg")


def find_background(scene_dir: Path, prefer: str | None = None) -> Path | None:
    if prefer:
        p = scene_dir / prefer
        if p.exists() and p.is_file():
            return p

    imgs = [p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not imgs:
        return None

    # If multiple, pick the largest file as a heuristic (often the real background)
    imgs.sort(key=lambda p: (p.stat().st_size, p.name))
    return imgs[-1]


def main():
    ap = argparse.ArgumentParser(description="Run batch_heatmap_groups.py across scene folders")
    ap.add_argument("--manifest", required=True, help="group_manifest.csv (name,SportFreq,Experience)")
    ap.add_argument("--scenes_root", required=True, help="Directory containing scene folders")
    ap.add_argument("--outdir", required=True, help="Output root; each scene uses a subfolder")
    ap.add_argument("--screen_w", type=int, required=True)
    ap.add_argument("--screen_h", type=int, required=True)

    ap.add_argument("--scene_glob", default="*", help="Glob for scene folders under scenes_root (default: *)")
    ap.add_argument("--background_filename", default=None, help="If provided, use this filename inside each scene folder")

    # forward common plotting params
    ap.add_argument("--bins", type=int, default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--thresh", type=float, default=None)
    ap.add_argument("--cmap", default=None)
    ap.add_argument("--title_mode", default=None)
    ap.add_argument("--quiet_glyph_warning", action="store_true")
    ap.add_argument("--font", default=None)

    # fixation-based heatmaps (forwarded)
    ap.add_argument("--point_source", default=None, choices=["gaze", "fixation"], help="Use gaze points or fixation points")
    ap.add_argument("--weight", default=None, choices=["none", "fixation_duration"], help="Optional weighting")
    ap.add_argument("--fixation_dedup", default=None, choices=["index", "none"], help="How to deduplicate fixation points")

    ap.add_argument("--fail_fast", action="store_true", help="Stop on first failed scene")

    args = ap.parse_args()

    scenes_root = Path(args.scenes_root)
    if not scenes_root.exists():
        raise SystemExit(f"scenes_root not found: {scenes_root}")

    outroot = Path(args.outdir)
    outroot.mkdir(parents=True, exist_ok=True)

    script_groups = Path(__file__).resolve().parent / "batch_heatmap_groups.py"

    scene_dirs = [p for p in scenes_root.glob(args.scene_glob) if p.is_dir()]
    scene_dirs.sort(key=lambda p: p.name)
    if not scene_dirs:
        raise SystemExit(f"No scene folders matched under {scenes_root} with glob={args.scene_glob!r}")

    failed = []

    for sd in scene_dirs:
        bg = find_background(sd, prefer=args.background_filename)
        if bg is None:
            msg = f"No background image found in scene folder: {sd}"
            if args.fail_fast:
                raise SystemExit(msg)
            print("[WARN]", msg, file=sys.stderr)
            failed.append((sd.name, msg))
            continue

        out_scene = outroot / sd.name
        out_scene.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(script_groups),
            "--manifest", str(args.manifest),
            "--csv_dir", str(sd),
            "--screen_w", str(args.screen_w),
            "--screen_h", str(args.screen_h),
            "--background_img", str(bg),
            "--outdir", str(out_scene),
        ]

        # forward params if provided
        def f(flag, v):
            if v is not None:
                cmd.extend([flag, str(v)])

        f("--bins", args.bins)
        f("--sigma", args.sigma)
        f("--alpha", args.alpha)
        f("--thresh", args.thresh)
        f("--cmap", args.cmap)
        f("--title_mode", args.title_mode)
        f("--font", args.font)
        f("--point_source", args.point_source)
        f("--weight", args.weight)
        f("--fixation_dedup", args.fixation_dedup)
        if args.quiet_glyph_warning:
            cmd.append("--quiet_glyph_warning")
        if args.fail_fast:
            cmd.append("--fail_fast")

        print(f"\n== Scene: {sd.name} ==")
        print("csv_dir:", sd)
        print("bg:     ", bg)
        print("outdir: ", out_scene)

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            msg = f"batch_heatmap_groups.py failed for scene={sd.name}: {e}"
            if args.fail_fast:
                raise SystemExit(msg)
            print("[WARN]", msg, file=sys.stderr)
            failed.append((sd.name, msg))

    if failed:
        # write a small summary for convenience
        fail_path = outroot / "failed_scenes.csv"
        with open(fail_path, "w", encoding="utf-8") as f:
            f.write("scene,reason\n")
            for s, r in failed:
                f.write(f"{s},{r.replace(',', ';')}\n")
        print(f"\n[WARN] Some scenes failed. See: {fail_path}")


if __name__ == "__main__":
    main()
