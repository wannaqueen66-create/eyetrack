#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from datetime import datetime

import pandas as pd
from PIL import Image


def is_formal_scene_folder(folder: str) -> bool:
    csvs = glob.glob(os.path.join(folder, "*.csv"))
    jsons = glob.glob(os.path.join(folder, "*.json"))
    imgs = []
    for ext in ("png", "jpg", "jpeg", "webp"):
        imgs += glob.glob(os.path.join(folder, f"*.{ext}"))
    return (len(csvs) > 0) and (len(jsons) > 0) and (len(imgs) > 0)


def pick_bg(scene_dir: str) -> str | None:
    imgs = []
    for ext in ("png", "jpg", "jpeg", "webp"):
        imgs += glob.glob(os.path.join(scene_dir, f"*.{ext}"))
    if not imgs:
        return None
    return max(imgs, key=lambda p: os.path.getsize(p))


def concat_if_exists(paths: list[str], out_csv: str) -> bool:
    dfs = []
    for p in paths:
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    if not dfs:
        return False
    pd.concat(dfs, ignore_index=True).to_csv(out_csv, index=False)
    return True


def run_batch(repo_dir: str, group_manifest: str, min_valid_ratio: str, aoi_json_mode: str, scenes_root_filtered: str, outdir: str, w: int, h: int, include_scenes: list[str]):
    tmp_root = f"/content/scenes_tmp_{w}x{h}"
    shutil.rmtree(tmp_root, ignore_errors=True)
    os.makedirs(tmp_root, exist_ok=True)

    for s in include_scenes:
        src = os.path.join(scenes_root_filtered, s)
        dst = os.path.join(tmp_root, s)
        if not os.path.exists(dst):
            os.symlink(src, dst)

    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/batch_aoi_metrics.py",
        "--group_manifest", group_manifest,
        "--scenes_root", tmp_root,
        "--aoi_json_mode", aoi_json_mode,
        "--unmatched_csv", "error",
        "--outdir", outdir,
        "--dwell_mode", "fixation",
        "--point_source", "fixation",
        "--screen_w", str(w),
        "--screen_h", str(h),
        "--image_match", "error",
        "--require_validity",
        "--min_valid_ratio", min_valid_ratio,
        "--report_time_segments",
        "--report_class_overlap",
        "--export_aoi_overlay",
    ]

    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_dir, check=True)


def main():
    ap = argparse.ArgumentParser(description="Colab-friendly AOI batch pipeline for mixed-size scenes")
    ap.add_argument("--scenes_root_orig", default="/content/drive/MyDrive/映射")
    ap.add_argument("--group_manifest", default=None)
    ap.add_argument("--repo_dir", default="/content/eyetrack")
    ap.add_argument("--aoi_json_mode", default="image_stem")
    ap.add_argument("--min_valid_ratio", default="0.6")
    ap.add_argument("--filtered_root", default="/content/scenes_aoi_filtered")
    args = ap.parse_args()

    group_manifest = args.group_manifest or os.path.join(args.scenes_root_orig, "group_manifest.csv")
    if not os.path.exists(args.repo_dir):
        raise SystemExit(f"REPO_DIR not found: {args.repo_dir}")
    if not os.path.exists(os.path.join(args.repo_dir, "scripts", "batch_aoi_metrics.py")):
        raise SystemExit("Missing scripts/batch_aoi_metrics.py")
    if not os.path.exists(os.path.join(args.repo_dir, "scripts", "optimize_merged_aoi_outputs.py")):
        raise SystemExit("Missing scripts/optimize_merged_aoi_outputs.py")
    if not os.path.exists(group_manifest):
        raise SystemExit(f"Missing group_manifest: {group_manifest}")

    run_tag = datetime.now().strftime("AOI输出_%Y%m%d_%H%M%S")
    out_root = os.path.join(args.scenes_root_orig, run_tag)
    os.makedirs(out_root, exist_ok=True)
    out_merged = os.path.join(out_root, "输出结果_AOI_合并")

    print("OUT_ROOT:", out_root)

    shutil.rmtree(args.filtered_root, ignore_errors=True)
    os.makedirs(args.filtered_root, exist_ok=True)

    picked = []
    for name in sorted(os.listdir(args.scenes_root_orig)):
        p = os.path.join(args.scenes_root_orig, name)
        if not os.path.isdir(p):
            continue
        if is_formal_scene_folder(p):
            picked.append(name)
            dst = os.path.join(args.filtered_root, name)
            if not os.path.exists(dst):
                os.symlink(p, dst)

    print("Picked scenes:", len(picked))

    size_to_scenes: dict[tuple[int, int], list[str]] = {}
    for s in picked:
        bg = pick_bg(os.path.join(args.filtered_root, s))
        if bg is None:
            continue
        w, h = Image.open(bg).size
        size_to_scenes.setdefault((w, h), []).append(s)

    print("Size groups:", {k: len(v) for k, v in size_to_scenes.items()})

    size_outdirs = []
    for (w, h), scenes in sorted(size_to_scenes.items()):
        outdir = os.path.join(out_root, f"输出结果_AOI_{w}x{h}")
        size_outdirs.append(outdir)
        run_batch(args.repo_dir, group_manifest, args.min_valid_ratio, args.aoi_json_mode, args.filtered_root, outdir, w, h, scenes)

    shutil.rmtree(out_merged, ignore_errors=True)
    os.makedirs(out_merged, exist_ok=True)

    class_files = [os.path.join(d, "batch_aoi_metrics_by_class.csv") for d in size_outdirs]
    poly_files = [os.path.join(d, "batch_aoi_metrics_by_polygon.csv") for d in size_outdirs]
    ok_class = concat_if_exists(class_files, os.path.join(out_merged, "batch_aoi_metrics_by_class.csv"))
    ok_poly = concat_if_exists(poly_files, os.path.join(out_merged, "batch_aoi_metrics_by_polygon.csv"))

    for name in ["timestamp_segments_summary.csv", "batch_aoi_class_overlap.csv", "batch_exclusion_log.csv"]:
        files = [os.path.join(d, name) for d in size_outdirs]
        concat_if_exists(files, os.path.join(out_merged, name))

    overlay_dst = os.path.join(out_merged, "aoi_overlays")
    os.makedirs(overlay_dst, exist_ok=True)
    for d in size_outdirs:
        src = os.path.join(d, "aoi_overlays")
        if not os.path.isdir(src):
            continue
        for fn in os.listdir(src):
            s = os.path.join(src, fn)
            t = os.path.join(overlay_dst, fn)
            if os.path.exists(t):
                stem, ext = os.path.splitext(fn)
                t = os.path.join(overlay_dst, f"{stem}__dup{ext}")
            shutil.copy2(s, t)

    merged_plot_dir = os.path.join(out_merged, "metric_plots")
    os.makedirs(merged_plot_dir, exist_ok=True)
    for src_root in [os.path.join(d, "metric_plots") for d in size_outdirs]:
        if not os.path.isdir(src_root):
            continue
        for root, _, files in os.walk(src_root):
            rel = os.path.relpath(root, src_root)
            out_sub = merged_plot_dir if rel == "." else os.path.join(merged_plot_dir, rel)
            os.makedirs(out_sub, exist_ok=True)
            for fn in files:
                s = os.path.join(root, fn)
                t = os.path.join(out_sub, fn)
                if os.path.exists(t):
                    stem, ext = os.path.splitext(fn)
                    t = os.path.join(out_sub, f"{stem}__dup{ext}")
                shutil.copy2(s, t)

    if ok_class:
        cmd = [
            sys.executable,
            "scripts/optimize_merged_aoi_outputs.py",
            "--merged_outdir", out_merged,
            "--group_manifest", group_manifest,
        ]
        print("\nRunning:", " ".join(cmd))
        subprocess.run(cmd, cwd=args.repo_dir, check=True)

    print("\n=== DONE ===")
    print("Merged class csv:", ok_class)
    print("Merged polygon csv:", ok_poly)
    print("OUT_MERGED:", out_merged)


if __name__ == "__main__":
    main()
