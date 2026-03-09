#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def run(cmd: list[str]) -> None:
    print("[run]", " ".join(shlex.quote(str(x)) for x in cmd))
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))


def pick_first_image(folder: Path) -> Path | None:
    imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not imgs:
        return None
    imgs.sort(key=lambda p: (p.stat().st_size, p.name))
    return imgs[-1]


def detect_image_size(img_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image  # type: ignore
        with Image.open(img_path) as im:
            return int(im.size[0]), int(im.size[1])
    except Exception:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            arr = plt.imread(str(img_path))
            h, w = arr.shape[:2]
            return int(w), int(h)
        except Exception as e:
            raise SystemExit(f"Failed to detect image size for {img_path}: {e}")


def ensure_group_manifest_has_name(group_manifest: Path, out_path: Path) -> Path:
    import pandas as pd

    gm = pd.read_csv(group_manifest)
    if "name" not in gm.columns:
        if "participant_id" in gm.columns:
            gm = gm.copy()
            gm.insert(0, "name", gm["participant_id"])
        else:
            raise SystemExit("group_manifest must contain column 'name' or 'participant_id'.")
    gm.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="One-command AOI batch runner using only: csv_dir + group_manifest.csv + scene image + AOI json"
    )
    ap.add_argument("--csv_dir", required=True, help="Folder with participant eye-tracking CSV files")
    ap.add_argument("--group_manifest", required=True, help="group_manifest.csv with participant ids and groups")
    ap.add_argument("--scene_image", required=True, help="Scene background image (png/jpg/webp)")
    ap.add_argument("--aoi_json", required=True, help="AOI JSON drawn on the same scene image")
    ap.add_argument("--scene_id", default="scene1", help="Logical scene id used in outputs (default: scene1)")
    ap.add_argument("--outdir", default="outputs_minimal_aoi_bundle", help="Output directory")
    ap.add_argument("--columns_map", default=None, help="Optional custom columns mapping JSON")
    ap.add_argument("--require_validity", action="store_true", help="Require Validity Left/Right == 1 when those columns exist")
    ap.add_argument("--min_valid_ratio", type=float, default=0.6, help="Exclude trials below this valid-ratio threshold (default: 0.6)")
    ap.add_argument("--assume_clean", action="store_true", help="Skip screen/validity filtering and trust CSVs as already cleaned")
    ap.add_argument("--time_segments", default="warn", choices=["warn", "error", "ignore"], help="Policy for timestamp discontinuities")
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir).resolve()
    group_manifest = Path(args.group_manifest).resolve()
    scene_image = Path(args.scene_image).resolve()
    aoi_json = Path(args.aoi_json).resolve()
    outdir = Path(args.outdir).resolve()

    if not csv_dir.is_dir():
        raise SystemExit(f"csv_dir not found: {csv_dir}")
    for p, label in [(group_manifest, "group_manifest"), (scene_image, "scene_image"), (aoi_json, "aoi_json")]:
        if not p.exists():
            raise SystemExit(f"Missing {label}: {p}")

    screen_w, screen_h = detect_image_size(scene_image)

    stage_root = outdir / "_staging_one_scene"
    scene_dir = stage_root / args.scene_id
    shutil.rmtree(stage_root, ignore_errors=True)
    scene_dir.mkdir(parents=True, exist_ok=True)

    staged_manifest = ensure_group_manifest_has_name(group_manifest, stage_root / "group_manifest.csv")

    image_target = scene_dir / scene_image.name
    aoi_target = scene_dir / f"{scene_image.stem}.json"
    if image_target.exists() or image_target.is_symlink():
        image_target.unlink()
    if aoi_target.exists() or aoi_target.is_symlink():
        aoi_target.unlink()
    os.symlink(scene_image, image_target)
    os.symlink(aoi_json, aoi_target)

    csv_count = 0
    for p in sorted(csv_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".csv":
            continue
        target = scene_dir / p.name
        if target.exists() or target.is_symlink():
            target.unlink()
        os.symlink(p, target)
        csv_count += 1
    if csv_count == 0:
        raise SystemExit(f"No CSV files found in {csv_dir}")

    batch_out = outdir / "batch_outputs"
    cmd = [
        sys.executable,
        str(SCRIPTS / "batch_aoi_metrics.py"),
        "--group_manifest", str(staged_manifest),
        "--scenes_root", str(stage_root),
        "--aoi_json_mode", "image_stem",
        "--missing_aoi_json", "error",
        "--unmatched_csv", "error",
        "--outdir", str(batch_out),
        "--dwell_mode", "fixation",
        "--point_source", "fixation",
        "--screen_w", str(screen_w),
        "--screen_h", str(screen_h),
        "--image_match", "error",
        "--min_valid_ratio", str(args.min_valid_ratio),
        "--time_segments", str(args.time_segments),
        "--report_time_segments",
        "--report_class_overlap",
        "--export_aoi_overlay",
        "--optimize_outputs",
    ]
    if args.columns_map:
        cmd += ["--columns_map", str(Path(args.columns_map).resolve())]
    if args.require_validity:
        cmd += ["--require_validity"]
    if args.assume_clean:
        cmd += ["--assume_clean"]

    run(cmd)

    summary = {
        "inputs": {
            "csv_dir": str(csv_dir),
            "group_manifest": str(group_manifest),
            "scene_image": str(scene_image),
            "aoi_json": str(aoi_json),
            "scene_id": args.scene_id,
            "csv_count": csv_count,
        },
        "detected_scene_size": {"width": screen_w, "height": screen_h},
        "outputs": {
            "batch_outputs": str(batch_out),
            "class_csv": str(batch_out / "batch_aoi_metrics_by_class.csv"),
            "polygon_csv": str(batch_out / "batch_aoi_metrics_by_polygon.csv"),
            "optimized_outputs": str(batch_out / "optimized_outputs"),
            "aoi_overlays": str(batch_out / "aoi_overlays"),
        },
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved one-command minimal AOI bundle to:", outdir)
    print(" - class csv:", batch_out / "batch_aoi_metrics_by_class.csv")
    print(" - polygon csv:", batch_out / "batch_aoi_metrics_by_polygon.csv")
    print(" - optimized outputs:", batch_out / "optimized_outputs")


if __name__ == "__main__":
    main()
