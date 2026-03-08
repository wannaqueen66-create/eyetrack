#!/usr/bin/env python3
"""Research-bundle orchestrator for eyetrack.

Goal
----
Produce a stable, paper-oriented research bundle for the eye-tracking repo,
while preserving the content expectations that were previously associated with an `analysis-2` output folder.

Pipeline
--------
1) batch_aoi_metrics.py
2) optimize_aoi_outputs.py
3) summarize_aoi_by_condition_group.py
4) model_aoi_lmm_allocation.py
5) model_aoi_two_part.py (optional, when scene features are available)
6) aoi_distribution_diagnostics.py

Outputs
-------
results/research_bundle/
  task1/  grouped descriptive summaries + organized outputs
  task2/  LMM allocation models
  task3/  two-part models (if analysis table can be built)
  diagnostics/
  colab/
  reports/
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def run(cmd: list[str]):
    print("[run]", " ".join(shlex.quote(str(x)) for x in cmd))
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))


def main():
    ap = argparse.ArgumentParser(description="Run eyetrack research bundle")
    ap.add_argument("--group_manifest", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scenes_root", help="Scene-root mode for fresh batch run")
    src.add_argument("--batch_class_csv", help="Reuse existing batch_aoi_metrics_by_class.csv")

    ap.add_argument("--batch_polygon_csv", default=None, help="Optional existing batch_aoi_metrics_by_polygon.csv when reusing batch outputs")
    ap.add_argument("--scene_features_csv", default=None, help="Optional scene feature table for two-part/modeling extensions")
    ap.add_argument("--group_id_col", default="name")
    ap.add_argument("--aoi_json_mode", default="image_stem")
    ap.add_argument("--screen_w", type=int, default=None)
    ap.add_argument("--screen_h", type=int, default=None)
    ap.add_argument("--min_valid_ratio", type=float, default=0.6)
    ap.add_argument("--out_root", default="results/research_bundle")
    args = ap.parse_args()

    out_root = REPO_ROOT / args.out_root
    task1 = out_root / "task1"
    task2 = out_root / "task2"
    task3 = out_root / "task3"
    diag = out_root / "diagnostics"
    colab = out_root / "colab"
    reports = out_root / "reports"
    for p in [task1, task2, task3, diag, colab, reports]:
        p.mkdir(parents=True, exist_ok=True)

    if args.scenes_root:
        batch_out = out_root / "_batch_raw"
        cmd = [
            sys.executable,
            str(SCRIPTS / "batch_aoi_metrics.py"),
            "--group_manifest", args.group_manifest,
            "--scenes_root", args.scenes_root,
            "--aoi_json_mode", args.aoi_json_mode,
            "--unmatched_csv", "error",
            "--outdir", str(batch_out),
            "--dwell_mode", "fixation",
            "--point_source", "fixation",
            "--require_validity",
            "--min_valid_ratio", str(args.min_valid_ratio),
            "--report_time_segments",
            "--report_class_overlap",
            "--export_aoi_overlay",
        ]
        if args.screen_w is not None and args.screen_h is not None:
            cmd += ["--screen_w", str(args.screen_w), "--screen_h", str(args.screen_h), "--image_match", "error"]
        run(cmd)
        class_csv = batch_out / "batch_aoi_metrics_by_class.csv"
        poly_csv = batch_out / "batch_aoi_metrics_by_polygon.csv"
    else:
        class_csv = Path(args.batch_class_csv)
        poly_csv = Path(args.batch_polygon_csv) if args.batch_polygon_csv else None
        if not class_csv.exists():
            raise SystemExit(f"Missing batch_class_csv: {class_csv}")

    # Task1: organized outputs + descriptive summaries
    cmd = [
        sys.executable,
        str(SCRIPTS / "optimize_aoi_outputs.py"),
        "--aoi_class_csv", str(class_csv),
        "--group_manifest", args.group_manifest,
        "--group_id_col", args.group_id_col,
        "--outdir", str(task1 / "organized_outputs"),
    ]
    if poly_csv and poly_csv.exists():
        cmd += ["--aoi_polygon_csv", str(poly_csv)]
    run(cmd)

    run([
        sys.executable,
        str(SCRIPTS / "summarize_aoi_by_condition_group.py"),
        "--aoi_class_csv", str(class_csv),
        "--group_manifest", args.group_manifest,
        "--group_id_col", args.group_id_col,
        "--outdir", str(task1 / "condition_group_summary"),
        "--include_round",
    ])

    # Task2: allocation LMM
    run([
        sys.executable,
        str(SCRIPTS / "model_aoi_lmm_allocation.py"),
        "--aoi_class_csv", str(class_csv),
        "--group_manifest", args.group_manifest,
        "--group_id_col", args.group_id_col,
        "--outdir", str(task2 / "allocation_lmm"),
    ])

    # Diagnostics
    run([
        sys.executable,
        str(SCRIPTS / "aoi_distribution_diagnostics.py"),
        "--aoi_class_csv", str(class_csv),
        "--outdir", str(diag / "distribution"),
    ])

    # Optional Task3: two-part with merged scene features
    if args.scene_features_csv:
        analysis_csv = out_root / "analysis2_analysis_table.csv"
        run([
            sys.executable,
            str(SCRIPTS / "merge_scene_features.py"),
            "--aoi_class_csv", str(class_csv),
            "--scene_features_csv", args.scene_features_csv,
            "--out_csv", str(analysis_csv),
        ])
        run([
            sys.executable,
            str(SCRIPTS / "model_aoi_two_part.py"),
            "--analysis_csv", str(analysis_csv),
            "--outdir", str(task3 / "two_part_models"),
            "--log1p_tff",
            "--log1p_tfd",
        ])

    (reports / "README.txt").write_text(
        "Canonical research bundle for eyetrack.\n"
        "Keeps the content expected from the former analysis-2 folder, but under a more generic bundle-oriented naming scheme.\n",
        encoding="utf-8",
    )

    # Colab helper note
    (colab / "README.txt").write_text(
        "Use scripts/run_colab_analysis2_pipeline.py in Colab for mixed-size scene folders.\n"
        "This research_bundle directory is the canonical eyetrack output bundle.\n",
        encoding="utf-8",
    )

    print("Saved research bundle to:", out_root)


if __name__ == "__main__":
    main()
