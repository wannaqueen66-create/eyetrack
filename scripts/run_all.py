#!/usr/bin/env python3
"""Unified end-to-end runner for eyetrack.

This script is a convenience wrapper that calls existing scripts in order.
It avoids adding heavy dependencies; it just orchestrates the workflow.

Typical flow:
  1) run_pipeline.py (clean + quality + heatmap + scanpath)
  2) run_aoi_metrics.py (polygon/class AOI metrics)
  3) merge_scene_features.py (optional)
  4) mixed_effects_indoor_pingpong.py (optional)
  5) paper_figures_indoor_pingpong.py (optional)

You can run only part of the flow with flags.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / 'scripts'


def run(cmd, cwd=None):
    print('[run]', ' '.join(shlex.quote(str(x)) for x in cmd))
    subprocess.check_call(cmd, cwd=cwd)


def main():
    ap = argparse.ArgumentParser(description='Run eyetrack end-to-end workflow')

    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument('--input_csv', help='Raw eye-tracking CSV (single file mode)')
    src_group.add_argument('--manifest', help='Batch manifest CSV (columns: participant_id,scene_id,csv_path,aoi_path)')

    ap.add_argument('--aoi_json', default=None, help='aoi.json exported from eyetrack-aoi (single file mode)')

    ap.add_argument('--workdir', default='outputs_run_all', help='Base output directory')
    ap.add_argument('--screen_w', type=int, default=1280)
    ap.add_argument('--screen_h', type=int, default=1440)
    ap.add_argument('--columns_map', default=None, help='Custom columns mapping JSON (optional)')

    ap.add_argument('--dwell_mode', default='fixation', choices=['row', 'fixation'])

    # Control which stages to run
    ap.add_argument('--skip_pipeline', action='store_true')
    ap.add_argument('--skip_aoi', action='store_true')
    ap.add_argument('--skip_merge', action='store_true')
    ap.add_argument('--skip_model', action='store_true')
    ap.add_argument('--skip_figures', action='store_true')

    # Optional downstream files
    ap.add_argument('--scene_features_csv', default=None, help='Scene features CSV (for merge/model/figures)')

    # In batch mode, pipeline/merge/model/figures are not run automatically (AOI batch is the focus).
    ap.add_argument('--batch_only', action='store_true', help='Batch mode helper: only run batch AOI metrics (no extra filtering flags added by run_all)')
    ap.add_argument('--batch_filter', action='store_true', help='In batch mode, also apply --screen_w/--screen_h and --require_validity in batch_aoi_metrics')

    args = ap.parse_args()

    base = Path(args.workdir)
    base.mkdir(parents=True, exist_ok=True)

    out_pipeline = base / 'pipeline'
    out_aoi = base / 'aoi'
    out_model = base / 'model'
    out_fig = base / 'figures'
    out_analysis = base / 'analysis_table.csv'

    # ---- Batch mode: manifest ----
    if args.manifest:
        # Focus on batch AOI metrics; other stages are scene-specific and not auto-run here.
        cmd = [
            sys.executable, str(SCRIPTS / 'batch_aoi_metrics.py'),
            '--manifest', args.manifest,
            '--outdir', str(out_aoi),
            '--dwell_mode', args.dwell_mode,
        ]
        if args.columns_map:
            cmd += ['--columns_map', args.columns_map]

        if args.batch_filter:
            cmd += ['--screen_w', str(args.screen_w), '--screen_h', str(args.screen_h)]
            cmd += ['--require_validity']
        # default: no extra filtering flags (batch inputs may be heterogeneous)

        run(cmd)
        print('Done (batch). Outputs in:', str(base))
        return

    # ---- Single file mode ----
    if not args.input_csv or not args.aoi_json:
        raise SystemExit('Single file mode requires --input_csv and --aoi_json')

    # 1) pipeline
    if not args.skip_pipeline:
        cmd = [
            sys.executable, str(SCRIPTS / 'run_pipeline.py'),
            '--input', args.input_csv,
            '--outdir', str(out_pipeline),
            '--screen_w', str(args.screen_w),
            '--screen_h', str(args.screen_h),
        ]
        if args.columns_map:
            cmd += ['--columns_map', args.columns_map]
        run(cmd)

    # 2) AOI metrics
    if not args.skip_aoi:
        cmd = [
            sys.executable, str(SCRIPTS / 'run_aoi_metrics.py'),
            '--csv', args.input_csv,
            '--aoi', args.aoi_json,
            '--outdir', str(out_aoi),
            '--dwell_mode', args.dwell_mode,
            '--screen_w', str(args.screen_w),
            '--screen_h', str(args.screen_h),
            '--require_validity',
        ]
        if args.columns_map:
            cmd += ['--columns_map', args.columns_map]
        run(cmd)

    # 3) merge scene features (optional)
    if (not args.skip_merge) or (not args.skip_model) or (not args.skip_figures):
        if not args.scene_features_csv:
            raise SystemExit('scene_features_csv is required for merge/model/figures stages')

    if not args.skip_merge:
        cmd = [
            sys.executable, str(SCRIPTS / 'merge_scene_features.py'),
            '--aoi_class_csv', str(out_aoi / 'aoi_metrics_by_class.csv'),
            '--scene_features_csv', args.scene_features_csv,
            '--out_csv', str(out_analysis),
        ]
        run(cmd)

    # 4) mixed effects model (optional)
    if not args.skip_model:
        cmd = [
            sys.executable, str(SCRIPTS / 'mixed_effects_indoor_pingpong.py'),
            '--analysis_csv', str(out_analysis),
            '--outdir', str(out_model),
        ]
        run(cmd)

    # 5) figures (optional)
    if not args.skip_figures:
        cmd = [
            sys.executable, str(SCRIPTS / 'paper_figures_indoor_pingpong.py'),
            '--analysis_csv', str(out_analysis),
            '--outdir', str(out_fig),
        ]
        run(cmd)

    print('Done. Outputs in:', str(base))


if __name__ == '__main__':
    main()
