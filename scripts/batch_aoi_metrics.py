#!/usr/bin/env python3
import argparse
import os
import sys

# Ensure repo root is on sys.path so `import src.*` works when running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.aoi_metrics import load_aoi_json, compute_metrics
from src.filters import filter_by_screen_and_validity, compute_valid_mask


def main():
    ap = argparse.ArgumentParser(description='Batch AOI metrics for multiple participants/scenes')

    # Input mode A: explicit AOI batch manifest
    ap.add_argument('--manifest', default=None, help='CSV with columns: participant_id,scene_id,csv_path,aoi_path')

    # Input mode B: group manifest + scene folders (auto-build the AOI batch manifest)
    ap.add_argument('--group_manifest', default=None, help='CSV with at least column: name (participant id). Extra columns (e.g., SportFreq/Experience) are allowed.')
    ap.add_argument('--scenes_root', default=None, help='Root directory containing scene subfolders. Each scene folder should contain: background image (png/jpg), AOI json, and multiple participant CSVs.')
    ap.add_argument('--aoi_json_mode', default='image_stem', choices=['image_stem', 'aoi_json', 'auto'], help="How to find AOI json inside each scene folder: image_stem=<bg_stem>.json (default), aoi_json=aoi.json, auto=any *.json")
    ap.add_argument('--unmatched_csv', default='skip', choices=['skip', 'error'], help='What to do when a CSV filename cannot be matched to any participant in group_manifest (default: skip)')
    ap.add_argument('--assume_clean', action='store_true', help='If set, skip screen/validity filtering regardless of flags (useful if input CSVs already cleaned)')
    ap.add_argument('--outdir', default='outputs_batch')
    ap.add_argument('--dwell_mode', default='row', choices=['row', 'fixation'], help="Dwell time aggregation: 'row' (legacy) or 'fixation' (dedup by Fixation Index)")
    ap.add_argument('--point_source', default='fixation', choices=['gaze', 'fixation'], help="AOI hit testing source: fixation (default) or gaze. Use fixation to align with fixation-based dwell/TTFF.")
    ap.add_argument('--dwell_empty_as_zero', action='store_true', help='If set, dwell_time_ms will be 0.0 (instead of NaN) when visited==0')

    # TTFF t0 control
    ap.add_argument('--trial_start_ms', type=float, default=None, help='Optional trial start timestamp (ms). If set, TTFF_ms = first_hit_ts - trial_start_ms')
    ap.add_argument('--trial_start_col', default=None, help='Optional column name used to derive trial start (t0 = min(col)). Used only if --trial_start_ms is not set.')

    # AOI overlap warning / report
    ap.add_argument('--warn_class_overlap', action='store_true', help='If set, print warnings when different AOI classes overlap in screen space')
    ap.add_argument('--no_warn_class_overlap', action='store_true', help='Disable class-overlap warnings')
    ap.add_argument('--report_class_overlap', action='store_true', help='If set, write batch_aoi_class_overlap.csv into outdir when overlap exists')

    # Multi-trial / timestamp discontinuity checks
    ap.add_argument('--time_segments', default='warn', choices=['warn', 'error', 'ignore'], help='Policy when multiple timestamp segments detected (default: warn)')
    ap.add_argument('--time_segment_gap_ms', type=float, default=5000.0, help='Gap threshold (ms) to split segments when timestamp jumps forward (default: 5000)')
    ap.add_argument('--report_time_segments', action='store_true', help='If set, write timestamp_segments_summary.csv into outdir')
    ap.add_argument('--columns_map', default=None, help="Path to JSON mapping of required columns to candidate names (default: use configs/columns_default.json)")
    ap.add_argument('--screen_w', type=int, default=None, help='Optional screen width for coordinate filtering')
    ap.add_argument('--screen_h', type=int, default=None, help='Optional screen height for coordinate filtering')
    ap.add_argument('--require_validity', action='store_true', help='If set, enforce Validity Left/Right == 1 (only if those columns exist)')
    ap.add_argument('--min_valid_ratio', type=float, default=None, help='Optional trial-level minimum valid ratio (0-1). If provided, will mark trial_excluded when below threshold and write batch_exclusion_log.csv')

    # Image size validation (aoi.json may include image width/height)
    ap.add_argument('--image_match', default='error', choices=['error', 'warn', 'ignore'], help="If aoi.json provides image width/height and you pass --screen_w/--screen_h: what to do on mismatch (default: error)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Resolve input table (explicit manifest OR auto-build from group_manifest + scenes_root)
    if args.manifest:
        m = pd.read_csv(args.manifest)
    else:
        if not args.group_manifest or not args.scenes_root:
            raise SystemExit('Provide either --manifest OR (--group_manifest and --scenes_root).')

        gm = pd.read_csv(args.group_manifest)
        if 'name' not in gm.columns:
            raise SystemExit('group_manifest must contain column: name')
        names = [str(x).strip() for x in gm['name'].tolist()]
        names_set = set(names)

        scenes_root = args.scenes_root
        if not os.path.isdir(scenes_root):
            raise SystemExit(f'--scenes_root not found or not a dir: {scenes_root}')

        IMG_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}
        SKIP_CSV = {os.path.basename(args.group_manifest), 'aoi_batch_manifest.csv'}

        def pick_bg(scene_dir):
            imgs = []
            for fn in os.listdir(scene_dir):
                p = os.path.join(scene_dir, fn)
                if not os.path.isfile(p):
                    continue
                ext = os.path.splitext(fn)[1].lower()
                if ext in IMG_EXTS:
                    imgs.append(p)
            if not imgs:
                return None
            return max(imgs, key=lambda p: os.path.getsize(p))

        def find_aoi_json(scene_dir, bg_path):
            if args.aoi_json_mode == 'aoi_json':
                p = os.path.join(scene_dir, 'aoi.json')
                return p if os.path.exists(p) else None
            if args.aoi_json_mode == 'auto':
                js = [os.path.join(scene_dir, fn) for fn in os.listdir(scene_dir) if fn.lower().endswith('.json')]
                # prefer aoi.json if exists
                for p in js:
                    if os.path.basename(p).lower() == 'aoi.json':
                        return p
                return js[0] if js else None
            # default: image_stem
            stem = os.path.splitext(os.path.basename(bg_path))[0]
            p = os.path.join(scene_dir, f'{stem}.json')
            return p if os.path.exists(p) else None

        def match_participant_id_from_filename(csv_name):
            # Try match any name as substring in filename
            for nm in names:
                if nm and (nm in csv_name):
                    return nm
            return None

        rows = []
        problems = []
        for scene_id in sorted(os.listdir(scenes_root)):
            scene_dir = os.path.join(scenes_root, scene_id)
            if not os.path.isdir(scene_dir):
                continue

            bg = pick_bg(scene_dir)
            if not bg:
                problems.append((scene_id, 'NO_BG_IMAGE', 'no image found'))
                continue
            aoi_path = find_aoi_json(scene_dir, bg)
            if not aoi_path:
                problems.append((scene_id, 'NO_AOI_JSON', f'mode={args.aoi_json_mode}'))
                continue

            # collect CSVs
            csvs = [fn for fn in os.listdir(scene_dir) if fn.lower().endswith('.csv') and fn not in SKIP_CSV]
            if not csvs:
                problems.append((scene_id, 'NO_CSV', 'no csv found'))
                continue

            for fn in sorted(csvs):
                pid = match_participant_id_from_filename(fn)
                if pid is None:
                    if args.unmatched_csv == 'error':
                        raise SystemExit(f'Unmatched CSV filename under scene={scene_id}: {fn}. Cannot match any name in group_manifest.')
                    else:
                        continue
                rows.append({
                    'participant_id': pid,
                    'scene_id': scene_id,
                    'csv_path': os.path.join(scene_dir, fn),
                    'aoi_path': aoi_path,
                })

        if problems:
            print('[WARN] Scene scan issues (these scenes may be skipped):')
            for p in problems[:50]:
                print('  -', p)
            if len(problems) > 50:
                print('  ... total problems:', len(problems))

        m = pd.DataFrame(rows)
        if len(m) == 0:
            raise SystemExit('Auto-built manifest is empty. Check scenes_root structure and unmatched CSV policy.')

        # Save the auto-built manifest for auditability
        auto_path = os.path.join(args.outdir, 'aoi_batch_manifest_autobuilt.csv')
        m.to_csv(auto_path, index=False, encoding='utf-8-sig')
        print('Auto-built AOI manifest saved:', auto_path)

    req = {'participant_id', 'scene_id', 'csv_path', 'aoi_path'}
    miss = req - set(m.columns)
    if miss:
        raise ValueError(f'Manifest missing columns: {sorted(miss)}')

    all_poly, all_class = [], []
    time_rows = []
    exclusions = []

    for _, r in m.iterrows():
        df = pd.read_csv(r['csv_path'], encoding='utf-8-sig')
        for c in df.columns:
            if df[c].dtype == 'object':
                df[c] = df[c].astype(str).str.strip()

        # Column mapping (default map unless overridden)
        from src.columns import load_columns_map, rename_df_columns_inplace
        cmap = load_columns_map(args.columns_map)
        rename_df_columns_inplace(df, cmap)

        # Trial-level valid ratio (auditable inclusion/exclusion)
        x_col = 'Fixation Point X[px]' if args.point_source == 'fixation' else 'Gaze Point X[px]'
        y_col = 'Fixation Point Y[px]' if args.point_source == 'fixation' else 'Gaze Point Y[px]'
        valid_mask = compute_valid_mask(
            df,
            screen_w=args.screen_w,
            screen_h=args.screen_h,
            require_validity=args.require_validity,
            x_col=x_col,
            y_col=y_col,
        )
        n_total = int(len(df))
        n_valid = int(valid_mask.sum()) if n_total else 0
        valid_ratio = (n_valid / n_total) if n_total else None
        trial_excluded = False
        exclusion_reason = None
        if args.min_valid_ratio is not None and valid_ratio is not None and valid_ratio < float(args.min_valid_ratio):
            trial_excluded = True
            exclusion_reason = f"valid_ratio<{float(args.min_valid_ratio)}"

        # Optional filtering to align with pipeline cleaning
        if not args.assume_clean:
            df = df[valid_mask].copy()

        # Timestamp continuity diagnostics (multi-trial protection)
        time_diag = {}
        if 'Recording Time Stamp[ms]' in df.columns:
            ts = pd.to_numeric(df['Recording Time Stamp[ms]'], errors='coerce').to_numpy()
            ts = ts[np.isfinite(ts)]
            if ts.size >= 2:
                dif = ts[1:] - ts[:-1]
                neg = int((dif < 0).sum())
                gap_thr = float(args.time_segment_gap_ms) if args.time_segment_gap_ms is not None else None
                gap = int((dif > gap_thr).sum()) if gap_thr is not None else 0
                segments = 1 + neg + gap
                time_diag = {
                    'neg_jumps': neg,
                    'gap_jumps': gap,
                    'gap_threshold_ms': gap_thr,
                    'segments_estimated': int(segments),
                }
                if segments > 1:
                    msg = f"Detected multiple timestamp segments for csv_path={r['csv_path']} (segments={segments}, neg_jumps={neg}, gap_jumps={gap}, gap_thr_ms={gap_thr}). TTFF may be unreliable unless you set --trial_start_ms/--trial_start_col or pre-split trials."
                    if args.time_segments == 'error':
                        raise SystemExit(msg)
                    elif args.time_segments == 'warn':
                        print('[WARN]', msg)

        # Optional image size validation
        from src.aoi_metrics import load_aoi_json_meta
        meta = load_aoi_json_meta(r['aoi_path'])
        img = meta.get('image') if isinstance(meta, dict) else None
        if img and (args.screen_w is not None) and (args.screen_h is not None):
            iw = img.get('width')
            ih = img.get('height')
            if isinstance(iw, (int, float)) and isinstance(ih, (int, float)):
                iw, ih = int(iw), int(ih)
                if (iw != int(args.screen_w)) or (ih != int(args.screen_h)):
                    msg = f"AOI image size mismatch for aoi_path={r['aoi_path']}: aoi.json image=({iw},{ih}) vs screen=({args.screen_w},{args.screen_h})."
                    if args.image_match == 'error':
                        raise SystemExit(msg)
                    elif args.image_match == 'warn':
                        print('[WARN]', msg)

        aois = load_aoi_json(r['aoi_path'])
        warn_overlap = True
        if args.no_warn_class_overlap:
            warn_overlap = False
        elif args.warn_class_overlap:
            warn_overlap = True

        poly_df, class_df = compute_metrics(
            df,
            aois,
            dwell_mode=args.dwell_mode,
            point_source=args.point_source,
            dwell_empty_as_zero=args.dwell_empty_as_zero,
            trial_start_ms=args.trial_start_ms,
            trial_start_col=args.trial_start_col,
            warn_class_overlap=warn_overlap,
        )
        # merge in time diagnostics
        try:
            diag = poly_df.attrs.get('diagnostics', {})
            if isinstance(diag, dict):
                diag = {**diag, 'time_segments': time_diag}
                poly_df.attrs['diagnostics'] = diag
                class_df.attrs['diagnostics'] = diag
        except Exception:
            pass

        # Add trial-level audit fields
        for _df in (poly_df, class_df):
            _df.insert(0, 'exclusion_reason', exclusion_reason)
            _df.insert(0, 'trial_excluded', int(trial_excluded))
            _df.insert(0, 'min_valid_ratio', args.min_valid_ratio)
            _df.insert(0, 'valid_ratio', valid_ratio)
            _df.insert(0, 'n_valid_rows', n_valid)
            _df.insert(0, 'n_total_rows', n_total)

        poly_df.insert(0, 'scene_id', r['scene_id'])
        poly_df.insert(0, 'participant_id', r['participant_id'])
        class_df.insert(0, 'scene_id', r['scene_id'])
        class_df.insert(0, 'participant_id', r['participant_id'])

        # Exclusion log row (even if not excluded, if threshold provided)
        if args.min_valid_ratio is not None:
            exclusions.append({
                'participant_id': r['participant_id'],
                'scene_id': r['scene_id'],
                'csv_path': r['csv_path'],
                'valid_ratio': valid_ratio,
                'min_valid_ratio': float(args.min_valid_ratio),
                'trial_excluded': int(trial_excluded),
                'exclusion_reason': exclusion_reason,
                'n_total_rows': n_total,
                'n_valid_rows': n_valid,
            })

        if args.report_time_segments:
            time_rows.append({
                'participant_id': r['participant_id'],
                'scene_id': r['scene_id'],
                'csv_path': r['csv_path'],
                **time_diag,
            })

        all_poly.append(poly_df)
        all_class.append(class_df)

    poly_all = pd.concat(all_poly, ignore_index=True) if all_poly else pd.DataFrame()
    class_all = pd.concat(all_class, ignore_index=True) if all_class else pd.DataFrame()

    poly_path = os.path.join(args.outdir, 'batch_aoi_metrics_by_polygon.csv')
    class_path = os.path.join(args.outdir, 'batch_aoi_metrics_by_class.csv')
    poly_all.to_csv(poly_path, index=False)
    class_all.to_csv(class_path, index=False)

    # Optional overlap report
    if args.report_class_overlap:
        overlap_rows = []
        for pdf in all_poly:
            diag = pdf.attrs.get('diagnostics', {})
            overlaps = diag.get('class_overlap', []) if isinstance(diag, dict) else []
            if not overlaps:
                continue
            pid = str(pdf['participant_id'].iloc[0]) if 'participant_id' in pdf.columns and len(pdf) else None
            sid = str(pdf['scene_id'].iloc[0]) if 'scene_id' in pdf.columns and len(pdf) else None
            for d in overlaps:
                overlap_rows.append({
                    'participant_id': pid,
                    'scene_id': sid,
                    **d,
                })
        if overlap_rows:
            odf = pd.DataFrame(overlap_rows)
            odf.to_csv(os.path.join(args.outdir, 'batch_aoi_class_overlap.csv'), index=False)

    # Optional time segment report
    if args.report_time_segments and time_rows:
        pd.DataFrame(time_rows).to_csv(os.path.join(args.outdir, 'timestamp_segments_summary.csv'), index=False)

    # Optional exclusion log
    if args.min_valid_ratio is not None and exclusions:
        pd.DataFrame(exclusions).to_csv(os.path.join(args.outdir, 'batch_exclusion_log.csv'), index=False)

    # Save run config for reproducibility
    cfg_path = os.path.join(args.outdir, 'run_config.json')
    try:
        import json
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'manifest': args.manifest,
                    'group_manifest': args.group_manifest,
                    'scenes_root': args.scenes_root,
                    'aoi_json_mode': args.aoi_json_mode,
                    'unmatched_csv': args.unmatched_csv,
                    'image_match': args.image_match,
                    'dwell_mode': args.dwell_mode,
                    'point_source': args.point_source,
                    'point_source_note': "AOI hit-testing used x/y columns based on point_source (fixation uses Fixation Point X/Y; gaze uses Gaze Point X/Y).",
                    'dwell_empty_as_zero': bool(args.dwell_empty_as_zero),
                    'trial_start_ms': args.trial_start_ms,
                    'trial_start_col': args.trial_start_col,
                    'warn_class_overlap': bool(warn_overlap),
                    'diagnostics': (all_poly[0].attrs.get('diagnostics', {}) if all_poly else {}),
                    'time_segments': args.time_segments,
                    'time_segment_gap_ms': args.time_segment_gap_ms,
                    'report_class_overlap': bool(args.report_class_overlap),
                    'report_time_segments': bool(args.report_time_segments),
                    'screen_w': args.screen_w,
                    'screen_h': args.screen_h,
                    'require_validity': bool(args.require_validity),
                    'min_valid_ratio': args.min_valid_ratio,
                    'assume_clean': bool(args.assume_clean),
                    'columns_map': args.columns_map,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception:
        pass

    print('Saved:')
    print(' -', poly_path)
    print(' -', class_path)
    if not args.manifest:
        print(' -', os.path.join(args.outdir, 'aoi_batch_manifest_autobuilt.csv'))


if __name__ == '__main__':
    main()
