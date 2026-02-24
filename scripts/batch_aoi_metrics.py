#!/usr/bin/env python3
import argparse
import os
import sys

# Ensure repo root is on sys.path so `import src.*` works when running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.aoi_metrics import load_aoi_json, compute_metrics
from src.filters import filter_by_screen_and_validity


def main():
    ap = argparse.ArgumentParser(description='Batch AOI metrics for multiple participants/scenes')
    ap.add_argument('--manifest', required=True, help='CSV with columns: participant_id,scene_id,csv_path,aoi_path')
    ap.add_argument('--assume_clean', action='store_true', help='If set, skip screen/validity filtering regardless of flags (useful if input CSVs already cleaned)')
    ap.add_argument('--outdir', default='outputs_batch')
    ap.add_argument('--dwell_mode', default='row', choices=['row', 'fixation'], help="Dwell time aggregation: 'row' (legacy) or 'fixation' (dedup by Fixation Index)")
    ap.add_argument('--point_source', default='gaze', choices=['gaze', 'fixation'], help="AOI hit testing source: gaze (default) or fixation (Fixation Point X/Y)")
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
    ap.add_argument('--columns_map', default=None, help="Path to JSON mapping of required columns to candidate names (default: use configs/columns_default.json)")
    ap.add_argument('--screen_w', type=int, default=None, help='Optional screen width for coordinate filtering')
    ap.add_argument('--screen_h', type=int, default=None, help='Optional screen height for coordinate filtering')
    ap.add_argument('--require_validity', action='store_true', help='If set, enforce Validity Left/Right == 1 (only if those columns exist)')

    # Image size validation (aoi.json may include image width/height)
    ap.add_argument('--image_match', default='error', choices=['error', 'warn', 'ignore'], help="If aoi.json provides image width/height and you pass --screen_w/--screen_h: what to do on mismatch (default: error)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    m = pd.read_csv(args.manifest)

    req = {'participant_id', 'scene_id', 'csv_path', 'aoi_path'}
    miss = req - set(m.columns)
    if miss:
      raise ValueError(f'Manifest missing columns: {sorted(miss)}')

    all_poly, all_class = [], []

    for _, r in m.iterrows():
        df = pd.read_csv(r['csv_path'], encoding='utf-8-sig')
        for c in df.columns:
            if df[c].dtype == 'object':
                df[c] = df[c].astype(str).str.strip()

        # Column mapping (default map unless overridden)
        from src.columns import load_columns_map, rename_df_columns_inplace
        cmap = load_columns_map(args.columns_map)
        rename_df_columns_inplace(df, cmap)

        # Optional filtering to align with pipeline cleaning
        if not args.assume_clean:
            df = filter_by_screen_and_validity(
                df,
                screen_w=args.screen_w,
                screen_h=args.screen_h,
                require_validity=args.require_validity,
            )

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

        poly_df.insert(0, 'scene_id', r['scene_id'])
        poly_df.insert(0, 'participant_id', r['participant_id'])
        class_df.insert(0, 'scene_id', r['scene_id'])
        class_df.insert(0, 'participant_id', r['participant_id'])

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

    # Save run config for reproducibility
    cfg_path = os.path.join(args.outdir, 'run_config.json')
    try:
        import json
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'manifest': args.manifest,
                    'image_match': args.image_match,
                    'dwell_mode': args.dwell_mode,
                    'point_source': args.point_source,
                    'dwell_empty_as_zero': bool(args.dwell_empty_as_zero),
                    'trial_start_ms': args.trial_start_ms,
                    'trial_start_col': args.trial_start_col,
                    'warn_class_overlap': bool(warn_overlap),
                    'diagnostics': (all_poly[0].attrs.get('diagnostics', {}) if all_poly else {}),
                    'time_segments': args.time_segments,
                    'time_segment_gap_ms': args.time_segment_gap_ms,
                    'report_class_overlap': bool(args.report_class_overlap),
                    'screen_w': args.screen_w,
                    'screen_h': args.screen_h,
                    'require_validity': bool(args.require_validity),
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


if __name__ == '__main__':
    main()
