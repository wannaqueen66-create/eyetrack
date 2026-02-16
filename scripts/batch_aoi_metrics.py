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
    ap.add_argument('--columns_map', default=None, help="Path to JSON mapping of required columns to candidate names (default: use configs/columns_default.json)")
    ap.add_argument('--screen_w', type=int, default=None, help='Optional screen width for coordinate filtering')
    ap.add_argument('--screen_h', type=int, default=None, help='Optional screen height for coordinate filtering')
    ap.add_argument('--require_validity', action='store_true', help='If set, enforce Validity Left/Right == 1 (only if those columns exist)')
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

        aois = load_aoi_json(r['aoi_path'])
        poly_df, class_df = compute_metrics(df, aois, dwell_mode=args.dwell_mode)

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

    print('Saved:')
    print(' -', poly_path)
    print(' -', class_path)


if __name__ == '__main__':
    main()
