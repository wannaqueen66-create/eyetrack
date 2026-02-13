#!/usr/bin/env python3
import argparse
import os
import pandas as pd
from src.aoi_metrics import load_aoi_json, compute_metrics


def main():
    ap = argparse.ArgumentParser(description='Compute AOI metrics by polygon and class')
    ap.add_argument('--csv', required=True, help='Eye-tracking CSV path')
    ap.add_argument('--aoi', required=True, help='aoi.json path')
    ap.add_argument('--outdir', default='outputs')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv, encoding='utf-8-sig')
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = df[c].astype(str).str.strip()

    aois = load_aoi_json(args.aoi)
    poly_df, class_df = compute_metrics(df, aois)

    poly_path = os.path.join(args.outdir, 'aoi_metrics_by_polygon.csv')
    class_path = os.path.join(args.outdir, 'aoi_metrics_by_class.csv')
    poly_df.to_csv(poly_path, index=False)
    class_df.to_csv(class_path, index=False)

    print('Saved:')
    print(' -', poly_path)
    print(' -', class_path)


if __name__ == '__main__':
    main()
