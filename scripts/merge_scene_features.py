#!/usr/bin/env python3
import argparse
import os
import sys

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description='Merge AOI class metrics with scene-level features')
    ap.add_argument('--aoi_class_csv', required=True)
    ap.add_argument('--scene_features_csv', required=True)
    ap.add_argument('--out_csv', default='outputs/analysis_table.csv')
    args = ap.parse_args()

    aoi = pd.read_csv(args.aoi_class_csv)
    feat = pd.read_csv(args.scene_features_csv)

    req_aoi = {'participant_id', 'scene_id', 'class_name', 'dwell_time_ms', 'TTFF_ms', 'fixation_count'}
    req_feat = {'participant_id', 'scene_id'}

    if not req_aoi.issubset(set(aoi.columns)):
        raise ValueError(f'aoi csv missing columns: {sorted(req_aoi - set(aoi.columns))}')
    if not req_feat.issubset(set(feat.columns)):
        raise ValueError(f'scene feature csv missing columns: {sorted(req_feat - set(feat.columns))}')

    out = aoi.merge(feat, on=['participant_id', 'scene_id'], how='left')
    out.to_csv(args.out_csv, index=False)
    print('Saved:', args.out_csv)


if __name__ == '__main__':
    main()
