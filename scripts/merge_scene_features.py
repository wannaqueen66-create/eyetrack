#!/usr/bin/env python3
import argparse
import os
import sys

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

    req_aoi_base = {'participant_id', 'scene_id', 'class_name', 'TFD', 'FC', 'TTFF'}
    if not req_aoi_base.issubset(set(aoi.columns)):
        raise ValueError(f'aoi csv missing columns: {sorted(req_aoi_base - set(aoi.columns))}')
    req_feat = {'participant_id', 'scene_id'}
    if not req_feat.issubset(set(feat.columns)):
        raise ValueError(f'scene feature csv missing columns: {sorted(req_feat - set(feat.columns))}')

    out = aoi.merge(feat, on=['participant_id', 'scene_id'], how='left')
    if 'dwell_time_ms' not in out.columns and 'TFD' in out.columns:
        out['dwell_time_ms'] = out['TFD']
    if 'fixation_count' not in out.columns and 'FC' in out.columns:
        out['fixation_count'] = out['FC']
    if 'RF' not in out.columns and 'RFF' in out.columns:
        out['RF'] = out['RFF']
    out.to_csv(args.out_csv, index=False)
    print('Saved:', args.out_csv)


if __name__ == '__main__':
    main()
