#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import statsmodels.formula.api as smf

from src.aoi_metrics import normalize_aoi_class_series


def fit_and_save(df, formula, group_col, out_txt, title):
    model = smf.mixedlm(formula, data=df, groups=df[group_col])
    res = model.fit(reml=False, method='lbfgs')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(title + '\n\n')
        f.write(str(res.summary()))
    return res


def main():
    ap = argparse.ArgumentParser(description='Scene-feature mixed-effects models for eye-tracking outputs')
    ap.add_argument('--analysis_csv', required=True, help='Merged table from merge_scene_features.py')
    ap.add_argument('--outdir', default='outputs_scene_feature_models')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.analysis_csv)

    if 'class_name' in df.columns:
        df['class_name'] = normalize_aoi_class_series(df['class_name'])
        table_df = df[df['class_name'] == 'table'].copy()
        dfm = table_df if len(table_df) >= 8 else df.copy()
    else:
        dfm = df.copy()

    if 'TFD' not in dfm.columns and 'dwell_time_ms' in dfm.columns:
        dfm['TFD'] = pd.to_numeric(dfm['dwell_time_ms'], errors='coerce')
    if 'FC' not in dfm.columns and 'fixation_count' in dfm.columns:
        dfm['FC'] = pd.to_numeric(dfm['fixation_count'], errors='coerce')

    required = ['participant_id', 'TFD', 'TTFF', 'FC']
    for c in required:
        if c not in dfm.columns:
            raise ValueError(f'Missing required column: {c}')

    predictor_candidates = ['table_density', 'distance_to_table_center_m', 'table_center_offset_ratio', 'illum_lux', 'crowding_level', 'occlusion_ratio', 'aoi_coverage_ratio', 'non_table_aoi_coverage_ratio', 'WWR']
    predictors = [c for c in predictor_candidates if c in dfm.columns]
    if not predictors:
        raise ValueError('No predictor columns found. Provide at least one scene feature predictor (manual or auto-generated).')

    rhs = ' + '.join(predictors)
    fit_and_save(dfm.dropna(subset=['TFD'] + predictors + ['participant_id']), f'TFD ~ {rhs}', 'participant_id', os.path.join(args.outdir, 'model_tfd.txt'), 'MixedLM: TFD')
    fit_and_save(dfm.dropna(subset=['TTFF'] + predictors + ['participant_id']), f'TTFF ~ {rhs}', 'participant_id', os.path.join(args.outdir, 'model_ttff.txt'), 'MixedLM: TTFF')
    fit_and_save(dfm.dropna(subset=['FC'] + predictors + ['participant_id']), f'FC ~ {rhs}', 'participant_id', os.path.join(args.outdir, 'model_fc.txt'), 'MixedLM: FC')
    print('Saved models to', args.outdir)


if __name__ == '__main__':
    main()
