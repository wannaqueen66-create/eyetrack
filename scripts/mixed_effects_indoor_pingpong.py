#!/usr/bin/env python3
import argparse
import os
import sys

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import statsmodels.formula.api as smf


def fit_and_save(df, formula, group_col, out_txt, title):
    model = smf.mixedlm(formula, data=df, groups=df[group_col])
    res = model.fit(reml=False, method='lbfgs')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(title + '\n\n')
        f.write(str(res.summary()))
    return res


def main():
    ap = argparse.ArgumentParser(description='Mixed-effects models for indoor pingpong scene eye-tracking')
    ap.add_argument('--analysis_csv', required=True, help='Merged table from merge_scene_features.py')
    ap.add_argument('--outdir', default='outputs_model')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.analysis_csv)

    # Focus on table-related AOI class for this indoor pingpong study
    if 'class_name' in df.columns:
        table_df = df[df['class_name'] == 'pingpong_table'].copy()
        if len(table_df) >= 8:
            dfm = table_df
        else:
            dfm = df.copy()
    else:
        dfm = df.copy()

    required = ['participant_id', 'dwell_time_ms', 'TTFF_ms', 'fixation_count']
    for c in required:
        if c not in dfm.columns:
            raise ValueError(f'Missing required column: {c}')

    # choose available predictors
    predictors = [c for c in ['table_density', 'distance_to_table_center_m', 'illum_lux', 'crowding_level', 'occlusion_ratio'] if c in dfm.columns]
    if not predictors:
        raise ValueError('No predictor columns found. Provide at least one scene feature predictor.')

    rhs = ' + '.join(predictors)

    # Model 1: dwell time
    formula1 = f'dwell_time_ms ~ {rhs}'
    fit_and_save(dfm.dropna(subset=['dwell_time_ms'] + predictors + ['participant_id']), formula1, 'participant_id', os.path.join(args.outdir, 'model_dwell_time.txt'), 'MixedLM: dwell_time_ms')

    # Model 2: TTFF
    formula2 = f'TTFF_ms ~ {rhs}'
    fit_and_save(dfm.dropna(subset=['TTFF_ms'] + predictors + ['participant_id']), formula2, 'participant_id', os.path.join(args.outdir, 'model_ttff.txt'), 'MixedLM: TTFF_ms')

    # Model 3: fixation count
    formula3 = f'fixation_count ~ {rhs}'
    fit_and_save(dfm.dropna(subset=['fixation_count'] + predictors + ['participant_id']), formula3, 'participant_id', os.path.join(args.outdir, 'model_fixation_count.txt'), 'MixedLM: fixation_count')

    print('Saved models to', args.outdir)


if __name__ == '__main__':
    main()
