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

    if 'TFD' not in dfm.columns and 'dwell_time_ms' in dfm.columns:
        dfm['TFD'] = pd.to_numeric(dfm['dwell_time_ms'], errors='coerce')
    if 'TTFF' not in dfm.columns and 'TFF' in dfm.columns:
        dfm['TTFF'] = pd.to_numeric(dfm['TFF'], errors='coerce')
    if 'TTFF' not in dfm.columns and 'TTFF_ms' in dfm.columns:
        dfm['TTFF'] = pd.to_numeric(dfm['TTFF_ms'], errors='coerce')
    if 'TFF' not in dfm.columns and 'TTFF' in dfm.columns:
        dfm['TFF'] = pd.to_numeric(dfm['TTFF'], errors='coerce')
    if 'FC' not in dfm.columns and 'fixation_count' in dfm.columns:
        dfm['FC'] = pd.to_numeric(dfm['fixation_count'], errors='coerce')

    required = ['participant_id', 'TFD', 'TFF', 'FC']
    for c in required:
        if c not in dfm.columns:
            raise ValueError(f'Missing required column: {c}')

    # choose available predictors (prefer physically meaningful manual fields when present,
    # but also accept auto-generated AOI/image-derived fields so the Colab one-command flow can run)
    predictor_candidates = [
        'table_density',
        'distance_to_table_center_m',
        'table_center_offset_ratio',
        'illum_lux',
        'crowding_level',
        'occlusion_ratio',
        'aoi_coverage_ratio',
        'non_table_aoi_coverage_ratio',
        'WWR',
    ]
    predictors = [c for c in predictor_candidates if c in dfm.columns]
    if not predictors:
        raise ValueError('No predictor columns found. Provide at least one scene feature predictor (manual or auto-generated).')

    rhs = ' + '.join(predictors)

    # Model 1: TFD
    formula1 = f'TFD ~ {rhs}'
    fit_and_save(dfm.dropna(subset=['TFD'] + predictors + ['participant_id']), formula1, 'participant_id', os.path.join(args.outdir, 'model_tfd.txt'), 'MixedLM: TFD')

    # Model 2: TFF
    formula2 = f'TFF ~ {rhs}'
    fit_and_save(dfm.dropna(subset=['TFF'] + predictors + ['participant_id']), formula2, 'participant_id', os.path.join(args.outdir, 'model_tff.txt'), 'MixedLM: TFF')

    # Model 3: FC
    formula3 = f'FC ~ {rhs}'
    fit_and_save(dfm.dropna(subset=['FC'] + predictors + ['participant_id']), formula3, 'participant_id', os.path.join(args.outdir, 'model_fc.txt'), 'MixedLM: FC')

    print('Saved models to', args.outdir)


if __name__ == '__main__':
    main()
