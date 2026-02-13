#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style='whitegrid')


def main():
    ap = argparse.ArgumentParser(description='Generate paper-ready figures for indoor pingpong eye-tracking')
    ap.add_argument('--analysis_csv', required=True)
    ap.add_argument('--outdir', default='figures_paper')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.analysis_csv)

    if 'class_name' in df.columns:
        p = df[df['class_name'] == 'pingpong_table'].copy()
        if len(p) >= 3:
            df = p

    # Figure 1: Dwell by condition
    if 'condition' in df.columns:
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=df, x='condition', y='dwell_time_ms')
        sns.stripplot(data=df, x='condition', y='dwell_time_ms', color='black', alpha=0.45, size=3)
        plt.title('Dwell Time on Pingpong-Table AOI by Condition')
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, 'fig1_dwell_by_condition.png'), dpi=300)
        plt.close()

    # Figure 2: TTFF vs table density
    if 'table_density' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.regplot(data=df, x='table_density', y='TTFF_ms', scatter_kws={'alpha':0.6, 's':20})
        plt.title('TTFF vs Table Density')
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, 'fig2_ttff_vs_table_density.png'), dpi=300)
        plt.close()

    # Figure 3: fixation count vs crowding
    if 'crowding_level' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.regplot(data=df, x='crowding_level', y='fixation_count', scatter_kws={'alpha':0.6, 's':20})
        plt.title('Fixation Count vs Crowding Level')
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, 'fig3_fixcount_vs_crowding.png'), dpi=300)
        plt.close()

    print('Saved figures to', args.outdir)


if __name__ == '__main__':
    main()
