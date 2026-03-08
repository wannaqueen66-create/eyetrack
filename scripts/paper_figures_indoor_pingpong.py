#!/usr/bin/env python3
import argparse
import os
import sys

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    if 'TFD' not in df.columns and 'dwell_time_ms' in df.columns:
        df['TFD'] = pd.to_numeric(df['dwell_time_ms'], errors='coerce')
    if 'TFF' not in df.columns and 'TTFF_ms' in df.columns:
        df['TFF'] = pd.to_numeric(df['TTFF_ms'], errors='coerce')
    if 'FC' not in df.columns and 'fixation_count' in df.columns:
        df['FC'] = pd.to_numeric(df['fixation_count'], errors='coerce')

    # Figure 1: TFD by condition
    if 'condition' in df.columns and 'TFD' in df.columns:
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=df, x='condition', y='TFD')
        sns.stripplot(data=df, x='condition', y='TFD', color='black', alpha=0.45, size=3)
        plt.title('Total Fixation Duration on Pingpong-Table AOI by Condition')
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, 'fig1_tfd_by_condition.png'), dpi=300)
        plt.close()

    # Figure 2: TFF vs table density
    if 'table_density' in df.columns and 'TFF' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.regplot(data=df, x='table_density', y='TFF', scatter_kws={'alpha':0.6, 's':20})
        plt.title('TFF vs Table Density')
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, 'fig2_tff_vs_table_density.png'), dpi=300)
        plt.close()

    # Figure 3: FC vs crowding
    if 'crowding_level' in df.columns and 'FC' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.regplot(data=df, x='crowding_level', y='FC', scatter_kws={'alpha':0.6, 's':20})
        plt.title('Fixation Count vs Crowding Level')
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, 'fig3_fc_vs_crowding.png'), dpi=300)
        plt.close()

    print('Saved figures to', args.outdir)


if __name__ == '__main__':
    main()
