#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.figure_style import PALETTE, apply_paper_style, metric_label, soften_axes


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
    if 'FC' not in df.columns and 'fixation_count' in df.columns:
        df['FC'] = pd.to_numeric(df['fixation_count'], errors='coerce')
    if 'TTFF' not in df.columns:
        raise ValueError('analysis_csv missing required column: TTFF')

    apply_paper_style()

    if 'condition' in df.columns and 'TFD' in df.columns:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        sns.boxplot(data=df, x='condition', y='TFD', ax=ax, color=PALETTE['blue'], width=0.55, fliersize=0, linewidth=1.0)
        sns.stripplot(data=df, x='condition', y='TFD', ax=ax, color=PALETTE['ink'], alpha=0.35, size=3, jitter=0.18)
        ax.set_title('Total Fixation Duration by Condition', pad=10)
        ax.set_xlabel('Condition')
        ax.set_ylabel(metric_label('TFD'))
        soften_axes(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, 'fig1_tfd_by_condition.png'), dpi=300)
        plt.close(fig)

    if 'table_density' in df.columns and 'TTFF' in df.columns:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        sns.regplot(data=df, x='table_density', y='TTFF', ax=ax, color=PALETTE['orange'], scatter_kws={'alpha': 0.55, 's': 24, 'edgecolor': 'white', 'linewidth': 0.4}, line_kws={'linewidth': 1.8})
        ax.set_title('TTFF vs Table Density', pad=10)
        ax.set_xlabel('Table density')
        ax.set_ylabel(metric_label('TTFF'))
        soften_axes(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, 'fig2_ttff_vs_table_density.png'), dpi=300)
        plt.close(fig)

    if 'crowding_level' in df.columns and 'FC' in df.columns:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        sns.regplot(data=df, x='crowding_level', y='FC', ax=ax, color=PALETTE['green'], scatter_kws={'alpha': 0.55, 's': 24, 'edgecolor': 'white', 'linewidth': 0.4}, line_kws={'linewidth': 1.8})
        ax.set_title('Fixation Count vs Crowding Level', pad=10)
        ax.set_xlabel('Crowding level')
        ax.set_ylabel(metric_label('FC'))
        soften_axes(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, 'fig3_fc_vs_crowding.png'), dpi=300)
        plt.close(fig)

    print('Saved figures to', args.outdir)


if __name__ == '__main__':
    main()
