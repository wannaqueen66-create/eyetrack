#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.figure_style import PALETTE, apply_paper_style, metric_label, soften_axes
from src.aoi_metrics import normalize_aoi_class_series


def _annotate_box_means(ax, df, x_col: str, y_col: str, hue_col: str | None = None, order=None, hue_order=None):
    if df.empty:
        return
    order = order or [x for x in pd.Series(df[x_col]).dropna().astype(str).unique().tolist()]
    if hue_col:
        hue_order = hue_order or [x for x in pd.Series(df[hue_col]).dropna().astype(str).unique().tolist()]
        n_hue = max(1, len(hue_order))
        offsets = np.linspace(-0.20, 0.20, n_hue)
        for i, xv in enumerate(order):
            subx = df[df[x_col].astype(str) == str(xv)]
            if subx.empty:
                continue
            for j, hv in enumerate(hue_order):
                sub = subx[subx[hue_col].astype(str) == str(hv)]
                if sub.empty:
                    continue
                y = pd.to_numeric(sub[y_col], errors='coerce').dropna()
                if y.empty:
                    continue
                yy = float(y.mean())
                ax.text(i + offsets[j], yy, f"{yy:.1f}", ha='center', va='bottom', fontsize=7,
                        color=PALETTE['ink'], bbox=dict(boxstyle='round,pad=0.14', facecolor='white', edgecolor='none', alpha=0.92), zorder=6)
    else:
        for i, xv in enumerate(order):
            sub = df[df[x_col].astype(str) == str(xv)]
            y = pd.to_numeric(sub[y_col], errors='coerce').dropna()
            if y.empty:
                continue
            yy = float(y.mean())
            ax.text(i, yy, f"{yy:.1f}", ha='center', va='bottom', fontsize=7,
                    color=PALETTE['ink'], bbox=dict(boxstyle='round,pad=0.14', facecolor='white', edgecolor='none', alpha=0.92), zorder=6)


def main():
    ap = argparse.ArgumentParser(description='Generate manuscript-ready figures for eye-tracking outputs')
    ap.add_argument('--analysis_csv', required=True)
    ap.add_argument('--outdir', default='figures_manuscript')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.analysis_csv)

    if 'class_name' in df.columns:
        df['class_name'] = normalize_aoi_class_series(df['class_name'])
        p = df[df['class_name'] == 'table'].copy()
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
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        hue_col = 'Experience' if 'Experience' in df.columns else None
        order = [x for x in pd.Series(df['condition']).dropna().astype(str).unique().tolist()]
        hue_order = None
        palette = None
        if hue_col:
            hue_order = [g for g in ['Low', 'High'] if g in set(df[hue_col].astype(str))] or sorted(df[hue_col].dropna().astype(str).unique().tolist())
            palette = {'Low': PALETTE['blue'], 'High': PALETTE['orange']}
            sns.boxplot(data=df, x='condition', y='TFD', hue=hue_col, order=order, hue_order=hue_order,
                        ax=ax, palette=palette, width=0.64, dodge=True, fliersize=0, linewidth=0.95)
            sns.stripplot(data=df, x='condition', y='TFD', hue=hue_col, order=order, hue_order=hue_order,
                          ax=ax, dodge=True, palette=palette, alpha=0.26, size=2.6, jitter=0.10,
                          edgecolor='white', linewidth=0.28)
            # de-duplicate legend
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                n = len(hue_order)
                ax.legend(handles[:n], labels[:n], title=hue_col, frameon=False, ncol=min(2, n))
        else:
            sns.boxplot(data=df, x='condition', y='TFD', ax=ax, color=PALETTE['light_blue'], width=0.58, fliersize=0, linewidth=0.95)
            sns.stripplot(data=df, x='condition', y='TFD', ax=ax, color=PALETTE['ink'], alpha=0.24, size=2.6, jitter=0.10,
                          edgecolor='white', linewidth=0.28)
        _annotate_box_means(ax, df, x_col='condition', y_col='TFD', hue_col=hue_col, order=order, hue_order=hue_order)
        ax.set_title('Total Fixation Duration by Condition', pad=10)
        ax.set_xlabel('Condition')
        ax.set_ylabel(metric_label('TFD'))
        soften_axes(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, 'fig1_tfd_by_condition.png'), dpi=320)
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
