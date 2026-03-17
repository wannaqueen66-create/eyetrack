#!/usr/bin/env python3
"""Summarize AOI metrics by (WWR × Complexity) × group and export PNG interaction plots."""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.figure_style import apply_paper_style, soften_axes, PALETTE, metric_label, annotate_series_smart, metric_value_label
from src.manifest_scene_order import attach_manifest_trial_metadata


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _norm_hilo(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    m = {"high": "High", "h": "High", "1": "High", "true": "High", "yes": "High", "low": "Low", "l": "Low", "0": "Low", "false": "Low", "no": "Low"}
    return m.get(s.lower(), s)


def _pick_col(df: pd.DataFrame, primary: str, fallback: str | None = None) -> str | None:
    if primary in df.columns:
        return primary
    if fallback and fallback in df.columns:
        return fallback
    return None


def _complexity_to_label(x) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    if s in ("0", "C0", "LOW", "L"):
        return "C0"
    if s in ("1", "C1", "HIGH", "H"):
        return "C1"
    import re
    m = re.search(r"C([01])", s)
    return f"C{m.group(1)}" if m else (s if s else None)


def _ensure_dirs(outdir: Path):
    (outdir / 'tables').mkdir(parents=True, exist_ok=True)
    (outdir / 'png').mkdir(parents=True, exist_ok=True)
    (outdir / 'data').mkdir(parents=True, exist_ok=True)


def _ci95(x: pd.Series) -> tuple[float, float, float, int]:
    x = _safe_num(x).dropna(); n = int(len(x))
    if n == 0:
        return (np.nan, np.nan, np.nan, 0)
    m = float(x.mean()); sd = float(x.std(ddof=1)) if n >= 2 else 0.0; se = sd / np.sqrt(n) if n else np.nan; ci = 1.96 * se if np.isfinite(se) else np.nan
    return (m, m - ci, m + ci, n)


def _make_share(df: pd.DataFrame, dwell_col: str, key_cols: list[str]):
    d = df.copy(); d[dwell_col] = _safe_num(d[dwell_col]).clip(lower=0)
    tot = d.groupby(key_cols, dropna=False)[dwell_col].sum(min_count=1).rename("dwell_total")
    d = d.merge(tot.reset_index(), on=key_cols, how="left")
    d["share"] = d[dwell_col] / d["dwell_total"]
    d.loc[~np.isfinite(d["share"]), "share"] = np.nan
    d["share_pct"] = 100.0 * d["share"]
    d.loc[~np.isfinite(d["share_pct"]), "share_pct"] = np.nan
    return d


def _export_plot_companion(summary: pd.DataFrame, out_png: Path, outcome: str):
    if summary is None or summary.empty:
        return
    data_df = summary.copy()
    data_df['value_label'] = data_df['mean'].apply(
        lambda v: metric_value_label(outcome, float(v)) if pd.notna(v) else ''
    )
    target = (out_png.parent.parent / 'data' / f"{out_png.stem}_data.csv") if out_png.parent.name == 'png' else out_png.with_name(out_png.stem + '_data.csv')
    target.parent.mkdir(parents=True, exist_ok=True)
    data_df.to_csv(target, index=False, encoding='utf-8-sig')


def plot_grid(summary: pd.DataFrame, out_png: Path, outcome: str, group_var: str, title: str):
    if summary.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    aoi_levels = sorted(summary["class_name"].astype(str).unique().tolist())
    group_levels = [g for g in ["Low", "High"] if g in set(summary["group_value"].astype(str))] or sorted(summary["group_value"].astype(str).unique().tolist())
    wwr_levels = [15, 45, 75]
    fig, axes = plt.subplots(nrows=max(1, len(group_levels)), ncols=max(1, len(aoi_levels)), figsize=(4.6 * max(1, len(aoi_levels)), 3.5 * max(1, len(group_levels))), sharey=True, sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape((1, -1))
    apply_paper_style(); colors = {"C0": PALETTE["blue"], "C1": PALETTE["orange"]}
    complexity_legend = {"C0": "Low", "C1": "High"}
    for i, gv in enumerate(group_levels):
        for j, aoi in enumerate(aoi_levels):
            ax = axes[i, j] if (i < axes.shape[0] and j < axes.shape[1]) else None
            if ax is None:
                continue
            sub = summary[(summary["group_value"].astype(str) == str(gv)) & (summary["class_name"].astype(str) == str(aoi))].copy()
            if sub.empty:
                ax.set_axis_off(); continue
            for cx in ["C0", "C1"]:
                s2 = sub[sub["Complexity"].astype(str) == cx].copy()
                if s2.empty:
                    continue
                xs, ys, lo, hi = [], [], [], []
                for w in wwr_levels:
                    r = s2[s2["WWR"] == w]
                    if len(r) == 0:
                        xs.append(w); ys.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                    else:
                        rr = r.iloc[0]; xs.append(w); ys.append(float(rr["mean"])); lo.append(float(rr["ci_low"])); hi.append(float(rr["ci_high"]))
                line_color = colors.get(cx, PALETTE['gray'])
                ax.plot(xs, ys, marker="o", linewidth=1.9, color=line_color, label=complexity_legend.get(cx, cx))
                ax.fill_between(xs, lo, hi, color=line_color, alpha=0.08, linewidth=0)
                label_budget = 2 if str(outcome).upper() in {"TTFF", "FFD", "TFD", "MFD", "RFF", "MPD"} else 3
                annotate_series_smart(ax, xs, ys, metric=outcome, color=line_color, max_labels=label_budget)
            ax.set_title(f"{group_var}={gv} | AOI={aoi}", pad=8)
            ax.set_xticks(wwr_levels)
            ax.set_xlabel("WWR")
            # sharex=True hides upper-row x tick labels by default in matplotlib.
            # Force every subplot to show its horizontal axis so the top row does
            # not lose WWR labels in the exported descriptive PNG grids.
            ax.tick_params(axis='x', labelbottom=True)
            for lbl in ax.get_xticklabels():
                lbl.set_visible(True)
            if j == 0:
                ax.set_ylabel(metric_label(outcome) if outcome in ['FC', 'TTFF', 'FFD', 'TFD', 'MFD', 'RFF', 'MPD'] else outcome)
            soften_axes(ax)
    # Collect legend entries from all subplots instead of only the first one,
    # because the top-left panel may contain only one complexity level and would
    # otherwise drop the other legend item.
    handle_map = {}
    for ax in axes.ravel():
        if ax is None:
            continue
        hs, ls = ax.get_legend_handles_labels()
        for h, l in zip(hs, ls):
            if l not in handle_map:
                handle_map[l] = h
    ordered_labels = [l for l in ["Low", "High"] if l in handle_map]
    if ordered_labels:
        fig.legend(
            [handle_map[l] for l in ordered_labels],
            ordered_labels,
            loc="upper right",
            frameon=False,
            title="Complexity",
            ncol=1,
        )
    fig.suptitle(title, y=1.01, fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)
    _export_plot_companion(summary, out_png, outcome)


def main():
    ap = argparse.ArgumentParser(description="Summarize AOI metrics by WWR×Complexity×Group and export PNGs")
    ap.add_argument("--aoi_class_csv", required=True)
    ap.add_argument("--group_manifest", required=True)
    ap.add_argument("--group_id_col", default="name")
    ap.add_argument("--outdir", default="outputs_aoi_summary")
    ap.add_argument("--include_round", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir); _ensure_dirs(outdir)
    df = pd.read_csv(args.aoi_class_csv, encoding="utf-8-sig"); gm = pd.read_csv(args.group_manifest, encoding="utf-8-sig")
    if args.group_id_col not in gm.columns:
        raise SystemExit(f"group_manifest missing id col: {args.group_id_col}")
    gm = gm.copy(); gm["participant_id"] = gm[args.group_id_col].astype(str).str.strip()
    for c in ["SportFreq", "Experience"]:
        if c in gm.columns: gm[c] = gm[c].apply(_norm_hilo)
    df["participant_id"] = df["participant_id"].astype(str).str.strip()
    df = df.merge(gm[[c for c in ["participant_id", "SportFreq", "Experience"] if c in gm.columns]], on="participant_id", how="left")
    scene_src = "scene_id_raw" if "scene_id_raw" in df.columns else ("scene_id" if "scene_id" in df.columns else None)
    if scene_src:
        df["scene_id_match"] = df[scene_src].astype(str).str.strip()
        df = attach_manifest_trial_metadata(df, gm, id_col=args.group_id_col, scene_col="scene_id_match")
        if "round" in df.columns:
            df["round"] = _safe_num(df["round"]).astype("Int64")
        if "WWR" in df.columns:
            df["WWR"] = _safe_num(df["WWR"]).astype("Int64")
        if "Complexity" in df.columns:
            df["Complexity"] = df["Complexity"].apply(_complexity_to_label)
    if "WWR" not in df.columns or "Complexity" not in df.columns:
        raise SystemExit("Missing WWR/Complexity columns. Re-run batch_aoi_metrics.py after latest patch.")
    df["WWR"] = _safe_num(df["WWR"]).astype("Int64"); df["Complexity"] = df["Complexity"].apply(_complexity_to_label)
    if "round" in df.columns: df["round"] = _safe_num(df["round"]).astype("Int64")
    if "trial_excluded" in df.columns:
        df["trial_excluded"] = _safe_num(df["trial_excluded"]).fillna(0).astype(int); df = df[df["trial_excluded"] == 0].copy()
    dwell_col = _pick_col(df, "TFD", "dwell_time_ms"); ttff_col = _pick_col(df, "TTFF"); fc_col = _pick_col(df, "FC", "fixation_count")
    outcomes = []
    key_scene = "scene_id_raw" if "scene_id_raw" in df.columns else ("scene_id" if "scene_id" in df.columns else None)
    if dwell_col:
        df["TFD"] = _safe_num(df[dwell_col]).clip(lower=0); df["tfd_y"] = np.log1p(df["TFD"]); outcomes += [("TFD", "Total Fixation Duration (TFD)"), ("tfd_y", "log1p(TFD)")]
        if key_scene:
            df2 = _make_share(df, "TFD", ["participant_id", key_scene]); df2["share_logit"] = np.log((df2["share"] + 1e-6) / (1 - df2["share"] + 1e-6)); df_share = df2; outcomes += [("share", "share(TFD within trial)"), ("share_pct", "share_pct (TFD % within trial)"), ("share_logit", "logit(share)")]
        else:
            df_share = df
    else:
        df_share = df
    if ttff_col:
        df["TTFF"] = _safe_num(df[ttff_col]); df["ttff_y"] = np.log1p(df["TTFF"].clip(lower=0)); outcomes += [("TTFF", "Time to First Fixation (TTFF)"), ("ttff_y", "log1p(TTFF)")]
    if fc_col:
        df["FC"] = _safe_num(df[fc_col]).clip(lower=0); df["fc_y"] = np.log1p(df["FC"]); outcomes += [("FC", "Fixation Count (FC)"), ("fc_y", "log1p(FC)")]
    if "FC_share" in df.columns:
        df["FC_share"] = _safe_num(df["FC_share"]).clip(lower=0, upper=1)
        outcomes += [("FC_share", "FC_share (AOI FC share within trial)"), ("FC_prop", "FC_prop (alias of FC_share)")]
        df["FC_prop"] = df["FC_share"]
    if "FC_rate" in df.columns:
        df["FC_rate"] = _safe_num(df["FC_rate"]).clip(lower=0)
        outcomes += [("FC_rate", "FC_rate (AOI FC per second)")]
    if "FFD" in df.columns:
        df["FFD"] = _safe_num(df["FFD"]).clip(lower=0)
        df["ffd_y"] = np.log1p(df["FFD"])
        outcomes += [("FFD", "First Fixation Duration (FFD)"), ("ffd_y", "log1p(FFD)")]
    if "MFD" in df.columns:
        df["MFD"] = _safe_num(df["MFD"]).clip(lower=0)
        df["mfd_y"] = np.log1p(df["MFD"])
        outcomes += [("MFD", "Mean Fixation Duration (MFD)"), ("mfd_y", "log1p(MFD)")]
    if "RFF" in df.columns:
        df["RFF"] = _safe_num(df["RFF"]).clip(lower=0)
        df["rff_y"] = np.log1p(df["RFF"])
        outcomes += [("RFF", "Re-fixation Frequency (RFF)"), ("rff_y", "log1p(RFF)")]
    if "MPD" in df.columns:
        df["MPD"] = _safe_num(df["MPD"])
        outcomes += [("MPD", "Mean Pupil Diameter (MPD)")]
    if "visited" in df.columns: df["visited"] = _safe_num(df["visited"]).fillna(0).astype(int)
    group_vars = [g for g in ["Experience", "SportFreq"] if g in df.columns]
    if not group_vars:
        raise SystemExit("No Experience/SportFreq columns found after merge.")
    base_cols = ["class_name", "WWR", "Complexity"]
    if args.include_round and ("round" in df.columns) and df["round"].notna().any(): base_cols = ["round"] + base_cols
    for gv in group_vars:
        d0 = df_share.copy() if any(o[0].startswith("share") for o in outcomes) else df.copy()
        for ycol, yname in outcomes:
            if ycol not in d0.columns:
                continue
            d = d0.copy()
            if ycol in {"TTFF", "ttff_y", "FC", "fc_y", "FFD", "ffd_y", "MFD", "mfd_y", "MPD"}:
                if "visited" in d.columns: d = d[d["visited"] == 1].copy()
            d = d.dropna(subset=[ycol, gv, "WWR", "Complexity", "class_name"])
            if d.empty:
                continue
            rows = []; grp_cols = base_cols + [gv]
            for keys, sub in d.groupby(grp_cols, dropna=False):
                if not isinstance(keys, tuple): keys = (keys,)
                key_map = dict(zip(grp_cols, keys)); mean, lo, hi, n = _ci95(sub[ycol])
                row = {"group_var": gv, "group_value": key_map[gv], "class_name": key_map["class_name"], "WWR": int(key_map["WWR"]) if pd.notna(key_map["WWR"]) else np.nan, "Complexity": key_map["Complexity"], "outcome": yname, "mean": mean, "ci_low": lo, "ci_high": hi, "n": n}
                if "round" in key_map: row["round"] = int(key_map["round"]) if pd.notna(key_map["round"]) else np.nan
                rows.append(row)
            summ = pd.DataFrame(rows); summ["group_value"] = summ["group_value"].apply(_norm_hilo)
            summ.to_csv(outdir / "tables" / f"summary_{gv}_{ycol}.csv", index=False, encoding="utf-8-sig")
            pretty_group = {
                'Experience': 'Experience group',
                'SportFreq': 'Sport frequency group',
            }.get(gv, gv)
            title = f"Descriptive AOI pattern of {yname} by WWR, complexity, and {pretty_group}"
            plot_grid(summ, outdir / 'png' / f"plot_{gv}_{ycol}.png", outcome=yname, group_var=gv, title=title)
    (outdir / "RUNINFO.txt").write_text("AOI summary by condition/group\n" + f"aoi_class_csv: {args.aoi_class_csv}\n" + f"group_manifest: {args.group_manifest}\n", encoding="utf-8")
    print("Saved:", str(outdir))


if __name__ == "__main__":
    main()
