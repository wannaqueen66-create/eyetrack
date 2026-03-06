#!/usr/bin/env python3
"""Re-organize AOI outputs into scene/participant/grouped views + B&E-style PNGs.

Input
- --aoi_class_csv: batch_aoi_metrics_by_class.csv
- --aoi_polygon_csv: batch_aoi_metrics_by_polygon.csv (optional)
- --group_manifest: CSV with participant id + SportFreq/Experience (optional but recommended)

Output structure
- outdir/by_scene/<scene_id>/participants/*.csv
- outdir/by_participant/<participant_id>/*.csv
- outdir/grouped/tables/*.csv
- outdir/grouped/plots/*.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _norm_group(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    m = {
        "high": "High", "h": "High", "1": "High", "true": "High", "yes": "High",
        "low": "Low", "l": "Low", "0": "Low", "false": "Low", "no": "Low",
    }
    return m.get(s.lower(), s)


def _safe_num(s):
    return pd.to_numeric(s, errors="coerce")


def _apply_be_style():
    # clean journal-like style (Building and Environment friendly)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "savefig.dpi": 300,
    })


def _metric_col(df: pd.DataFrame, primary: str, fallback: str | None = None) -> str | None:
    if primary in df.columns:
        return primary
    if fallback and fallback in df.columns:
        return fallback
    return None


def _ensure_dirs(base: Path):
    (base / "by_scene").mkdir(parents=True, exist_ok=True)
    (base / "by_participant").mkdir(parents=True, exist_ok=True)
    (base / "grouped" / "tables").mkdir(parents=True, exist_ok=True)
    (base / "grouped" / "plots").mkdir(parents=True, exist_ok=True)


def _write_views(df_class: pd.DataFrame, df_poly: pd.DataFrame | None, outdir: Path):
    # by_scene
    for scene_id, g in df_class.groupby("scene_id", dropna=False):
        sname = str(scene_id)
        sdir = outdir / "by_scene" / sname / "participants"
        sdir.mkdir(parents=True, exist_ok=True)
        for pid, gp in g.groupby("participant_id", dropna=False):
            gp.to_csv(sdir / f"{pid}_class.csv", index=False)

    if df_poly is not None and len(df_poly):
        for scene_id, g in df_poly.groupby("scene_id", dropna=False):
            sname = str(scene_id)
            sdir = outdir / "by_scene" / sname / "participants"
            sdir.mkdir(parents=True, exist_ok=True)
            for pid, gp in g.groupby("participant_id", dropna=False):
                gp.to_csv(sdir / f"{pid}_polygon.csv", index=False)

    # by_participant
    for pid, g in df_class.groupby("participant_id", dropna=False):
        pdir = outdir / "by_participant" / str(pid)
        pdir.mkdir(parents=True, exist_ok=True)
        for scene_id, gs in g.groupby("scene_id", dropna=False):
            gs.to_csv(pdir / f"{scene_id}_class.csv", index=False)

    if df_poly is not None and len(df_poly):
        for pid, g in df_poly.groupby("participant_id", dropna=False):
            pdir = outdir / "by_participant" / str(pid)
            pdir.mkdir(parents=True, exist_ok=True)
            for scene_id, gs in g.groupby("scene_id", dropna=False):
                gs.to_csv(pdir / f"{scene_id}_polygon.csv", index=False)


def _group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    tfd_col = _metric_col(df, "TFD", "dwell_time_ms")
    ttff_col = _metric_col(df, "TTFF", "TTFF_ms")
    fc_col = _metric_col(df, "FC", "fixation_count")

    for (scene_id, class_name, gv), sub in df.groupby(["scene_id", "class_name", group_col], dropna=False):
        if pd.isna(gv) or str(gv).strip() == "":
            continue
        v = _safe_num(sub.get("visited", 0)).fillna(0)
        sub_v = sub[v.astype(int) == 1]

        row = {
            "scene_id": scene_id,
            "class_name": class_name,
            "group_col": group_col,
            "group_value": gv,
            "n_trials": int(len(sub)),
            "visited_rate": float(v.mean()) if len(v) else np.nan,
        }
        if ttff_col:
            tt = _safe_num(sub_v.get(ttff_col))
            row["TTFF_mean_given_visited"] = float(tt.mean()) if tt.notna().any() else np.nan
        if tfd_col:
            td = _safe_num(sub.get(tfd_col))
            row["TFD_mean_all"] = float(td.mean()) if td.notna().any() else np.nan
            tdv = _safe_num(sub_v.get(tfd_col))
            row["TFD_mean_given_visited"] = float(tdv.mean()) if tdv.notna().any() else np.nan
        if fc_col:
            fc = _safe_num(sub.get(fc_col))
            row["FC_mean_all"] = float(fc.mean()) if fc.notna().any() else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def _plot_group_metric(summary_df: pd.DataFrame, metric: str, out_png: Path, title: str):
    if summary_df.empty or metric not in summary_df.columns:
        return

    _apply_be_style()
    data = summary_df.copy()
    data[metric] = _safe_num(data[metric])
    data = data.dropna(subset=[metric])
    if data.empty:
        return

    # aggregate to scene x group mean across classes for cleaner figure
    agg = data.groupby(["scene_id", "group_value"], as_index=False)[metric].mean(numeric_only=True)
    scenes = list(agg["scene_id"].astype(str).unique())
    groups = sorted(agg["group_value"].astype(str).unique())

    width = 0.8 / max(1, len(groups))
    x = np.arange(len(scenes))

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]

    for i, g in enumerate(groups):
        y = []
        for s in scenes:
            tmp = agg[(agg["scene_id"].astype(str) == s) & (agg["group_value"].astype(str) == g)][metric]
            y.append(float(tmp.iloc[0]) if len(tmp) else np.nan)
        offs = x + (i - (len(groups) - 1) / 2) * width
        ax.bar(offs, y, width=width, label=g, color=palette[i % len(palette)], alpha=0.92)

    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_xlabel("Scene")
    ax.set_ylabel(metric)
    ax.legend(title="Group", frameon=False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Reorganize AOI outputs + grouped summaries + B&E-style plots")
    ap.add_argument("--aoi_class_csv", required=True)
    ap.add_argument("--aoi_polygon_csv", default=None)
    ap.add_argument("--group_manifest", default=None)
    ap.add_argument("--group_id_col", default="name", help="id column in group manifest (default: name)")
    ap.add_argument("--outdir", default="outputs_organized")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    _ensure_dirs(outdir)

    df_class = pd.read_csv(args.aoi_class_csv)
    df_poly = pd.read_csv(args.aoi_polygon_csv) if args.aoi_polygon_csv and os.path.exists(args.aoi_polygon_csv) else None

    for c in ["participant_id", "scene_id", "class_name"]:
        if c not in df_class.columns:
            raise SystemExit(f"aoi_class_csv missing required column: {c}")

    _write_views(df_class, df_poly, outdir)

    # grouped summaries
    if args.group_manifest and os.path.exists(args.group_manifest):
        gm = pd.read_csv(args.group_manifest)
        if args.group_id_col not in gm.columns:
            raise SystemExit(f"group_manifest missing id col: {args.group_id_col}")

        gm = gm.copy()
        gm["participant_id"] = gm[args.group_id_col].astype(str).str.strip()
        if "SportFreq" in gm.columns:
            gm["SportFreq"] = gm["SportFreq"].apply(_norm_group)
        if "Experience" in gm.columns:
            gm["Experience"] = gm["Experience"].apply(_norm_group)

        d = df_class.copy()
        d["participant_id"] = d["participant_id"].astype(str).str.strip()
        d = d.merge(gm[[c for c in ["participant_id", "SportFreq", "Experience"] if c in gm.columns]], on="participant_id", how="left")

        if "SportFreq" in d.columns:
            sport = _group_summary(d, "SportFreq")
            sport.to_csv(outdir / "grouped" / "tables" / "summary_sportfreq.csv", index=False)
            _plot_group_metric(sport, "visited_rate", outdir / "grouped" / "plots" / "sportfreq_visited_rate.png", "Visited rate by SportFreq")
            _plot_group_metric(sport, "TTFF_mean_given_visited", outdir / "grouped" / "plots" / "sportfreq_TTFF.png", "TTFF (visited trials) by SportFreq")
            _plot_group_metric(sport, "TFD_mean_given_visited", outdir / "grouped" / "plots" / "sportfreq_TFD.png", "TFD (visited trials) by SportFreq")

        if "Experience" in d.columns:
            exp = _group_summary(d, "Experience")
            exp.to_csv(outdir / "grouped" / "tables" / "summary_experience.csv", index=False)
            _plot_group_metric(exp, "visited_rate", outdir / "grouped" / "plots" / "experience_visited_rate.png", "Visited rate by Experience")
            _plot_group_metric(exp, "TTFF_mean_given_visited", outdir / "grouped" / "plots" / "experience_TTFF.png", "TTFF (visited trials) by Experience")
            _plot_group_metric(exp, "TFD_mean_given_visited", outdir / "grouped" / "plots" / "experience_TFD.png", "TFD (visited trials) by Experience")


    print("Saved organized outputs to:", outdir)


if __name__ == "__main__":
    main()
