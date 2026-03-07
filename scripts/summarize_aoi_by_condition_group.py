#!/usr/bin/env python3
"""Summarize AOI metrics by (WWR × Complexity) × group and export PNG interaction plots.

This is the "table + plots" companion to scripts/model_aoi_lmm_allocation.py.

Inputs
------
- --aoi_class_csv: batch_aoi_metrics_by_class.csv from scripts/batch_aoi_metrics.py
  Expected to include (after latest patch):
    participant_id, class_name, scene_id_raw, round, WWR, Complexity, condition_id
  Plus AOI metrics columns like dwell_time_ms/TFD, TTFF_ms/TTFF, fixation_count/FC, visited, RF, MPD.

- --group_manifest: group_manifest.csv with participant id column (default: name)
  and group columns SportFreq and/or Experience.

Outputs
-------
- summary tables (CSV) under outdir/tables/
- interaction plots (PNG) under outdir/plots/

Plots
-----
For each group variable (Experience, SportFreq) and each outcome, export a grid:
- rows: group level (Low/High)
- cols: AOI class_name (e.g., pingpong_table/window/equipment)
- x: WWR (15/45/75)
- line/color: Complexity (C0/C1)

Notes
-----
- This script is descriptive (means + CI). It does not do inferential tests.
  Use model_aoi_lmm_allocation.py (or R lme4+emmeans) for inference.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _norm_hilo(x):
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
    # allow extracting C0/C1
    import re
    m = re.search(r"C([01])", s)
    if m:
        return f"C{m.group(1)}"
    return s if s else None


def _ensure_dirs(outdir: Path):
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)


def _ci95(x: pd.Series) -> tuple[float, float, float, int]:
    x = _safe_num(x).dropna()
    n = int(len(x))
    if n == 0:
        return (np.nan, np.nan, np.nan, 0)
    m = float(x.mean())
    sd = float(x.std(ddof=1)) if n >= 2 else 0.0
    se = sd / np.sqrt(n) if n else np.nan
    ci = 1.96 * se if np.isfinite(se) else np.nan
    return (m, m - ci, m + ci, n)


def _make_share(df: pd.DataFrame, dwell_col: str, key_cols: list[str]):
    d = df.copy()
    d[dwell_col] = _safe_num(d[dwell_col]).clip(lower=0)
    tot = d.groupby(key_cols, dropna=False)[dwell_col].sum(min_count=1).rename("dwell_total")
    d = d.merge(tot.reset_index(), on=key_cols, how="left")
    d["share"] = d[dwell_col] / d["dwell_total"]
    d.loc[~np.isfinite(d["share"]), "share"] = np.nan
    return d


def plot_grid(summary: pd.DataFrame, out_png: Path, outcome: str, group_var: str, title: str):
    if summary.empty:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    # Determine AOI class order
    aoi_levels = sorted(summary["class_name"].astype(str).unique().tolist())
    group_levels = [g for g in ["Low", "High"] if g in set(summary["group_value"].astype(str))]
    if not group_levels:
        group_levels = sorted(summary["group_value"].astype(str).unique().tolist())

    # WWR order
    wwr_levels = [15, 45, 75]

    fig, axes = plt.subplots(
        nrows=max(1, len(group_levels)),
        ncols=max(1, len(aoi_levels)),
        figsize=(4.2 * max(1, len(aoi_levels)), 3.2 * max(1, len(group_levels))),
        sharey=True,
        sharex=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        # one row
        axes = axes.reshape((1, -1))

    colors = {"C0": "#4C78A8", "C1": "#F58518"}

    for i, gv in enumerate(group_levels):
        for j, aoi in enumerate(aoi_levels):
            ax = axes[i, j] if (i < axes.shape[0] and j < axes.shape[1]) else None
            if ax is None:
                continue

            sub = summary[(summary["group_value"].astype(str) == str(gv)) & (summary["class_name"].astype(str) == str(aoi))].copy()
            if sub.empty:
                ax.set_axis_off()
                continue

            for cx in ["C0", "C1"]:
                s2 = sub[sub["Complexity"].astype(str) == cx].copy()
                if s2.empty:
                    continue
                # align to wwr_levels
                xs = []
                ys = []
                lo = []
                hi = []
                for w in wwr_levels:
                    r = s2[s2["WWR"] == w]
                    if len(r) == 0:
                        xs.append(w)
                        ys.append(np.nan)
                        lo.append(np.nan)
                        hi.append(np.nan)
                    else:
                        rr = r.iloc[0]
                        xs.append(w)
                        ys.append(float(rr["mean"]))
                        lo.append(float(rr["ci_low"]))
                        hi.append(float(rr["ci_high"]))
                ax.plot(xs, ys, marker="o", linewidth=1.8, color=colors.get(cx, "#666666"), label=cx)
                ax.fill_between(xs, lo, hi, color=colors.get(cx, "#666666"), alpha=0.18, linewidth=0)

            ax.set_title(f"{group_var}={gv} | AOI={aoi}")
            ax.set_xticks(wwr_levels)
            ax.set_xlabel("WWR")
            if j == 0:
                ax.set_ylabel(outcome)
            ax.grid(True, axis="y", alpha=0.25)

    # one legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False, title="Complexity")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Summarize AOI metrics by WWR×Complexity×Group and export PNGs")
    ap.add_argument("--aoi_class_csv", required=True)
    ap.add_argument("--group_manifest", required=True)
    ap.add_argument("--group_id_col", default="name")
    ap.add_argument("--outdir", default="outputs_aoi_summary")
    ap.add_argument("--include_round", action="store_true", help="If set, also stratify summaries by round")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    _ensure_dirs(outdir)

    df = pd.read_csv(args.aoi_class_csv, encoding="utf-8-sig")
    gm = pd.read_csv(args.group_manifest, encoding="utf-8-sig")

    if args.group_id_col not in gm.columns:
        raise SystemExit(f"group_manifest missing id col: {args.group_id_col}")

    gm = gm.copy()
    gm["participant_id"] = gm[args.group_id_col].astype(str).str.strip()
    for c in ["SportFreq", "Experience"]:
        if c in gm.columns:
            gm[c] = gm[c].apply(_norm_hilo)

    df["participant_id"] = df["participant_id"].astype(str).str.strip()
    df = df.merge(gm[[c for c in ["participant_id", "SportFreq", "Experience"] if c in gm.columns]], on="participant_id", how="left")

    # core columns
    if "WWR" not in df.columns or "Complexity" not in df.columns:
        raise SystemExit("Missing WWR/Complexity columns. Re-run batch_aoi_metrics.py after latest patch.")

    df["WWR"] = _safe_num(df["WWR"]).astype("Int64")
    df["Complexity"] = df["Complexity"].apply(_complexity_to_label)

    if "round" in df.columns:
        df["round"] = _safe_num(df["round"]).astype("Int64")

    # Exclude flagged trials
    if "trial_excluded" in df.columns:
        df["trial_excluded"] = _safe_num(df["trial_excluded"]).fillna(0).astype(int)
        df = df[df["trial_excluded"] == 0].copy()

    # outcomes
    dwell_col = _pick_col(df, "dwell_time_ms", "TFD")
    ttff_col = _pick_col(df, "TTFF_ms", "TTFF")
    fc_col = _pick_col(df, "fixation_count", "FC")

    outcomes = []
    if dwell_col:
        df["dwell_ms"] = _safe_num(df[dwell_col]).clip(lower=0)
        df["dwell_y"] = np.log1p(df["dwell_ms"])
        outcomes += [("dwell_ms", "dwell_time_ms"), ("dwell_y", "log1p(dwell)")]

        # share within participant x scene (raw)
        key_scene = "scene_id_raw" if "scene_id_raw" in df.columns else ("scene_id" if "scene_id" in df.columns else None)
        if key_scene:
            df2 = _make_share(df, "dwell_ms", ["participant_id", key_scene])
            df2["share_logit"] = np.log((df2["share"] + 1e-6) / (1 - df2["share"] + 1e-6))
            df_share = df2
            outcomes += [("share", "share(dwell within trial)"), ("share_logit", "logit(share)")]
        else:
            df_share = df
    else:
        df_share = df

    if ttff_col:
        df["ttff_ms"] = _safe_num(df[ttff_col])
        df["ttff_y"] = np.log1p(df["ttff_ms"].clip(lower=0))
        outcomes += [("ttff_ms", "TTFF_ms"), ("ttff_y", "log1p(TTFF)")]

    if fc_col:
        df["fc"] = _safe_num(df[fc_col]).clip(lower=0)
        df["fc_y"] = np.log1p(df["fc"])
        outcomes += [("fc", "fixation_count"), ("fc_y", "log1p(fix_count)")]

    if "visited" in df.columns:
        df["visited"] = _safe_num(df["visited"]).fillna(0).astype(int)

    group_vars = [g for g in ["Experience", "SportFreq"] if g in df.columns]
    if not group_vars:
        raise SystemExit("No Experience/SportFreq columns found after merge.")

    # prepare base grouping
    base_cols = ["class_name", "WWR", "Complexity"]
    if args.include_round and ("round" in df.columns) and df["round"].notna().any():
        base_cols = ["round"] + base_cols

    # Generate summaries + plots
    for gv in group_vars:
        d0 = df_share.copy() if any(o[0].startswith("share") for o in outcomes) else df.copy()

        for ycol, yname in outcomes:
            if ycol not in d0.columns:
                continue

            # For TTFF/fix_count: optional conditional on visited==1 for interpretability
            d = d0.copy()
            if ycol.startswith("ttff") or ycol.startswith("fc"):
                if "visited" in d.columns:
                    d = d[d["visited"] == 1].copy()

            d = d.dropna(subset=[ycol, gv, "WWR", "Complexity", "class_name"])
            if d.empty:
                continue

            rows = []
            grp_cols = base_cols + [gv]
            for keys, sub in d.groupby(grp_cols, dropna=False):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                key_map = dict(zip(grp_cols, keys))
                mean, lo, hi, n = _ci95(sub[ycol])
                row = {
                    "group_var": gv,
                    "group_value": key_map[gv],
                    "class_name": key_map["class_name"],
                    "WWR": int(key_map["WWR"]) if pd.notna(key_map["WWR"]) else np.nan,
                    "Complexity": key_map["Complexity"],
                    "outcome": yname,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": n,
                }
                if "round" in key_map:
                    row["round"] = int(key_map["round"]) if pd.notna(key_map["round"]) else np.nan
                rows.append(row)

            summ = pd.DataFrame(rows)
            # normalize group values
            summ["group_value"] = summ["group_value"].apply(_norm_hilo)
            summ.to_csv(outdir / "tables" / f"summary_{gv}_{ycol}.csv", index=False, encoding="utf-8-sig")

            # plots (not stratifying by round by default to keep simple)
            title = f"AOI outcome by WWR×Complexity×{gv} ({yname})"
            out_png = outdir / "plots" / f"plot_{gv}_{ycol}.png"
            plot_grid(summ, out_png, outcome=yname, group_var=gv, title=title)

    # Run info
    (outdir / "RUNINFO.txt").write_text(
        "AOI summary by condition/group\n"
        f"aoi_class_csv: {args.aoi_class_csv}\n"
        f"group_manifest: {args.group_manifest}\n"
        f"group_id_col: {args.group_id_col}\n"
        f"include_round: {bool(args.include_round)}\n",
        encoding="utf-8",
    )

    print("Saved:", str(outdir))


if __name__ == "__main__":
    main()
