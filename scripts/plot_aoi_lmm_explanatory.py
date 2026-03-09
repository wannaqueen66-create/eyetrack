#!/usr/bin/env python3
"""Create explanatory PNGs for AOI LMM interpretation.

Focus:
- Show whether different groups redistribute visual attention across AOIs
  under WWR × Complexity conditions.
- Use AOI-level attention share as the primary allocation metric.
- Also export companion plots for TFD / TTFF / FC when available.

Outputs
-------
<outdir>/
  tables/
    summary_<GroupVar>_<metric>_condition.csv
    summary_<GroupVar>_<metric>_scene.csv
  png/
    condition_group_interaction_<GroupVar>_<metric>.png
    scene_group_profile_<GroupVar>_<metric>.png

Default metrics
---------------
- share_pct   : primary recommended metric, % of TFD within each trial allocated to each AOI
- FC_share    : fixation-count share within each trial allocated to each AOI
- FC_rate     : fixation-count rate per second
- TFD         : total fixation duration (ms)
- TTFF        : time to first fixation (ms), visited==1 only
- FC          : fixation count, visited==1 only
- FFD         : first fixation duration (ms), visited==1 only
- MFD         : mean fixation duration (ms), visited==1 only
- RFF         : re-fixation frequency
- MPD         : mean pupil diameter, visited==1 only
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.figure_style import apply_paper_style, soften_axes, PALETTE
from src.aoi_metrics import normalize_aoi_class_name
from src.manifest_scene_order import attach_manifest_trial_metadata


AOI_LABELS = {
    "table": "Pingpong table",
    "window": "Window",
    "windows": "Window",
    "equipment": "Equipment",
    "apparatus": "Equipment",
    "device": "Equipment",
    "facility": "Equipment",
    "net": "Net",
}

GROUP_LABELS = {
    "High": "High",
    "Low": "Low",
}

GROUPVAR_LABELS = {
    "Experience": "Experience group",
    "SportFreq": "Sport-frequency group",
}

METRIC_LABELS = {
    "share_pct": "Attention allocation share (%)",
    "FC_share": "Fixation-count share within trial",
    "FC_rate": "Fixation-count rate (FC/s)",
    "TFD": "Total fixation duration TFD (ms)",
    "TTFF": "Time to first fixation TTFF (ms)",
    "FC": "Fixation count FC",
    "FFD": "First fixation duration FFD (ms)",
    "MFD": "Mean fixation duration MFD (ms)",
    "RFF": "Re-fixation frequency RFF",
    "MPD": "Mean pupil diameter MPD",
}

METRIC_TITLES = {
    "share_pct": "attention allocation share",
    "FC_share": "fixation-count allocation share",
    "FC_rate": "fixation-count rate",
    "TFD": "total fixation duration",
    "TTFF": "time to first fixation",
    "FC": "fixation count",
    "FFD": "first fixation duration",
    "MFD": "mean fixation duration",
    "RFF": "re-fixation frequency",
    "MPD": "mean pupil diameter",
}

SCENE_ORDER_DEFAULT = [15, 45, 75]
COMPLEXITY_ORDER = ["C0", "C1"]
GROUP_ORDER = ["Low", "High"]
COLOR_MAP = {"Low": PALETTE["blue"], "High": PALETTE["orange"]}


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


def _complexity_to_label(x):
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    if s in ("0", "C0", "LOW", "L"):
        return "C0"
    if s in ("1", "C1", "HIGH", "H"):
        return "C1"
    m = re.search(r"C([01])", s)
    return f"C{m.group(1)}" if m else s


def _pick_col(df: pd.DataFrame, primary: str, fallback: str | None = None) -> str | None:
    if primary in df.columns:
        return primary
    if fallback and fallback in df.columns:
        return fallback
    return None


def _alias_aoi(name: str) -> str:
    s = str(name).strip()
    low = normalize_aoi_class_name(s)
    if low in AOI_LABELS:
        return low
    return low


def _aoi_label(name: str) -> str:
    key = _alias_aoi(name)
    return AOI_LABELS.get(key, str(name))


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


def _sort_scene_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "scene_order" in d.columns and d["scene_order"].notna().any():
        d["scene_order_ord"] = _safe_num(d["scene_order"]).fillna(999)
    else:
        d["scene_order_ord"] = 999
    if "round" in d.columns:
        d["round_ord"] = _safe_num(d["round"]).fillna(99)
    else:
        d["round_ord"] = 99
    d["WWR_ord"] = _safe_num(d["WWR"]).fillna(999)
    d["Complexity_ord"] = d["Complexity"].map({"C0": 0, "C1": 1}).fillna(9)
    return d.sort_values(["scene_order_ord", "round_ord", "WWR_ord", "Complexity_ord", "scene_label"])


def _build_scene_label(row: pd.Series) -> str:
    if pd.notna(row.get("scene_display")) and str(row.get("scene_display")).strip():
        return str(row.get("scene_display")).strip()
    wwr = row.get("WWR")
    comp = row.get("Complexity")
    rnd = row.get("round")
    base = None
    if pd.notna(wwr) and pd.notna(comp):
        base = f"WWR{int(wwr)}_{comp}"
    elif pd.notna(row.get("scene_label")) and str(row.get("scene_label")).strip():
        base = str(row.get("scene_label")).strip()
    elif pd.notna(row.get("scene_id")):
        base = str(row.get("scene_id"))
    else:
        base = "scene"
    if pd.notna(rnd):
        return f"R{int(rnd)} {base}"
    return base


def _ensure_dirs(outdir: Path):
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "png").mkdir(parents=True, exist_ok=True)


def _prepare_data(aoi_class_csv: str, group_manifest: str, group_id_col: str, aoi_classes: list[str] | None):
    df = pd.read_csv(aoi_class_csv, encoding="utf-8-sig")
    gm = pd.read_csv(group_manifest, encoding="utf-8-sig")
    if group_id_col not in gm.columns:
        raise SystemExit(f"group_manifest missing id col: {group_id_col}")

    gm = gm.copy()
    gm["participant_id"] = gm[group_id_col].astype(str).str.strip()
    for c in ["SportFreq", "Experience"]:
        if c in gm.columns:
            gm[c] = gm[c].apply(_norm_hilo)

    df["participant_id"] = df["participant_id"].astype(str).str.strip()
    df = df.merge(gm[[c for c in ["participant_id", "SportFreq", "Experience"] if c in gm.columns]], on="participant_id", how="left")

    if "trial_excluded" in df.columns:
        df["trial_excluded"] = _safe_num(df["trial_excluded"]).fillna(0).astype(int)
        df = df[df["trial_excluded"] == 0].copy()

    if "WWR" not in df.columns or "Complexity" not in df.columns:
        raise SystemExit("Missing WWR/Complexity columns. Re-run batch_aoi_metrics.py after latest patch.")

    df["WWR"] = _safe_num(df["WWR"]).astype("Int64")
    df["Complexity"] = df["Complexity"].apply(_complexity_to_label)
    if "round" in df.columns:
        df["round"] = _safe_num(df["round"]).astype("Int64")
    if "visited" in df.columns:
        df["visited"] = _safe_num(df["visited"]).fillna(0).astype(int)
    else:
        df["visited"] = 1

    dwell_col = _pick_col(df, "TFD", "dwell_time_ms")
    ttff_col = _pick_col(df, "TTFF")
    fc_col = _pick_col(df, "FC", "fixation_count")
    if not dwell_col:
        raise SystemExit("Missing TFD/dwell_time_ms column in aoi_class_csv")

    df["TFD"] = _safe_num(df[dwell_col]).clip(lower=0)
    if ttff_col:
        df["TTFF"] = _safe_num(df[ttff_col]).clip(lower=0)
    if fc_col:
        df["FC"] = _safe_num(df[fc_col]).clip(lower=0)
    if "FC_share" in df.columns:
        df["FC_share"] = _safe_num(df["FC_share"]).clip(lower=0, upper=1)
    if "FC_rate" in df.columns:
        df["FC_rate"] = _safe_num(df["FC_rate"]).clip(lower=0)
    if "FFD" in df.columns:
        df["FFD"] = _safe_num(df["FFD"]).clip(lower=0)
    if "MFD" in df.columns:
        df["MFD"] = _safe_num(df["MFD"]).clip(lower=0)
    if "RFF" in df.columns:
        df["RFF"] = _safe_num(df["RFF"]).clip(lower=0)
    if "MPD" in df.columns:
        df["MPD"] = _safe_num(df["MPD"])

    scene_col = "scene_id_raw" if "scene_id_raw" in df.columns else ("scene_id" if "scene_id" in df.columns else None)
    if scene_col:
        if "share_pct" in df.columns:
            df["share_pct"] = _safe_num(df["share_pct"]).clip(lower=0, upper=100)
        else:
            tot = df.groupby(["participant_id", scene_col], dropna=False)["TFD"].sum(min_count=1).rename("TFD_total")
            df = df.merge(tot.reset_index(), on=["participant_id", scene_col], how="left")
            df["share_pct"] = 100.0 * df["TFD"] / df["TFD_total"]
            df.loc[~np.isfinite(df["share_pct"]), "share_pct"] = np.nan
    else:
        df["share_pct"] = np.nan

    df["aoi_key"] = df["class_name"].map(_alias_aoi)
    df["aoi_label"] = df["class_name"].map(_aoi_label)

    wanted = [_alias_aoi(x) for x in (aoi_classes or ["table", "window", "equipment"])]
    keep = [x for x in wanted if x in set(df["aoi_key"])]
    if not keep:
        top = (df["aoi_key"].value_counts().head(3).index.tolist())
        keep = top
    df = df[df["aoi_key"].isin(keep)].copy()

    scene_src = scene_col if scene_col else "class_name"
    df["scene_id"] = df[scene_src].astype(str)
    df = attach_manifest_trial_metadata(df, gm, id_col=group_id_col, scene_col="scene_id")
    if "round" in df.columns:
        df["round"] = _safe_num(df["round"]).astype("Int64")
    if "WWR" in df.columns:
        df["WWR"] = _safe_num(df["WWR"]).astype("Int64")
    if "Complexity" in df.columns:
        df["Complexity"] = df["Complexity"].apply(_complexity_to_label)
    df["scene_label"] = df.apply(_build_scene_label, axis=1)

    group_vars = [g for g in ["Experience", "SportFreq"] if g in df.columns]
    if not group_vars:
        raise SystemExit("No Experience/SportFreq columns found after merge.")
    return df, group_vars


def _summarize(df: pd.DataFrame, metric: str, group_var: str, by_scene: bool) -> pd.DataFrame:
    d = df.copy()
    if metric in ["TTFF", "FC", "FFD", "MFD", "MPD"]:
        d = d[d["visited"] == 1].copy()
    d = d.dropna(subset=[metric, group_var, "WWR", "Complexity", "aoi_key"])
    if d.empty:
        return pd.DataFrame()

    grp_cols = ["aoi_key", "aoi_label", group_var, "WWR", "Complexity"]
    if by_scene:
        for extra in ["scene_order", "scene_view_id", "scene_label"]:
            if extra in d.columns:
                grp_cols += [extra]
        if "round" in d.columns:
            grp_cols += ["round"]
    rows = []
    for keys, sub in d.groupby(grp_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(grp_cols, keys))
        mean, lo, hi, n = _ci95(sub[metric])
        row = {
            "group_var": group_var,
            "group_value": _norm_hilo(key_map[group_var]),
            "aoi_key": key_map["aoi_key"],
            "aoi_label": key_map["aoi_label"],
            "WWR": int(key_map["WWR"]) if pd.notna(key_map["WWR"]) else np.nan,
            "Complexity": key_map["Complexity"],
            "metric": metric,
            "mean": mean,
            "ci_low": lo,
            "ci_high": hi,
            "n": n,
        }
        if by_scene:
            row["scene_label"] = key_map["scene_label"]
            if "scene_order" in key_map:
                row["scene_order"] = key_map["scene_order"]
            if "scene_view_id" in key_map:
                row["scene_view_id"] = key_map["scene_view_id"]
            if "round" in key_map and pd.notna(key_map["round"]):
                row["round"] = int(key_map["round"])
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    out["group_value"] = out["group_value"].apply(_norm_hilo)
    return out


def _ordered_groups(vals) -> list[str]:
    vals = [str(v) for v in pd.Series(vals).dropna().unique().tolist()]
    ordered = [g for g in GROUP_ORDER if g in vals]
    return ordered if ordered else sorted(vals)


def _ordered_aois(vals) -> list[str]:
    order = ["table", "window", "equipment"]
    vals = [str(v) for v in pd.Series(vals).dropna().unique().tolist()]
    out = [a for a in order if a in vals]
    out += [a for a in vals if a not in out]
    return out


def _format_metric_value(metric: str, value: float) -> str:
    if not np.isfinite(value):
        return ""
    if metric == "share_pct":
        return f"{value:.1f}%"
    if metric in {"TFD", "TTFF"}:
        return f"{value:.0f}"
    if metric == "FC":
        return f"{value:.1f}"
    return f"{value:.2f}"


def _annotate_series(ax, xs, ys, metric: str, color: str):
    points = [(float(x), float(y)) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
    if not points:
        return
    for idx, (x, y) in enumerate(points):
        dy = 6 if idx % 2 == 0 else -10
        ax.annotate(
            _format_metric_value(metric, y),
            xy=(x, y),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom" if dy >= 0 else "top",
            fontsize=7,
            color=color,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
        )


def _export_plot_companion(summary: pd.DataFrame, outdir: Path, stem: str):
    outdir.mkdir(parents=True, exist_ok=True)
    data_path = outdir / f"{stem}_data.csv"
    labels_path = outdir / f"{stem}_labels.csv"
    summary.to_csv(data_path, index=False, encoding="utf-8-sig")
    labels = summary.copy()
    labels["value_label"] = labels["mean"].apply(lambda v: _format_metric_value(str(labels["metric"].iloc[0]) if len(labels) else "", float(v) if pd.notna(v) else np.nan))
    labels.to_csv(labels_path, index=False, encoding="utf-8-sig")


def plot_condition_interaction(summary: pd.DataFrame, out_png: Path, group_var: str, metric: str):
    if summary.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    apply_paper_style()
    aois = _ordered_aois(summary["aoi_key"])
    groups = _ordered_groups(summary["group_value"])
    nrows = max(1, len(COMPLEXITY_ORDER))
    ncols = max(1, len(aois))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 3.6 * nrows), sharey=True, sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(nrows, ncols)

    for i, complexity in enumerate(COMPLEXITY_ORDER):
        for j, aoi in enumerate(aois):
            ax = axes[i, j]
            sub = summary[(summary["Complexity"] == complexity) & (summary["aoi_key"] == aoi)].copy()
            if sub.empty:
                ax.set_axis_off()
                continue
            for g in groups:
                s2 = sub[sub["group_value"].astype(str) == str(g)].copy().sort_values("WWR")
                if s2.empty:
                    continue
                xs = s2["WWR"].astype(float).tolist()
                ys = s2["mean"].astype(float).tolist()
                lo = s2["ci_low"].astype(float).tolist()
                hi = s2["ci_high"].astype(float).tolist()
                color = COLOR_MAP.get(g, PALETTE["gray"])
                ax.plot(xs, ys, marker="o", color=color, label=GROUP_LABELS.get(g, g))
                ax.fill_between(xs, lo, hi, color=color, alpha=0.16, linewidth=0)
                _annotate_series(ax, xs, ys, metric=metric, color=color)
            ax.set_xticks(SCENE_ORDER_DEFAULT)
            ax.set_xlabel("Window-to-wall ratio WWR (%)")
            if j == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric))
            ax.set_title(f"{_aoi_label(aoi)} | {complexity}")
            soften_axes(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False, title=GROUPVAR_LABELS.get(group_var, group_var))
    fig.suptitle(f"{GROUPVAR_LABELS.get(group_var, group_var)} × scene condition effects on AOI {METRIC_TITLES.get(metric, metric)}", y=1.02, fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scene_profile(summary: pd.DataFrame, out_png: Path, group_var: str, metric: str):
    if summary.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    apply_paper_style()
    summary = _sort_scene_df(summary)
    aois = _ordered_aois(summary["aoi_key"])
    groups = _ordered_groups(summary["group_value"])
    fig, axes = plt.subplots(nrows=len(aois), ncols=1, figsize=(12, 2.9 * len(aois)), sharex=True, sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    scene_labels = summary["scene_label"].dropna().astype(str).unique().tolist()
    xpos = {lab: i for i, lab in enumerate(scene_labels)}

    for ax, aoi in zip(axes, aois):
        sub = summary[summary["aoi_key"] == aoi].copy()
        if sub.empty:
            ax.set_axis_off()
            continue
        for g in groups:
            s2 = sub[sub["group_value"].astype(str) == str(g)].copy()
            s2["x"] = s2["scene_label"].map(xpos)
            s2 = s2.sort_values("x")
            color = COLOR_MAP.get(g, PALETTE["gray"])
            ax.plot(s2["x"], s2["mean"], marker="o", color=color, label=GROUP_LABELS.get(g, g))
            ax.fill_between(s2["x"], s2["ci_low"], s2["ci_high"], color=color, alpha=0.12, linewidth=0)
            _annotate_series(ax, s2["x"].tolist(), s2["mean"].tolist(), metric=metric, color=color)
        ax.set_title(_aoi_label(aoi), loc="left")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        soften_axes(ax)

    axes[-1].set_xticks(list(xpos.values()))
    axes[-1].set_xticklabels(scene_labels, rotation=35, ha="right")
    axes[-1].set_xlabel("Scene order (round × WWR × Complexity)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False, title=GROUPVAR_LABELS.get(group_var, group_var))
    fig.suptitle(f"{GROUPVAR_LABELS.get(group_var, group_var)} scene-level AOI {METRIC_TITLES.get(metric, metric)} profile", y=1.01, fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Export explanatory AOI PNGs for WWR × Complexity × Group interpretation")
    ap.add_argument("--aoi_class_csv", required=True)
    ap.add_argument("--group_manifest", required=True)
    ap.add_argument("--group_id_col", default="name")
    ap.add_argument("--outdir", default="outputs_aoi_lmm_visuals")
    ap.add_argument("--aoi_classes", default="table,window,equipment", help="Comma-separated AOI classes to prioritize")
    ap.add_argument("--metrics", default="share_pct,FC_share,FC_rate,TFD,TTFF,FC,FFD,MFD,RFF,MPD", help="Comma-separated metrics: share_pct,FC_share,FC_rate,TFD,TTFF,FC,FFD,MFD,RFF,MPD")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    _ensure_dirs(outdir)
    aoi_classes = [x.strip() for x in str(args.aoi_classes).split(",") if x.strip()]
    metrics = [x.strip() for x in str(args.metrics).split(",") if x.strip()]

    df, group_vars = _prepare_data(args.aoi_class_csv, args.group_manifest, args.group_id_col, aoi_classes)

    runinfo_lines = [
        "AOI explanatory LMM visuals",
        f"aoi_class_csv: {args.aoi_class_csv}",
        f"group_manifest: {args.group_manifest}",
        f"group_id_col: {args.group_id_col}",
        f"aoi_classes_requested: {','.join(aoi_classes)}",
        f"metrics: {','.join(metrics)}",
        f"aoi_classes_used: {','.join(df['aoi_key'].dropna().astype(str).unique().tolist())}",
    ]

    for gv in group_vars:
        for metric in metrics:
            if metric not in df.columns:
                continue
            cond = _summarize(df, metric=metric, group_var=gv, by_scene=False)
            scn = _summarize(df, metric=metric, group_var=gv, by_scene=True)
            if len(cond) == 0:
                continue
            cond.to_csv(outdir / "tables" / f"summary_{gv}_{metric}_condition.csv", index=False, encoding="utf-8-sig")
            _export_plot_companion(cond, outdir / "tables", f"condition_group_interaction_{gv}_{metric}")
            if len(scn):
                scn.to_csv(outdir / "tables" / f"summary_{gv}_{metric}_scene.csv", index=False, encoding="utf-8-sig")
                _export_plot_companion(scn, outdir / "tables", f"scene_group_profile_{gv}_{metric}")
            plot_condition_interaction(cond, outdir / "png" / f"condition_group_interaction_{gv}_{metric}.png", group_var=gv, metric=metric)
            if len(scn):
                plot_scene_profile(scn, outdir / "png" / f"scene_group_profile_{gv}_{metric}.png", group_var=gv, metric=metric)

    (outdir / "RUNINFO.txt").write_text("\n".join(runinfo_lines) + "\n", encoding="utf-8")
    print("Saved:", str(outdir))


if __name__ == "__main__":
    main()
