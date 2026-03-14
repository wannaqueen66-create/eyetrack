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
- outdir/grouped/png/*.png
- outdir/grouped/data/*_data.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `import src.*` works when running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figure_style import apply_paper_style, soften_axes, PALETTE, metric_label, annotate_series_smart, metric_value_label
from src.aoi_metrics import normalize_aoi_class_series
from src.manifest_scene_order import attach_manifest_trial_metadata, normalize_condition_token


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


def _normalize_scene_token(x) -> str | None:
    return normalize_condition_token(x)


def _parse_round_index(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    m = re.search(r'(?:组|ROUND|BLOCK|GROUP)\s*([12])', s, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return np.nan
    return np.nan


def _build_scene_fields(scene_id, scene_label=None, scene_order=None, round_index=None):
    sid = "" if pd.isna(scene_id) else str(scene_id).strip()
    cond = _normalize_scene_token(scene_label if scene_label is not None else sid)
    if cond is None:
        cond = _normalize_scene_token(sid)

    if pd.isna(round_index):
        round_index = _parse_round_index(sid)

    wwr_order = np.nan
    cond_order = np.nan
    if cond:
        upper = cond.upper()
        if 'WWR15' in upper:
            wwr_order = 15
        elif 'WWR45' in upper:
            wwr_order = 45
        elif 'WWR75' in upper:
            wwr_order = 75
        if 'C0' in upper:
            cond_order = 0
        elif 'C1' in upper:
            cond_order = 1

    scene_label_final = str(scene_label).strip() if scene_label is not None and not pd.isna(scene_label) and str(scene_label).strip() else sid
    if cond:
        scene_label_final = cond

    if pd.notna(round_index) and cond:
        scene_display = f"R{int(round_index)} {cond}"
    elif pd.notna(round_index):
        scene_display = f"R{int(round_index)} {scene_label_final}"
    else:
        scene_display = scene_label_final

    return {
        "scene_id": sid,
        "condition_id": cond,
        "scene_label": scene_label_final,
        "scene_display": scene_display,
        "scene_order": scene_order,
        "round_index": round_index,
        "wwr_order": wwr_order,
        "cond_order": cond_order,
    }


def _apply_be_style():
    apply_paper_style()


def _metric_col(df: pd.DataFrame, primary: str, fallback: str | None = None) -> str | None:
    if primary in df.columns:
        return primary
    if fallback and fallback in df.columns:
        return fallback
    return None


def _ensure_dirs(base: Path):
    (base / 'by_scene').mkdir(parents=True, exist_ok=True)
    (base / 'by_participant').mkdir(parents=True, exist_ok=True)
    (base / 'grouped' / 'tables').mkdir(parents=True, exist_ok=True)
    (base / 'grouped' / 'png').mkdir(parents=True, exist_ok=True)
    (base / 'grouped' / 'data').mkdir(parents=True, exist_ok=True)


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


def _group_summary(df: pd.DataFrame, group_col: str, level: str = "scene") -> pd.DataFrame:
    rows = []
    tfd_col = _metric_col(df, "TFD", "dwell_time_ms")
    tff_col = _metric_col(df, "TTFF", "TTFF")
    fc_col = _metric_col(df, "FC", "fixation_count")
    ffd_col = _metric_col(df, "FFD")
    mfd_col = _metric_col(df, "MFD")
    rff_col = _metric_col(df, "RFF", "RF")
    mpd_col = _metric_col(df, "MPD")

    if level == "condition":
        id_col = "condition_id"
        label_col = "condition_label"
        base_row = {"summary_level": "condition"}
    else:
        id_col = "scene_id"
        label_col = "scene_display"
        base_row = {"summary_level": "scene"}

    group_keys = [id_col, label_col, "class_name", group_col]
    for extra in ["scene_order", "round_index", "wwr_order", "cond_order"]:
        if extra in df.columns and df[extra].notna().any():
            group_keys.append(extra)

    for keys, sub in df.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_keys, keys))
        gv = key_map[group_col]
        if pd.isna(gv) or str(gv).strip() == "":
            continue
        v = _safe_num(sub.get("visited", 0)).fillna(0)
        sub_v = sub[v.astype(int) == 1]

        row = {
            **base_row,
            id_col: key_map[id_col],
            label_col: key_map[label_col],
            "class_name": key_map["class_name"],
            "group_col": group_col,
            "group_value": gv,
            "n_trials": int(len(sub)),
            "visited_rate": float(v.mean()) if len(v) else np.nan,
        }
        for extra in ["scene_order", "round_index", "wwr_order", "cond_order"]:
            if extra in key_map:
                row[extra] = key_map[extra]
        if tff_col:
            tt = _safe_num(sub_v.get(tff_col))
            row["TTFF_mean_given_visited"] = float(tt.mean()) if tt.notna().any() else np.nan
        if tfd_col:
            td = _safe_num(sub.get(tfd_col))
            row["TFD_mean_all"] = float(td.mean()) if td.notna().any() else np.nan
            tdv = _safe_num(sub_v.get(tfd_col))
            row["TFD_mean_given_visited"] = float(tdv.mean()) if tdv.notna().any() else np.nan
        if fc_col:
            fc = _safe_num(sub.get(fc_col))
            row["FC_mean_all"] = float(fc.mean()) if fc.notna().any() else np.nan
            fcv = _safe_num(sub_v.get(fc_col))
            row["FC_mean_given_visited"] = float(fcv.mean()) if fcv.notna().any() else np.nan
        if "FC_share" in sub.columns:
            fcs = _safe_num(sub.get("FC_share"))
            row["FC_share_mean_all"] = float(fcs.mean()) if fcs.notna().any() else np.nan
            fcsv = _safe_num(sub_v.get("FC_share"))
            row["FC_share_mean_given_visited"] = float(fcsv.mean()) if fcsv.notna().any() else np.nan
        if "FC_rate" in sub.columns:
            fcr = _safe_num(sub.get("FC_rate"))
            row["FC_rate_mean_all"] = float(fcr.mean()) if fcr.notna().any() else np.nan
            fcrv = _safe_num(sub_v.get("FC_rate"))
            row["FC_rate_mean_given_visited"] = float(fcrv.mean()) if fcrv.notna().any() else np.nan
        if "share_pct" in sub.columns:
            shp = _safe_num(sub.get("share_pct"))
            row["share_pct_mean_all"] = float(shp.mean()) if shp.notna().any() else np.nan
            shpv = _safe_num(sub_v.get("share_pct"))
            row["share_pct_mean_given_visited"] = float(shpv.mean()) if shpv.notna().any() else np.nan
        if ffd_col:
            ffd = _safe_num(sub_v.get(ffd_col))
            row["FFD_mean_given_visited"] = float(ffd.mean()) if ffd.notna().any() else np.nan
        if mfd_col:
            mfd = _safe_num(sub_v.get(mfd_col))
            row["MFD_mean_given_visited"] = float(mfd.mean()) if mfd.notna().any() else np.nan
        if rff_col:
            rf = _safe_num(sub.get(rff_col))
            row["RFF_mean_all"] = float(rf.mean()) if rf.notna().any() else np.nan
            rfv = _safe_num(sub_v.get(rff_col))
            row["RFF_mean_given_visited"] = float(rfv.mean()) if rfv.notna().any() else np.nan
        if mpd_col:
            mpd = _safe_num(sub_v.get(mpd_col))
            row["MPD_mean_given_visited"] = float(mpd.mean()) if mpd.notna().any() else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def _format_group_metric_label(metric: str, value: float) -> str:
    if pd.isna(value):
        return ""
    if "visited_rate" in metric:
        return f"{value * 100:.1f}%"
    metric_base = metric.replace('_mean_given_visited','').replace('_mean_all','')
    return metric_value_label(metric_base, value)


def _plot_group_metric(summary_df: pd.DataFrame, metric: str, out_png: Path, title: str):
    if summary_df.empty or metric not in summary_df.columns:
        return

    _apply_be_style()
    data = summary_df.copy()
    data[metric] = _safe_num(data[metric])
    data = data.dropna(subset=[metric])
    if data.empty:
        return

    label_col = "scene_display" if "scene_display" in data.columns else ("condition_label" if "condition_label" in data.columns else ("scene_label" if "scene_label" in data.columns else ("condition_id" if "condition_id" in data.columns else "scene_id")))

    preferred_order_cols = [c for c in ["scene_order", "round_index", "wwr_order", "cond_order"] if c in data.columns and data[c].notna().any()]
    if preferred_order_cols:
        data["plot_order_key"] = data[preferred_order_cols].astype(str).agg("|".join, axis=1)
        key_cols = preferred_order_cols + ["plot_order_key", label_col, "group_value"]
        agg = data.groupby(key_cols, as_index=False, dropna=False)[metric].mean(numeric_only=True)
        agg = agg.sort_values(preferred_order_cols + [label_col, "group_value"])
        order_df = agg[preferred_order_cols + ["plot_order_key", label_col]].drop_duplicates(subset=["plot_order_key"])  # one label per slot
        plot_slots = order_df["plot_order_key"].astype(str).tolist()
        slot_to_label = dict(zip(order_df["plot_order_key"].astype(str), order_df[label_col].astype(str)))
        round_boundaries = order_df["round_index"].tolist() if "round_index" in order_df.columns else None
    else:
        fallback_id_col = "scene_id" if "scene_id" in data.columns else ("condition_id" if "condition_id" in data.columns else label_col)
        tmp = data[[fallback_id_col, label_col, "group_value", metric]].copy()
        agg = tmp.groupby([fallback_id_col, label_col, "group_value"], as_index=False, dropna=False)[metric].mean(numeric_only=True)
        order_df = agg[[fallback_id_col, label_col]].drop_duplicates()
        plot_slots = order_df[fallback_id_col].astype(str).tolist()
        slot_to_label = dict(zip(order_df[fallback_id_col].astype(str), order_df[label_col].astype(str)))
        agg["plot_order_key"] = agg[fallback_id_col].astype(str)
        round_boundaries = None

    groups = [g for g in ["Low", "High"] if g in set(agg["group_value"].astype(str))]
    if not groups:
        groups = sorted(agg["group_value"].astype(str).unique())

    width = 0.58 / max(1, len(groups))
    x = np.arange(len(plot_slots))

    fig_w = max(12.8, 0.95 * len(plot_slots) + 3.6)
    fig, ax = plt.subplots(figsize=(fig_w, 5.2))
    palette_map = {"Low": PALETTE["blue"], "High": PALETTE["orange"]}
    fallback_palette = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["purple"]]

    plot_rows = []
    ymax_all = []
    for i, g in enumerate(groups):
        y = []
        for slot in plot_slots:
            tmp = agg[(agg["plot_order_key"].astype(str) == str(slot)) & (agg["group_value"].astype(str) == g)][metric]
            y.append(float(tmp.iloc[0]) if len(tmp) else np.nan)
        offs = x + (i - (len(groups) - 1) / 2) * width
        color = palette_map.get(g, fallback_palette[i % len(fallback_palette)])
        bars = ax.bar(offs, y, width=width, label=g, color=color, alpha=0.88, edgecolor='white', linewidth=0.9)
        vals_for_group = []
        xs_for_group = []
        for rect, val, slot in zip(bars, y, plot_slots):
            plot_rows.append({
                "plot_order_key": slot,
                "plot_label": slot_to_label.get(slot, slot),
                "group_value": g,
                "metric": metric,
                "value": val,
                "value_label": _format_group_metric_label(metric, val),
                "out_png": out_png.name,
            })
            if pd.isna(val):
                continue
            ymax_all.append(val)
            xs_for_group.append(rect.get_x() + rect.get_width() / 2)
            vals_for_group.append(val)
        annotate_series_smart(ax, xs_for_group, vals_for_group, metric=metric.replace('_mean_given_visited','').replace('_mean_all',''), color=color, max_labels=3)

    if round_boundaries is not None:
        for i in range(1, len(round_boundaries)):
            if pd.notna(round_boundaries[i]) and pd.notna(round_boundaries[i - 1]) and round_boundaries[i] != round_boundaries[i - 1]:
                ax.axvline(i - 0.5, color=PALETTE['gray'], linestyle='--', linewidth=1.0, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([slot_to_label[s] for s in plot_slots], rotation=28, ha="right")
    ax.set_title(title, pad=10)
    ax.set_xlabel("Condition" if "condition" in str(out_png).lower() else "Scene")
    ax.set_ylabel(metric_label(metric.replace('_mean_given_visited','').replace('_mean_all','')) if metric else metric)
    soften_axes(ax)
    if ymax_all:
        ymax = max(ymax_all)
        ax.set_ylim(top=float(ymax) * 1.28 if float(ymax) > 0 else 1.0)
    leg = ax.legend(title="Group", frameon=False, ncol=min(2, max(1, len(groups))))
    if leg is not None:
        for lh in leg.legendHandles:
            try:
                lh.set_alpha(0.88)
            except Exception:
                pass
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    target = (out_png.parent.parent / 'data' / f"{out_png.stem}_data.csv") if out_png.parent.name == 'png' else out_png.with_name(out_png.stem + '_data.csv')
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(plot_rows).to_csv(target, index=False, encoding='utf-8-sig')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Reorganize AOI outputs + grouped summaries + B&E-style plots")
    ap.add_argument("--aoi_class_csv", required=True)
    ap.add_argument("--aoi_polygon_csv", default=None)
    ap.add_argument("--group_manifest", default=None)
    ap.add_argument("--group_id_col", default="name", help="id column in group manifest (default: name)")
    ap.add_argument("--outdir", default="outputs_organized")
    ap.add_argument("--skip_if_exists", action="store_true", help="Skip when outdir already appears populated")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if args.skip_if_exists and (outdir / 'by_scene').exists() and (outdir / 'grouped').exists():
        print('Skip optimize: output already exists ->', outdir)
        return
    _ensure_dirs(outdir)

    df_class = pd.read_csv(args.aoi_class_csv)
    df_poly = pd.read_csv(args.aoi_polygon_csv) if args.aoi_polygon_csv and os.path.exists(args.aoi_polygon_csv) else None

    for c in ["participant_id", "scene_id", "class_name"]:
        if c not in df_class.columns:
            raise SystemExit(f"aoi_class_csv missing required column: {c}")

    df_class['class_name'] = normalize_aoi_class_series(df_class['class_name'])
    if df_poly is not None and 'class_name' in df_poly.columns:
        df_poly['class_name'] = normalize_aoi_class_series(df_poly['class_name'])

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

        # Scene labels/order from manifest columns like trial01_scene.
        # We attach metadata by original scene_id when possible, but always fall back to parsing scene_id itself.
        scene_meta_rows = []
        trial_scene_cols = sorted([c for c in gm.columns if c.endswith('_scene')])
        for c in trial_scene_cols:
            prefix = c[:-6]  # remove _scene, e.g. trial01
            digits = ''.join(ch for ch in prefix if ch.isdigit())
            trial_num = int(digits) if digits else np.nan
            round_col = f"{prefix}_Round"
            round_index = np.nan
            if round_col in gm.columns:
                vv = gm[round_col].dropna().tolist()
                if vv:
                    try:
                        round_index = float(vv[0])
                    except Exception:
                        round_index = np.nan
            if not np.isfinite(round_index) and pd.notna(trial_num):
                round_index = ((int(trial_num) - 1) // 6 + 1)

            vals = gm[c].dropna().astype(str).str.strip().unique().tolist()
            for scene_raw in vals:
                meta = _build_scene_fields(scene_raw, scene_label=scene_raw, scene_order=trial_num, round_index=round_index)
                scene_meta_rows.append(meta)

        scene_meta = pd.DataFrame(scene_meta_rows)
        if not scene_meta.empty:
            scene_meta = scene_meta.sort_values([col for col in ["scene_order", "round_index"] if col in scene_meta.columns], na_position="last")
            scene_meta = scene_meta.drop_duplicates(subset=["scene_id"], keep="first")

        d["scene_id"] = d["scene_id"].astype(str).str.strip()
        d = attach_manifest_trial_metadata(d, gm, id_col=args.group_id_col, scene_col="scene_id")
        if not scene_meta.empty and "scene_order" not in d.columns:
            d = d.merge(scene_meta, on="scene_id", how="left")

        fallback_scene = d.apply(
            lambda r: _build_scene_fields(
                r.get("scene_id"),
                scene_label=r.get("scene_label"),
                scene_order=r.get("scene_order"),
                round_index=r.get("round") if pd.notna(r.get("round")) else r.get("round_index"),
            ),
            axis=1,
            result_type="expand",
        )
        for col in ["condition_id", "scene_label", "scene_display", "scene_order", "round_index", "wwr_order", "cond_order"]:
            if col not in d.columns:
                d[col] = fallback_scene[col]
            else:
                d[col] = d[col].where(d[col].notna(), fallback_scene[col])

        if "round" not in d.columns:
            d["round"] = d["round_index"]
        else:
            round_num = _safe_num(d["round"])
            round_idx_num = _safe_num(d["round_index"])
            d["round"] = round_num.where(round_num.notna(), round_idx_num)
        d["round_index"] = _safe_num(d["round_index"]).where(_safe_num(d["round_index"]).notna(), _safe_num(d["round"]))

        d["condition_label"] = d["condition_id"].fillna(d["scene_label"])

        group_specs = []
        if "SportFreq" in d.columns:
            group_specs.append(("SportFreq", "sportfreq"))
        if "Experience" in d.columns:
            group_specs.append(("Experience", "experience"))

        metrics = [
            ("visited_rate", "Visited rate"),
            ("share_pct_mean_all", "share_pct (all trials)"),
            ("TTFF_mean_given_visited", "TTFF (visited trials)"),
            ("TFD_mean_given_visited", "TFD (visited trials)"),
            ("FC_share_mean_all", "FC_share (all trials)"),
            ("FC_rate_mean_all", "FC_rate (all trials)"),
            ("FC_mean_given_visited", "FC (visited trials)"),
            ("FFD_mean_given_visited", "FFD (visited trials)"),
            ("MFD_mean_given_visited", "MFD (visited trials)"),
            ("RFF_mean_given_visited", "RFF (visited trials)"),
            ("MPD_mean_given_visited", "MPD (visited trials)"),
        ]

        for gv, stem in group_specs:
            scene_summary = _group_summary(d, gv, level="scene")
            scene_summary.to_csv(outdir / "grouped" / "tables" / f"summary_{stem}_scene.csv", index=False)
            for metric, title_base in metrics:
                _plot_group_metric(scene_summary, metric, outdir / 'grouped' / 'png' / f"{stem}_scene_{metric.replace('_mean_given_visited','').replace('_mean_all','')}.png", f"{title_base} by {gv} (scene-level)")

            condition_df = d[d["condition_id"].notna()].copy()
            if not condition_df.empty:
                condition_summary = _group_summary(condition_df, gv, level="condition")
                condition_summary.to_csv(outdir / "grouped" / "tables" / f"summary_{stem}_condition.csv", index=False)
                for metric, title_base in metrics:
                    _plot_group_metric(condition_summary, metric, outdir / 'grouped' / 'png' / f"{stem}_condition_{metric.replace('_mean_given_visited','').replace('_mean_all','')}.png", f"{title_base} by {gv} (condition-level)")

    print("Saved organized outputs to:", outdir)


if __name__ == "__main__":
    main()
