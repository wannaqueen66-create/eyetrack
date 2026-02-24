#!/usr/bin/env python3
"""Summarize AOI outcomes by group (SportFreq / Experience) like heatmap grouping.

Inputs
- --aoi_class_csv: outputs from batch_aoi_metrics.py (by class)
  must include: participant_id, scene_id, class_name, visited, TTFF_ms, dwell_time_ms, fixation_count
- --group_manifest: group_manifest.csv with columns: name,SportFreq,Experience

Outputs
- aoi_group_summary.csv
  Aggregated at:
    scene_id × class_name × group_type × group_value
  Where group_type in {SportFreq, Experience, SportFreq_x_Experience}.

Notes
- TTFF is summarized conditionally given visited==1.
- visited_rate is computed as mean(visited).
- dwell_time can be summarized both overall (with NaN for unvisited unless dwell_empty_as_zero was used)
  and conditionally given visited==1.

This script is meant to provide paper-friendly, auditable group summaries.
"""

import argparse
import os

import numpy as np
import pandas as pd


def _norm_group(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    s2 = s.lower()
    mapping = {
        "high": "High",
        "low": "Low",
        "h": "High",
        "l": "Low",
        "1": "High",
        "0": "Low",
        "yes": "High",
        "no": "Low",
        "true": "High",
        "false": "Low",
    }
    return mapping.get(s2, s)


def _safe_num(s):
    return pd.to_numeric(s, errors="coerce")


def _save_bar(df: pd.DataFrame, x: str, y: str, hue: str, out_path: str, title: str):
    import matplotlib.pyplot as plt

    if df.empty:
        return

    # small helper: bar plot using group means already computed
    plt.figure(figsize=(9, 4.8))
    # pivot to make grouped bars stable
    # We'll plot one panel per class using subplots if many classes
    classes = sorted(df["class_name"].dropna().unique().tolist())
    n_cls = len(classes)

    # If too many classes, just save one combined (faceting kept minimal)
    if n_cls <= 4:
        fig, axes = plt.subplots(1, n_cls, figsize=(4.6 * n_cls, 4.2), sharey=True)
        if n_cls == 1:
            axes = [axes]
        for ax, cls in zip(axes, classes):
            sub = df[df["class_name"] == cls]
            # order group values for nicer display
            order = ["High", "Low"]
            gvals = [g for g in order if g in set(sub[hue])]
            if not gvals:
                gvals = sorted(sub[hue].unique().tolist())
            xs = sorted(sub[x].unique().tolist())
            width = 0.35
            for i, gv in enumerate(gvals):
                yv = [float(sub[(sub[x] == xx) & (sub[hue] == gv)][y].iloc[0]) if len(sub[(sub[x] == xx) & (sub[hue] == gv)]) else np.nan for xx in xs]
                ax.bar([j + (i - (len(gvals)-1)/2)*width for j in range(len(xs))], yv, width=width, label=str(gv))
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels(xs, rotation=0)
            ax.set_title(f"{cls}")
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel(y)
        axes[-1].legend(title=hue, loc="best")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    else:
        # fallback: only save overall without faceting
        fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
        sub = df.copy()
        xs = sorted(sub[x].unique().tolist())
        gvals = sorted(sub[hue].unique().tolist())
        width = 0.8 / max(1, len(gvals))
        for i, gv in enumerate(gvals):
            yv = [float(sub[(sub[x] == xx) & (sub[hue] == gv)][y].mean()) for xx in xs]
            ax.bar([j + (i - (len(gvals)-1)/2)*width for j in range(len(xs))], yv, width=width, label=str(gv))
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs)
        ax.set_ylabel(y)
        ax.set_title(title)
        ax.legend(title=hue)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def summarize(df: pd.DataFrame, group_col: str, group_type: str) -> pd.DataFrame:
    out_rows = []
    for (scene_id, class_name, gval), sub in df.groupby(["scene_id", "class_name", group_col], dropna=False):
        if pd.isna(gval) or str(gval).strip() == "":
            continue

        visited = _safe_num(sub["visited"]).fillna(0)
        visited_rate = float(visited.mean()) if len(visited) else np.nan
        n = int(len(sub))

        # conditional (visited==1)
        sub_v = sub[visited.astype(int) == 1]

        ttff = _safe_num(sub_v.get("TTFF_ms", pd.Series([], dtype=float)))
        dwell = _safe_num(sub.get("dwell_time_ms", pd.Series([], dtype=float)))
        dwell_v = _safe_num(sub_v.get("dwell_time_ms", pd.Series([], dtype=float)))
        fix = _safe_num(sub.get("fixation_count", pd.Series([], dtype=float)))

        row = {
            "scene_id": scene_id,
            "class_name": class_name,
            "group_type": group_type,
            "group_value": gval,
            "n_trials": n,
            "visited_rate": visited_rate,
            "n_visited": int(len(sub_v)),
            "TTFF_mean_given_visited": float(ttff.mean()) if ttff.notna().any() else np.nan,
            "TTFF_median_given_visited": float(ttff.median()) if ttff.notna().any() else np.nan,
            "dwell_mean_all": float(dwell.mean()) if dwell.notna().any() else np.nan,
            "dwell_median_all": float(dwell.median()) if dwell.notna().any() else np.nan,
            "dwell_mean_given_visited": float(dwell_v.mean()) if dwell_v.notna().any() else np.nan,
            "dwell_median_given_visited": float(dwell_v.median()) if dwell_v.notna().any() else np.nan,
            "fixation_count_mean_all": float(fix.mean()) if fix.notna().any() else np.nan,
            "fixation_count_median_all": float(fix.median()) if fix.notna().any() else np.nan,
        }
        out_rows.append(row)

    return pd.DataFrame(out_rows)


def main():
    ap = argparse.ArgumentParser(description="Summarize AOI outcomes by SportFreq/Experience groups")
    ap.add_argument("--aoi_class_csv", required=True, help="batch_aoi_metrics_by_class.csv")
    ap.add_argument("--group_manifest", required=True, help="group_manifest.csv (name,SportFreq,Experience)")
    ap.add_argument("--outdir", default="outputs_aoi_groups")
    ap.add_argument("--id_col", default="name", help="ID column name in group_manifest (default: name)")

    # Plots
    ap.add_argument("--plots", action="store_true", help="If set, export paper-friendly plots (visited_rate and conditional TTFF/dwell) by group")
    ap.add_argument("--plot_format", default="png", choices=["png", "pdf"], help="Plot format (default: png)")
    ap.add_argument("--quiet", action="store_true", help="Reduce console output")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    aoi = pd.read_csv(args.aoi_class_csv)
    gm = pd.read_csv(args.group_manifest)

    if args.id_col not in gm.columns:
        raise SystemExit(f"group_manifest missing id_col={args.id_col!r}. Available: {list(gm.columns)}")

    gm = gm.copy()
    gm["participant_id"] = gm[args.id_col].astype(str).str.strip()
    for c in ["SportFreq", "Experience"]:
        if c in gm.columns:
            gm[c] = gm[c].apply(_norm_group)

    # Merge group labels onto AOI table
    aoi = aoi.copy()
    aoi["participant_id"] = aoi["participant_id"].astype(str).str.strip()
    merged = aoi.merge(gm[["participant_id", "SportFreq", "Experience"]], on="participant_id", how="left")

    # 4-way cross
    merged["SportFreq_x_Experience"] = merged["SportFreq"].astype(str) + "_" + merged["Experience"].astype(str)

    out = []
    if "SportFreq" in merged.columns:
        out.append(summarize(merged, "SportFreq", "SportFreq"))
    if "Experience" in merged.columns:
        out.append(summarize(merged, "Experience", "Experience"))
    out.append(summarize(merged, "SportFreq_x_Experience", "SportFreq_x_Experience"))

    out_df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    out_path = os.path.join(args.outdir, "aoi_group_summary.csv")
    out_df.to_csv(out_path, index=False)

    # Also save merged analysis-ready table
    merged_path = os.path.join(args.outdir, "aoi_with_groups.csv")
    merged.to_csv(merged_path, index=False)

    if args.plots:
        plots_dir = os.path.join(args.outdir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # For SportFreq / Experience: x=scene_id, hue=group_value
        for gt, metric, ycol, title in [
            ("SportFreq", "visited_rate", "visited_rate", "Visited rate by SportFreq"),
            ("Experience", "visited_rate", "visited_rate", "Visited rate by Experience"),
            ("SportFreq", "TTFF_median_given_visited", "TTFF_median_given_visited", "TTFF (median | visited) by SportFreq"),
            ("Experience", "TTFF_median_given_visited", "TTFF_median_given_visited", "TTFF (median | visited) by Experience"),
            ("SportFreq", "dwell_median_given_visited", "dwell_median_given_visited", "Dwell (median | visited) by SportFreq"),
            ("Experience", "dwell_median_given_visited", "dwell_median_given_visited", "Dwell (median | visited) by Experience"),
        ]:
            sdf = out_df[out_df["group_type"] == gt].copy()
            if sdf.empty:
                continue
            outp = os.path.join(plots_dir, f"{gt}_{metric}.{args.plot_format}")
            _save_bar(sdf, x="scene_id", y=ycol, hue="group_value", out_path=outp, title=title)

    if not args.quiet:
        print("Saved:")
        print(" -", out_path)
        print(" -", merged_path)
        if args.plots:
            print(" -", os.path.join(args.outdir, "plots"))


if __name__ == "__main__":
    main()
