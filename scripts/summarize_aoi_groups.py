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

    print("Saved:")
    print(" -", out_path)
    print(" -", merged_path)


if __name__ == "__main__":
    main()
