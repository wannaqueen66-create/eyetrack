#!/usr/bin/env python3
"""Summarize AOI visit probability (p_not_visited) and conditional TTFF.

This script is meant to complement TTFF handling:
- TTFF_ms is NaN when an AOI was never visited in a trial/scene.
- We therefore report both:
  1) p_not_visited = P(visited==0)
  2) TTFF stats conditioned on visited==1

Input:
- aoi_class_csv: typically outputs from batch_aoi_metrics.py
  columns must include: participant_id, scene_id, class_name, visited, TTFF_ms

Condition definition:
- You may provide a mapping CSV with columns: scene_id,condition[,round]
- Or, provide --scene_regex to parse condition/round from scene_id.

Outputs:
- out_csv: summary by condition and AOI class
"""

import argparse
import os
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def parse_scene(scene_id: str, rx: Optional[re.Pattern]) -> Tuple[str, str]:
    if rx is None:
        return (str(scene_id), "")
    m = rx.search(str(scene_id))
    if not m:
        return (str(scene_id), "")
    cond = m.groupdict().get("condition") or m.group(1)
    rnd = m.groupdict().get("round") if m.groupdict() else ""
    if rnd is None and m.lastindex and m.lastindex >= 2:
        rnd = m.group(2)
    return (str(cond), str(rnd or ""))


def main():
    ap = argparse.ArgumentParser(description="Summarize p_not_visited and TTFF (given visited)")
    ap.add_argument("--aoi_class_csv", required=True, help="AOI metrics by class CSV (e.g., batch_aoi_metrics_by_class.csv)")
    ap.add_argument("--scene_map_csv", default=None, help="Optional CSV mapping scene_id -> condition[,round]")
    ap.add_argument("--scene_regex", default=None, help="Optional regex to parse condition/round from scene_id; use named groups (?P<condition>...) (?P<round>...)")
    ap.add_argument("--out_csv", default="outputs/aoi_visit_summary.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.aoi_class_csv)

    required = {"participant_id", "scene_id", "class_name", "visited", "TTFF_ms"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Input missing columns: {sorted(miss)}")

    # Attach condition/round
    if args.scene_map_csv:
        m = pd.read_csv(args.scene_map_csv)
        if "scene_id" not in m.columns or "condition" not in m.columns:
            raise ValueError("scene_map_csv must contain columns: scene_id, condition[, round]")
        out = df.merge(m, on="scene_id", how="left")
        if out["condition"].isna().any():
            # keep going but warn via summary row
            pass
    else:
        rx = re.compile(args.scene_regex) if args.scene_regex else None
        cond_round = df["scene_id"].apply(lambda s: parse_scene(s, rx))
        df = df.copy()
        df[["condition", "round"]] = pd.DataFrame(cond_round.tolist(), index=df.index)
        out = df

    # numeric coercions
    out["visited"] = pd.to_numeric(out["visited"], errors="coerce").fillna(0).astype(int)
    out["TTFF_ms"] = pd.to_numeric(out["TTFF_ms"], errors="coerce")

    grp_cols = ["condition", "class_name"]

    def agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        n_visited = int(g["visited"].sum())
        p_not = float(1.0 - (n_visited / n)) if n else np.nan
        ttff_vis = g.loc[g["visited"] == 1, "TTFF_ms"].dropna()
        return pd.Series(
            {
                "n_trials": int(n),
                "n_visited": int(n_visited),
                "p_not_visited": p_not,
                "TTFF_mean_ms_given_visited": float(ttff_vis.mean()) if len(ttff_vis) else np.nan,
                "TTFF_median_ms_given_visited": float(ttff_vis.median()) if len(ttff_vis) else np.nan,
            }
        )

    summ = out.groupby(grp_cols, dropna=False).apply(agg).reset_index()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    summ.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)


if __name__ == "__main__":
    main()
