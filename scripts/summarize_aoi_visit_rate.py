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


def _auto_parse_scene_id(scene_id: str) -> Tuple[str, str]:
    """Best-effort parser for common scene_id conventions.

    Returns (condition, round).

    Supported examples:
    - WWR45_C1            -> condition=C1W45, round=""
    - C1W45               -> condition=C1W45, round=""
    - 组1-C1W45           -> condition=C1W45, round=group1
    - group2-C0W15        -> condition=C0W15, round=group2
    - (.. ) 组1-C1W45     -> condition=C1W45, round=group1
    """
    s = str(scene_id)

    # 1) WWR45_C1
    m = re.search(r"WWR(?P<wwr>\d+)_C(?P<c>\d+)", s)
    if m:
        return (f"C{m.group('c')}W{m.group('wwr')}", "")

    # 2) C1W45
    m = re.search(r"C(?P<c>\d+)W(?P<wwr>\d+)", s)
    if m:
        # optional group/round prefix
        m2 = re.search(r"(?:组|group)(?P<r>\d+)", s)
        rnd = f"group{m2.group('r')}" if m2 else ""
        return (f"C{m.group('c')}W{m.group('wwr')}", rnd)

    # fallback
    return (s, "")


def parse_scene(scene_id: str, rx: Optional[re.Pattern]) -> Tuple[str, str]:
    if rx is None:
        return _auto_parse_scene_id(scene_id)
    m = rx.search(str(scene_id))
    if not m:
        return _auto_parse_scene_id(scene_id)
    cond = m.groupdict().get("condition") or m.group(1)
    rnd = m.groupdict().get("round") if m.groupdict() else ""
    if rnd is None and m.lastindex and m.lastindex >= 2:
        rnd = m.group(2)
    return (str(cond), str(rnd or ""))


def main():
    ap = argparse.ArgumentParser(description="Summarize p_not_visited and TTFF (given visited)")
    ap.add_argument("--aoi_class_csv", required=True, help="AOI metrics by class CSV (e.g., batch_aoi_metrics_by_class.csv)")
    ap.add_argument("--scene_map_csv", default=None, help="Optional CSV mapping scene_id -> condition[,round]")
    ap.add_argument("--long_csv", default=None, help="Optional long-format master table. If provided, will try to map SceneID->Condition/WWR/Complexity.")
    ap.add_argument("--scene_id_col", default="SceneID", help="Scene id column name in --long_csv (default: SceneID)")
    ap.add_argument("--condition_col", default="Condition", help="Condition column name in --long_csv (default: Condition)")
    ap.add_argument("--wwr_col", default="WWR", help="WWR column name in --long_csv (default: WWR)")
    ap.add_argument("--round_col", default=None, help="Optional round column name in --long_csv (e.g., Block/Repetition). Only usable if your AOI scene_id distinguishes rounds.")
    ap.add_argument("--scene_regex", default=None, help="Optional regex to parse condition/round from scene_id; use named groups (?P<condition>...) (?P<round>...)")
    ap.add_argument("--by_round", action="store_true", help="Group by (condition, round, class_name) if round is available")
    ap.add_argument("--out_csv", default="outputs/aoi_visit_summary.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.aoi_class_csv)

    required = {"participant_id", "scene_id", "class_name", "visited", "TTFF_ms"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Input missing columns: {sorted(miss)}")

    # Attach condition/round
    out = df.copy()

    # 1) scene_map_csv (highest priority)
    if args.scene_map_csv:
        m = pd.read_csv(args.scene_map_csv)
        if "scene_id" not in m.columns or "condition" not in m.columns:
            raise ValueError("scene_map_csv must contain columns: scene_id, condition[, round]")
        out = out.merge(m, on="scene_id", how="left")

    # 2) long_csv mapping (next priority)
    if (not args.scene_map_csv) and args.long_csv:
        long = pd.read_csv(args.long_csv)
        sid = args.scene_id_col
        if sid not in long.columns:
            raise ValueError(f"--long_csv missing scene id column: {sid}")

        cols = [sid]
        for c in [args.condition_col, args.wwr_col, args.round_col]:
            if c and c in long.columns:
                cols.append(c)

        m = long[cols].drop_duplicates().copy()
        m = m.rename(columns={sid: "scene_id"})
        # build a nice condition label if possible
        if args.condition_col in m.columns and args.wwr_col in m.columns:
            m["condition"] = m[args.condition_col].astype(str).str.strip() + "W" + m[args.wwr_col].astype(str).str.strip()
        elif args.condition_col in m.columns:
            m["condition"] = m[args.condition_col].astype(str).str.strip()

        if args.round_col and args.round_col in m.columns:
            m["round"] = m[args.round_col].astype(str).str.strip()

        out = out.merge(m[[c for c in ["scene_id", "condition", "round"] if c in m.columns]], on="scene_id", how="left")

    # 3) regex / auto-parse fallback
    if "condition" not in out.columns or out["condition"].isna().all():
        rx = re.compile(args.scene_regex) if args.scene_regex else None
        cond_round = out["scene_id"].apply(lambda s: parse_scene(s, rx))
        out[["condition", "round"]] = pd.DataFrame(cond_round.tolist(), index=out.index)

    # ensure round column exists
    if "round" not in out.columns:
        out["round"] = ""

    # numeric coercions
    out["visited"] = pd.to_numeric(out["visited"], errors="coerce").fillna(0).astype(int)
    out["TTFF_ms"] = pd.to_numeric(out["TTFF_ms"], errors="coerce")

    grp_cols = ["condition", "class_name"]
    if args.by_round:
        grp_cols = ["condition", "round", "class_name"]

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
