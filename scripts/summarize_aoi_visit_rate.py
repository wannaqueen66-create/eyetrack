#!/usr/bin/env python3
"""Summarize AOI visit probability (p_not_visited) and conditional TTFF."""

import argparse
import os
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _auto_parse_scene_id(scene_id: str) -> Tuple[str, str]:
    s = str(scene_id)
    m = re.search(r"WWR(?P<wwr>\d+)_C(?P<c>\d+)", s)
    if m:
        return (f"C{m.group('c')}W{m.group('wwr')}", "")
    m = re.search(r"C(?P<c>\d+)W(?P<wwr>\d+)", s)
    if m:
        m2 = re.search(r"(?:组|group)(?P<r>\d+)", s)
        rnd = f"group{m2.group('r')}" if m2 else ""
        return (f"C{m.group('c')}W{m.group('wwr')}", rnd)
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
    ap.add_argument("--aoi_class_csv", required=True)
    ap.add_argument("--scene_map_csv", default=None)
    ap.add_argument("--long_csv", default=None)
    ap.add_argument("--scene_id_col", default="SceneID")
    ap.add_argument("--condition_col", default="Condition")
    ap.add_argument("--wwr_col", default="WWR")
    ap.add_argument("--round_col", default=None)
    ap.add_argument("--scene_regex", default=None)
    ap.add_argument("--by_round", action="store_true")
    ap.add_argument("--out_csv", default="outputs/aoi_visit_summary.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.aoi_class_csv)
    required = {"participant_id", "scene_id", "class_name", "visited", "TTFF"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Input missing columns: {sorted(miss)}")
    out = df.copy()
    if args.scene_map_csv:
        m = pd.read_csv(args.scene_map_csv)
        if "scene_id" not in m.columns or "condition" not in m.columns:
            raise ValueError("scene_map_csv must contain columns: scene_id, condition[, round]")
        out = out.merge(m, on="scene_id", how="left")
    if (not args.scene_map_csv) and args.long_csv:
        long = pd.read_csv(args.long_csv)
        sid = args.scene_id_col
        if sid not in long.columns:
            raise ValueError(f"--long_csv missing scene id column: {sid}")
        cols = [sid] + [c for c in [args.condition_col, args.wwr_col, args.round_col] if c and c in long.columns]
        m = long[cols].drop_duplicates().copy().rename(columns={sid: "scene_id"})
        if args.condition_col in m.columns and args.wwr_col in m.columns:
            m["condition"] = m[args.condition_col].astype(str).str.strip() + "W" + m[args.wwr_col].astype(str).str.strip()
        elif args.condition_col in m.columns:
            m["condition"] = m[args.condition_col].astype(str).str.strip()
        if args.round_col and args.round_col in m.columns:
            m["round"] = m[args.round_col].astype(str).str.strip()
        out = out.merge(m[[c for c in ["scene_id", "condition", "round"] if c in m.columns]], on="scene_id", how="left")
    if "condition" not in out.columns or out["condition"].isna().all():
        rx = re.compile(args.scene_regex) if args.scene_regex else None
        out[["condition", "round"]] = pd.DataFrame(out["scene_id"].apply(lambda s: parse_scene(s, rx)).tolist(), index=out.index)
    if "round" not in out.columns:
        out["round"] = ""
    out["visited"] = pd.to_numeric(out["visited"], errors="coerce").fillna(0).astype(int)
    out["TTFF"] = pd.to_numeric(out["TTFF"], errors="coerce")
    grp_cols = ["condition", "round", "class_name"] if args.by_round else ["condition", "class_name"]

    def agg(g: pd.DataFrame) -> pd.Series:
        n = len(g); n_visited = int(g["visited"].sum()); p_not = float(1.0 - (n_visited / n)) if n else np.nan; ttff_vis = g.loc[g["visited"] == 1, "TTFF"].dropna()
        return pd.Series({"n_trials": int(n), "n_visited": int(n_visited), "p_not_visited": p_not, "TTFF_mean_ms_given_visited": float(ttff_vis.mean()) if len(ttff_vis) else np.nan, "TTFF_median_ms_given_visited": float(ttff_vis.median()) if len(ttff_vis) else np.nan})

    summ = out.groupby(grp_cols, dropna=False).apply(agg).reset_index()
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    summ.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)


if __name__ == "__main__":
    main()
