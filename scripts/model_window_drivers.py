#!/usr/bin/env python3
"""Model whether Window AOI outcomes are mainly driven by WWR or Complexity."""

from __future__ import annotations

import argparse
import os
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def _export_plots(summary: pd.DataFrame, outdir: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    s = summary.copy()
    if len(s) == 0:
        return
    for c in ["wwr_coef", "complexity_coef", "wwr_p", "complexity_p"]:
        if c in s.columns:
            s[c] = pd.to_numeric(s[c], errors="coerce")
    try:
        x = np.arange(len(s)); w = 0.36
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.bar(x - w / 2, s["wwr_coef"], width=w, label="WWR", color="#4e79a7")
        ax.bar(x + w / 2, s["complexity_coef"], width=w, label="Complexity", color="#f28e2b")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(s["outcome"], rotation=20, ha="right")
        ax.set_ylabel("Standardized coefficient"); ax.set_title("Window AOI: WWR vs Complexity effect sizes"); ax.legend(loc="best")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "window_driver_effect_sizes.png"), dpi=180)
    except Exception:
        pass
    try:
        vc = s["dominant"].fillna("undetermined").value_counts().reindex(["WWR", "Complexity", "both_or_close", "none_significant", "undetermined"]).fillna(0)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        bars = ax.bar(vc.index, vc.values, color=["#59a14f", "#e15759", "#9c755f", "#bab0ab", "#4e79a7"])
        ax.set_ylabel("Outcome count"); ax.set_title("Dominant driver summary"); ax.bar_label(bars, padding=2, fmt="%.0f")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "window_driver_dominance.png"), dpi=180)
    except Exception:
        pass


def _parse_scene(scene_id: str) -> Tuple[Optional[float], Optional[float]]:
    s = str(scene_id)
    m = re.search(r"WWR(?P<wwr>\d+)_C(?P<c>\d+)", s, flags=re.IGNORECASE)
    if m:
        return float(m.group("wwr")), float(m.group("c"))
    m = re.search(r"C(?P<c>\d+)W(?P<wwr>\d+)", s, flags=re.IGNORECASE)
    if m:
        return float(m.group("wwr")), float(m.group("c"))
    mw = re.search(r"(?:WWR|W)(?P<wwr>\d+)", s, flags=re.IGNORECASE)
    mc = re.search(r"(?:COMP(?:LEXITY)?|C)(?P<c>\d+)", s, flags=re.IGNORECASE)
    return (float(mw.group("wwr")) if mw else None, float(mc.group("c")) if mc else None)


def _z(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    m = x.mean(); sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=x.index)
    return (x - m) / sd


def _safe_fit_mixedlm(df: pd.DataFrame, y_col: str):
    d = df.copy(); d["y_z"] = _z(d[y_col]); d = d.dropna(subset=["y_z", "WWR_z", "Complexity_z", "participant_id"])
    if len(d) < 12:
        raise ValueError(f"Too few rows for mixedlm ({y_col}): n={len(d)}")
    return smf.mixedlm("y_z ~ WWR_z + Complexity_z", data=d, groups=d["participant_id"]).fit(reml=False, method="lbfgs"), d


def _safe_fit_logit(df: pd.DataFrame):
    d = df.copy(); d["visited"] = pd.to_numeric(d["visited"], errors="coerce"); d = d.dropna(subset=["visited", "WWR_z", "Complexity_z", "participant_id"]); d["visited"] = (d["visited"] > 0).astype(int)
    if d["visited"].nunique() < 2:
        raise ValueError("visited has <2 classes after filtering")
    if len(d) < 20:
        raise ValueError(f"Too few rows for visited model: n={len(d)}")
    model = smf.glm("visited ~ WWR_z + Complexity_z + C(participant_id)", data=d, family=__import__("statsmodels.api").api.families.Binomial())
    return model.fit(), d


def _coef_block(res, names=("WWR_z", "Complexity_z")):
    return {nm: {"coef": float(res.params.get(nm, np.nan)), "p": float(res.pvalues.get(nm, np.nan)) if hasattr(res, "pvalues") else np.nan} for nm in names}


def _dominance(wwr_coef: float, comp_coef: float, wwr_p: float, comp_p: float, p_alpha: float = 0.05, ratio_thr: float = 1.2) -> str:
    aw = abs(wwr_coef) if np.isfinite(wwr_coef) else np.nan; ac = abs(comp_coef) if np.isfinite(comp_coef) else np.nan
    if not np.isfinite(aw) or not np.isfinite(ac):
        return "undetermined"
    w_sig = np.isfinite(wwr_p) and (wwr_p < p_alpha); c_sig = np.isfinite(comp_p) and (comp_p < p_alpha)
    if (not w_sig) and (not c_sig):
        return "none_significant"
    if aw >= ratio_thr * max(ac, 1e-12) and w_sig:
        return "WWR"
    if ac >= ratio_thr * max(aw, 1e-12) and c_sig:
        return "Complexity"
    if w_sig and (not c_sig):
        return "WWR"
    if c_sig and (not w_sig):
        return "Complexity"
    return "both_or_close"


def main():
    ap = argparse.ArgumentParser(description="Compare WWR vs Complexity as AOI main drivers")
    ap.add_argument("--analysis_csv", required=True)
    ap.add_argument("--outdir", default="outputs_window_drivers")
    ap.add_argument("--class_name", default="window")
    ap.add_argument("--wwr_col", default="WWR")
    ap.add_argument("--complexity_col", default="Complexity")
    ap.add_argument("--p_alpha", type=float, default=0.05)
    ap.add_argument("--dominance_ratio", type=float, default=1.2)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.analysis_csv)
    if "TFD" not in df.columns and "dwell_time_ms" in df.columns:
        df["TFD"] = pd.to_numeric(df["dwell_time_ms"], errors="coerce")
    if "FC" not in df.columns and "fixation_count" in df.columns:
        df["FC"] = pd.to_numeric(df["fixation_count"], errors="coerce")
    req = ["participant_id", "scene_id", "class_name", "visited", "TTFF", "TFD", "FC"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"analysis_csv missing columns: {miss}")
    d = df[df["class_name"].astype(str).str.lower() == str(args.class_name).lower()].copy()
    if len(d) == 0:
        raise ValueError(f"No rows found for class_name={args.class_name}")
    d["WWR"] = pd.to_numeric(d[args.wwr_col], errors="coerce") if args.wwr_col in d.columns else np.nan
    d["Complexity"] = pd.to_numeric(d[args.complexity_col], errors="coerce") if args.complexity_col in d.columns else np.nan
    parsed = d["scene_id"].apply(_parse_scene)
    d[["WWR_parsed", "Complexity_parsed"]] = pd.DataFrame(parsed.tolist(), index=d.index)
    d["WWR"] = d["WWR"].where(d["WWR"].notna(), d["WWR_parsed"])
    d["Complexity"] = d["Complexity"].where(d["Complexity"].notna(), d["Complexity_parsed"])
    d["WWR_z"] = _z(d["WWR"]); d["Complexity_z"] = _z(d["Complexity"])
    d[["participant_id", "scene_id", "class_name", "WWR", "Complexity", "WWR_parsed", "Complexity_parsed", "visited", "TTFF", "TFD", "FC"]].to_csv(os.path.join(args.outdir, "window_driver_input_audit.csv"), index=False)

    rows = []; report = []
    try:
        res, dm = _safe_fit_logit(d); b = _coef_block(res); dom = _dominance(b["WWR_z"]["coef"], b["Complexity_z"]["coef"], b["WWR_z"]["p"], b["Complexity_z"]["p"], p_alpha=args.p_alpha, ratio_thr=args.dominance_ratio)
        rows.append({"outcome": "visited", "n": len(dm), "wwr_coef": b["WWR_z"]["coef"], "wwr_p": b["WWR_z"]["p"], "complexity_coef": b["Complexity_z"]["coef"], "complexity_p": b["Complexity_z"]["p"], "dominant": dom})
        open(os.path.join(args.outdir, "model_visited_glm.txt"), "w", encoding="utf-8").write(str(res.summary())); report.append(f"visited: dominant={dom}")
    except Exception as e:
        rows.append({"outcome": "visited", "n": 0, "wwr_coef": np.nan, "wwr_p": np.nan, "complexity_coef": np.nan, "complexity_p": np.nan, "dominant": f"FAILED: {repr(e)}"}); report.append(f"visited: FAILED {repr(e)}")

    dc = d[pd.to_numeric(d["visited"], errors="coerce") > 0].copy()
    for outcome, y_col, transform in [("TTFF_given_visited", "TTFF", "none"), ("TFD_given_visited", "TFD", "none"), ("FC_given_visited", "FC", "log1p")]:
        try:
            dm = dc.copy(); dm[y_col] = pd.to_numeric(dm[y_col], errors="coerce");
            if transform == "log1p": dm[y_col] = np.log1p(dm[y_col].clip(lower=0))
            res, fit_df = _safe_fit_mixedlm(dm, y_col=y_col); b = _coef_block(res); dom = _dominance(b["WWR_z"]["coef"], b["Complexity_z"]["coef"], b["WWR_z"]["p"], b["Complexity_z"]["p"], p_alpha=args.p_alpha, ratio_thr=args.dominance_ratio)
            rows.append({"outcome": outcome, "n": len(fit_df), "wwr_coef": b["WWR_z"]["coef"], "wwr_p": b["WWR_z"]["p"], "complexity_coef": b["Complexity_z"]["coef"], "complexity_p": b["Complexity_z"]["p"], "dominant": dom})
            open(os.path.join(args.outdir, f"model_{outcome}.txt"), "w", encoding="utf-8").write(str(res.summary())); report.append(f"{outcome}: dominant={dom}")
        except Exception as e:
            rows.append({"outcome": outcome, "n": 0, "wwr_coef": np.nan, "wwr_p": np.nan, "complexity_coef": np.nan, "complexity_p": np.nan, "dominant": f"FAILED: {repr(e)}"}); report.append(f"{outcome}: FAILED {repr(e)}")
    summary = pd.DataFrame(rows); summary.to_csv(os.path.join(args.outdir, "window_driver_summary.csv"), index=False); _export_plots(summary, args.outdir)
    with open(os.path.join(args.outdir, "window_driver_report.txt"), "w", encoding="utf-8") as f:
        f.write("Window AOI driver comparison (WWR vs Complexity)\n")
        for ln in report: f.write("- " + ln + "\n")
    print("Saved:"); print(" -", os.path.join(args.outdir, "window_driver_summary.csv")); print(" -", os.path.join(args.outdir, "window_driver_report.txt"))


if __name__ == "__main__":
    main()
