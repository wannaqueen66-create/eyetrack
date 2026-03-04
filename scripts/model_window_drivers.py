#!/usr/bin/env python3
"""Model whether Window AOI outcomes are mainly driven by WWR or Complexity.

Goal
----
Given an AOI long table (typically batch_aoi_metrics_by_class.csv or merged analysis table),
fit comparable models with both predictors and report which predictor has the larger
standardized effect for each outcome.

Outcomes:
- visited (binary)
- TTFF_ms (conditional on visited==1)
- dwell_time_ms (conditional on visited==1)
- fixation_count (conditional on visited==1; modeled on log1p scale)

Predictors:
- WWR
- Complexity

Input columns expected (minimum):
- participant_id, scene_id, class_name, visited, TTFF_ms, dwell_time_ms, fixation_count

WWR/Complexity source priority:
1) Existing numeric columns (--wwr_col / --complexity_col)
2) Parse from scene_id patterns like: WWR45_C1, C1W45, 组1-C1W45, group2-C0W15
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def _parse_scene(scene_id: str) -> Tuple[Optional[float], Optional[float]]:
    s = str(scene_id)

    # WWR45_C1
    m = re.search(r"WWR(?P<wwr>\d+)_C(?P<c>\d+)", s, flags=re.IGNORECASE)
    if m:
        return float(m.group("wwr")), float(m.group("c"))

    # C1W45
    m = re.search(r"C(?P<c>\d+)W(?P<wwr>\d+)", s, flags=re.IGNORECASE)
    if m:
        return float(m.group("wwr")), float(m.group("c"))

    # fallback: W45 + C1 appearing separately
    mw = re.search(r"(?:WWR|W)(?P<wwr>\d+)", s, flags=re.IGNORECASE)
    mc = re.search(r"(?:COMP(?:LEXITY)?|C)(?P<c>\d+)", s, flags=re.IGNORECASE)
    wwr = float(mw.group("wwr")) if mw else None
    comp = float(mc.group("c")) if mc else None
    return wwr, comp


def _z(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    m = x.mean()
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=x.index)
    return (x - m) / sd


def _safe_fit_mixedlm(df: pd.DataFrame, y_col: str):
    # Standardize y so coefficients are comparable standardized effects.
    d = df.copy()
    d["y_z"] = _z(d[y_col])
    d = d.dropna(subset=["y_z", "WWR_z", "Complexity_z", "participant_id"])
    if len(d) < 12:
        raise ValueError(f"Too few rows for mixedlm ({y_col}): n={len(d)}")
    model = smf.mixedlm("y_z ~ WWR_z + Complexity_z", data=d, groups=d["participant_id"])
    res = model.fit(reml=False, method="lbfgs")
    return res, d


def _safe_fit_logit(df: pd.DataFrame):
    # GLM with participant fixed effects for binary visited outcome.
    d = df.copy()
    d["visited"] = pd.to_numeric(d["visited"], errors="coerce")
    d = d.dropna(subset=["visited", "WWR_z", "Complexity_z", "participant_id"])
    d["visited"] = (d["visited"] > 0).astype(int)
    if d["visited"].nunique() < 2:
        raise ValueError("visited has <2 classes after filtering")
    if len(d) < 20:
        raise ValueError(f"Too few rows for visited model: n={len(d)}")
    model = smf.glm("visited ~ WWR_z + Complexity_z + C(participant_id)", data=d, family=__import__("statsmodels.api").api.families.Binomial())
    res = model.fit()
    return res, d


def _coef_block(res, names=("WWR_z", "Complexity_z")):
    out = {}
    for nm in names:
        out[nm] = {
            "coef": float(res.params.get(nm, np.nan)),
            "p": float(res.pvalues.get(nm, np.nan)) if hasattr(res, "pvalues") else np.nan,
        }
    return out


def _dominance(wwr_coef: float, comp_coef: float, wwr_p: float, comp_p: float, p_alpha: float = 0.05, ratio_thr: float = 1.2) -> str:
    aw = abs(wwr_coef) if np.isfinite(wwr_coef) else np.nan
    ac = abs(comp_coef) if np.isfinite(comp_coef) else np.nan
    if not np.isfinite(aw) or not np.isfinite(ac):
        return "undetermined"

    w_sig = np.isfinite(wwr_p) and (wwr_p < p_alpha)
    c_sig = np.isfinite(comp_p) and (comp_p < p_alpha)

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
    ap.add_argument("--analysis_csv", required=True, help="Input long table CSV")
    ap.add_argument("--outdir", default="outputs_window_drivers")
    ap.add_argument("--class_name", default="window", help="AOI class to analyze (default: window)")
    ap.add_argument("--wwr_col", default="WWR", help="Existing WWR column name (if present)")
    ap.add_argument("--complexity_col", default="Complexity", help="Existing Complexity column name (if present)")
    ap.add_argument("--p_alpha", type=float, default=0.05)
    ap.add_argument("--dominance_ratio", type=float, default=1.2, help="Dominance threshold on |standardized coef| ratio")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.analysis_csv)

    # Basic required columns
    req = ["participant_id", "scene_id", "class_name", "visited", "TTFF_ms", "dwell_time_ms", "fixation_count"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"analysis_csv missing columns: {miss}")

    # Focus on target AOI class
    d = df[df["class_name"].astype(str).str.lower() == str(args.class_name).lower()].copy()
    if len(d) == 0:
        raise ValueError(f"No rows found for class_name={args.class_name}")

    # Build WWR / Complexity
    if args.wwr_col in d.columns:
        d["WWR"] = pd.to_numeric(d[args.wwr_col], errors="coerce")
    else:
        d["WWR"] = np.nan

    if args.complexity_col in d.columns:
        d["Complexity"] = pd.to_numeric(d[args.complexity_col], errors="coerce")
    else:
        d["Complexity"] = np.nan

    parsed = d["scene_id"].apply(_parse_scene)
    d[["WWR_parsed", "Complexity_parsed"]] = pd.DataFrame(parsed.tolist(), index=d.index)

    d["WWR"] = d["WWR"].where(d["WWR"].notna(), d["WWR_parsed"])
    d["Complexity"] = d["Complexity"].where(d["Complexity"].notna(), d["Complexity_parsed"])

    d["WWR_z"] = _z(d["WWR"])
    d["Complexity_z"] = _z(d["Complexity"])

    # Save parsed audit table
    audit_cols = [
        "participant_id", "scene_id", "class_name",
        "WWR", "Complexity", "WWR_parsed", "Complexity_parsed",
        "visited", "TTFF_ms", "dwell_time_ms", "fixation_count",
    ]
    d[audit_cols].to_csv(os.path.join(args.outdir, "window_driver_input_audit.csv"), index=False)

    rows = []
    report = []

    # Model A: visited (binary)
    try:
        res, dm = _safe_fit_logit(d)
        b = _coef_block(res)
        dom = _dominance(
            b["WWR_z"]["coef"], b["Complexity_z"]["coef"],
            b["WWR_z"]["p"], b["Complexity_z"]["p"],
            p_alpha=args.p_alpha,
            ratio_thr=args.dominance_ratio,
        )
        rows.append({
            "outcome": "visited",
            "n": len(dm),
            "wwr_coef": b["WWR_z"]["coef"],
            "wwr_p": b["WWR_z"]["p"],
            "complexity_coef": b["Complexity_z"]["coef"],
            "complexity_p": b["Complexity_z"]["p"],
            "dominant": dom,
        })
        with open(os.path.join(args.outdir, "model_visited_glm.txt"), "w", encoding="utf-8") as f:
            f.write(str(res.summary()))
        report.append(f"visited: dominant={dom}, beta_WWR={b['WWR_z']['coef']:.4g} (p={b['WWR_z']['p']:.3g}), beta_Complexity={b['Complexity_z']['coef']:.4g} (p={b['Complexity_z']['p']:.3g})")
    except Exception as e:
        rows.append({"outcome": "visited", "n": 0, "wwr_coef": np.nan, "wwr_p": np.nan, "complexity_coef": np.nan, "complexity_p": np.nan, "dominant": f"FAILED: {repr(e)}"})
        report.append(f"visited: FAILED {repr(e)}")

    # Conditional subset for continuous/count outcomes
    dc = d[pd.to_numeric(d["visited"], errors="coerce") > 0].copy()

    for outcome, y_col, transform in [
        ("TTFF_ms_given_visited", "TTFF_ms", "none"),
        ("dwell_time_ms_given_visited", "dwell_time_ms", "none"),
        ("fixation_count_given_visited", "fixation_count", "log1p"),
    ]:
        try:
            dm = dc.copy()
            dm[y_col] = pd.to_numeric(dm[y_col], errors="coerce")
            if transform == "log1p":
                dm[y_col] = np.log1p(dm[y_col].clip(lower=0))

            res, fit_df = _safe_fit_mixedlm(dm, y_col=y_col)
            b = _coef_block(res)
            dom = _dominance(
                b["WWR_z"]["coef"], b["Complexity_z"]["coef"],
                b["WWR_z"]["p"], b["Complexity_z"]["p"],
                p_alpha=args.p_alpha,
                ratio_thr=args.dominance_ratio,
            )
            rows.append({
                "outcome": outcome,
                "n": len(fit_df),
                "wwr_coef": b["WWR_z"]["coef"],
                "wwr_p": b["WWR_z"]["p"],
                "complexity_coef": b["Complexity_z"]["coef"],
                "complexity_p": b["Complexity_z"]["p"],
                "dominant": dom,
            })
            out_model = os.path.join(args.outdir, f"model_{outcome}.txt")
            with open(out_model, "w", encoding="utf-8") as f:
                f.write(str(res.summary()))
            report.append(f"{outcome}: dominant={dom}, beta_WWR={b['WWR_z']['coef']:.4g} (p={b['WWR_z']['p']:.3g}), beta_Complexity={b['Complexity_z']['coef']:.4g} (p={b['Complexity_z']['p']:.3g})")
        except Exception as e:
            rows.append({"outcome": outcome, "n": 0, "wwr_coef": np.nan, "wwr_p": np.nan, "complexity_coef": np.nan, "complexity_p": np.nan, "dominant": f"FAILED: {repr(e)}"})
            report.append(f"{outcome}: FAILED {repr(e)}")

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(args.outdir, "window_driver_summary.csv")
    summary.to_csv(summary_path, index=False)

    report_path = os.path.join(args.outdir, "window_driver_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Window AOI driver comparison (WWR vs Complexity)\n")
        f.write(f"input={args.analysis_csv}\n")
        f.write(f"class_name={args.class_name}\n")
        f.write(f"p_alpha={args.p_alpha}, dominance_ratio={args.dominance_ratio}\n\n")
        for ln in report:
            f.write("- " + ln + "\n")

    print("Saved:")
    print(" -", summary_path)
    print(" -", report_path)


if __name__ == "__main__":
    main()
