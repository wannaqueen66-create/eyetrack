#!/usr/bin/env python3
"""LMM-style modeling for AOI-based visual attention allocation.

This script is designed for the experiment where:
- "group" ONLY means between-subject population groups (e.g., Experience/SportFreq).
- Chinese "组1/组2" in scene folder names means Round/Block 1/2 (NOT group).

Inputs
------
- --aoi_class_csv: typically <...>/batch_aoi_metrics_by_class.csv produced by scripts/batch_aoi_metrics.py
  Required columns (minimum):
    participant_id, scene_id_raw (or scene_id), class_name
    dwell_time_ms (or TFD), TTFF_ms (or TTFF), fixation_count (or FC), visited
    round, WWR, Complexity   (added by our patched batch_aoi_metrics.py)
- --group_manifest: optional CSV with columns: name(or id) + SportFreq + Experience

Outputs
-------
For each group variable (Experience, SportFreq) and each outcome, export:
- model summary txt
- tidy fixed-effect table csv

Outcomes implemented
-------------------
1) tfd_y: log1p(TFD)
2) share_logit: logit( share_of_total_fixation_duration_within_trial )  (exploratory allocation proxy)
3) tff_y: log1p(TFF) on visited==1
4) fc_y: log1p(FC) on visited==1 (approximate; count-GLMM is better but out of scope)

Model form (statsmodels MixedLM)
-------------------------------
Fixed:
  C(class_name) * WWR_z * Complexity_z * C(GroupVar) + C(round)
Random:
  - random intercept by participant_id
  - variance component random intercept by scene_id_raw (if present)

Notes
-----
- This is a pragmatic MixedLM implementation in Python (statsmodels).
  For publication-grade GLMM (binary/count), consider R (lme4/glmmTMB).
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _norm_hilo(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    m = {
        "high": "High",
        "h": "High",
        "1": "High",
        "true": "High",
        "yes": "High",
        "low": "Low",
        "l": "Low",
        "0": "Low",
        "false": "Low",
        "no": "Low",
    }
    return m.get(s.lower(), s)


def _z(x: pd.Series) -> pd.Series:
    x = _safe_num(x)
    m = x.mean()
    sd = x.std(ddof=0)
    if (not np.isfinite(sd)) or sd <= 0:
        return pd.Series(np.nan, index=x.index)
    return (x - m) / sd


def _pick_col(df: pd.DataFrame, primary: str, fallback: str | None = None) -> str | None:
    if primary in df.columns:
        return primary
    if fallback and fallback in df.columns:
        return fallback
    return None


def _complexity_to_num(s: pd.Series) -> pd.Series:
    # Accept: C0/C1 strings, 0/1 numeric, Low/High, etc.
    x = s.copy()
    if x.dtype == object:
        u = x.astype(str).str.strip().str.upper()
        out = pd.Series(np.nan, index=x.index, dtype=float)
        out[u.isin(["C0", "0", "LOW", "L"]) ] = 0.0
        out[u.isin(["C1", "1", "HIGH", "H"]) ] = 1.0
        # try extract trailing digit
        m = u.str.extract(r"C(?P<c>[01])", expand=True)["c"]
        out = out.fillna(pd.to_numeric(m, errors="coerce"))
        return out
    return _safe_num(x)


def _tidy_fixef(res) -> pd.DataFrame:
    params = pd.Series(res.params)
    bse = pd.Series(res.bse)
    pvals = pd.Series(getattr(res, "pvalues", np.nan), index=params.index)

    try:
        ci = res.conf_int()
        if isinstance(ci, np.ndarray):
            ci = pd.DataFrame(ci, index=params.index, columns=["ci_low", "ci_high"])
        else:
            ci = ci.rename(columns={0: "ci_low", 1: "ci_high"})
    except Exception:
        ci = pd.DataFrame(index=params.index, columns=["ci_low", "ci_high"], data=np.nan)

    out = pd.DataFrame({
        "term": params.index,
        "coef": params.values,
        "se": bse.reindex(params.index).values,
        "p": pvals.reindex(params.index).values,
        "ci_low": ci.reindex(params.index)["ci_low"].values,
        "ci_high": ci.reindex(params.index)["ci_high"].values,
    })
    return out


def fit_mixedlm(df: pd.DataFrame, formula: str, group_col: str, vc_scene_col: str | None = None):
    # variance components: random intercept per scene_id_raw (optional)
    vc = None
    if vc_scene_col and vc_scene_col in df.columns:
        vc = {"scene": f"0 + C({vc_scene_col})"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if vc:
            model = smf.mixedlm(formula, data=df, groups=df[group_col], vc_formula=vc)
        else:
            model = smf.mixedlm(formula, data=df, groups=df[group_col])
        res = model.fit(reml=False, method="lbfgs")
    return res


def main():
    ap = argparse.ArgumentParser(description="AOI allocation LMM (Experience & SportFreq)")
    ap.add_argument("--aoi_class_csv", required=True)
    ap.add_argument("--group_manifest", default=None)
    ap.add_argument("--group_id_col", default="name", help="ID column in group_manifest (default: name)")
    ap.add_argument("--outdir", default="outputs_aoi_lmm")
    ap.add_argument("--min_rows", type=int, default=40)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.aoi_class_csv, encoding="utf-8-sig")

    # canonical id columns
    if "participant_id" not in df.columns:
        raise SystemExit("aoi_class_csv missing participant_id")

    # Attach group labels if provided
    if args.group_manifest and os.path.exists(args.group_manifest):
        gm = pd.read_csv(args.group_manifest, encoding="utf-8-sig")
        if args.group_id_col not in gm.columns:
            raise SystemExit(f"group_manifest missing id col: {args.group_id_col}")
        gm = gm.copy()
        gm["participant_id"] = gm[args.group_id_col].astype(str).str.strip()
        for c in ["SportFreq", "Experience"]:
            if c in gm.columns:
                gm[c] = gm[c].apply(_norm_hilo)
        keep = [c for c in ["participant_id", "SportFreq", "Experience"] if c in gm.columns]
        df["participant_id"] = df["participant_id"].astype(str).str.strip()
        df = df.merge(gm[keep], on="participant_id", how="left")

    # pick metric columns
    dwell_col = _pick_col(df, "TFD", "dwell_time_ms")
    ttff_col = _pick_col(df, "TFF", "TTFF_ms")
    fc_col = _pick_col(df, "FC", "fixation_count")

    need = ["participant_id", "class_name"]
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"aoi_class_csv missing required col: {c}")

    # scene id for variance components
    vc_scene_col = "scene_id_raw" if "scene_id_raw" in df.columns else ("scene_id" if "scene_id" in df.columns else None)

    # Exclude flagged trials if column exists
    if "trial_excluded" in df.columns:
        df["trial_excluded"] = _safe_num(df["trial_excluded"]).fillna(0).astype(int)
        df = df[df["trial_excluded"] == 0].copy()

    # Core predictors
    if "WWR" not in df.columns or "Complexity" not in df.columns:
        raise SystemExit("Missing WWR/Complexity columns. Re-run batch_aoi_metrics.py after latest patch.")

    df["WWR"] = _safe_num(df["WWR"])
    df["Complexity_num"] = _complexity_to_num(df["Complexity"])
    df["WWR_z"] = _z(df["WWR"])
    df["Complexity_z"] = _z(df["Complexity_num"])

    # Round (optional but recommended)
    if "round" in df.columns:
        df["round"] = _safe_num(df["round"]).astype("Int64")
    else:
        df["round"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    # visited
    if "visited" in df.columns:
        df["visited"] = _safe_num(df["visited"]).fillna(0).astype(int)
    else:
        df["visited"] = 1

    # Outcomes
    if dwell_col:
        df["TFD"] = _safe_num(df[dwell_col])
        df["tfd_y"] = np.log1p(df["TFD"].clip(lower=0))

        # share within (participant, scene)
        if vc_scene_col:
            key_cols = ["participant_id", vc_scene_col]
            tot = df.groupby(key_cols, dropna=False)["TFD"].sum(min_count=1).rename("TFD_total_ms")
            df = df.merge(tot.reset_index(), on=key_cols, how="left")
            eps = 1e-6
            df["share"] = df["TFD"] / df["TFD_total_ms"]
            df["share"] = df["share"].clip(lower=0, upper=1)
            df["share_logit"] = np.log((df["share"] + eps) / (1 - df["share"] + eps))

    if ttff_col:
        df["TFF"] = _safe_num(df[ttff_col])
        df["tff_y"] = np.log1p(df["TFF"].clip(lower=0))

    if fc_col:
        df["FC"] = _safe_num(df[fc_col])
        df["fc_y"] = np.log1p(df["FC"].clip(lower=0))

    # Run per group variable
    group_vars = []
    for gv in ["Experience", "SportFreq"]:
        if gv in df.columns:
            group_vars.append(gv)

    if not group_vars:
        raise SystemExit("No group variables found (Experience/SportFreq). Pass --group_manifest or ensure columns exist.")

    outcomes = [
        ("tfd_y", "All rows"),
        ("share_logit", "All rows (allocation proxy)"),
        ("tff_y", "visited==1"),
        ("fc_y", "visited==1"),
    ]

    for gv in group_vars:
        gdir = os.path.join(args.outdir, f"groupvar_{gv}")
        os.makedirs(gdir, exist_ok=True)

        for ycol, subset_note in outcomes:
            if ycol not in df.columns:
                continue

            d = df.copy()
            d = d.dropna(subset=[ycol, "WWR_z", "Complexity_z", "class_name", gv, "participant_id"])
            if "round" in d.columns:
                # allow missing round; treat as category when available
                pass

            if "visited" in d.columns and subset_note == "visited==1":
                d = d[d["visited"] == 1].copy()

            if len(d) < args.min_rows:
                # still write a note for auditability
                with open(os.path.join(gdir, f"model_{ycol}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"SKIP: too few rows for outcome={ycol} (n={len(d)}; min_rows={args.min_rows})\n")
                continue

            # Fixed effects
            # NOTE: use C() for categorical terms.
            # Round as categorical when present (Round1/2); if all missing, it will drop.
            formula = f"{ycol} ~ C(class_name) * WWR_z * Complexity_z * C({gv})"
            if "round" in d.columns and d["round"].notna().any():
                formula += " + C(round)"

            try:
                res = fit_mixedlm(d, formula, group_col="participant_id", vc_scene_col=vc_scene_col)
                summ_txt = str(res.summary())
                with open(os.path.join(gdir, f"model_{ycol}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Outcome: {ycol}\n")
                    f.write(f"Subset: {subset_note}\n")
                    f.write(f"Formula: {formula}\n")
                    f.write(f"n={len(d)}\n")
                    f.write(f"vc_scene_col={vc_scene_col}\n\n")
                    f.write(summ_txt)

                tidy = _tidy_fixef(res)
                tidy.insert(0, "group_var", gv)
                tidy.insert(0, "outcome", ycol)
                tidy.insert(0, "n", int(len(d)))
                tidy.to_csv(os.path.join(gdir, f"fixef_{ycol}.csv"), index=False, encoding="utf-8-sig")
            except Exception as e:
                with open(os.path.join(gdir, f"model_{ycol}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"FAILED outcome={ycol}\n")
                    f.write(f"Subset: {subset_note}\n")
                    f.write(f"Formula: {formula}\n")
                    f.write(f"n={len(d)}\n")
                    f.write(repr(e) + "\n")

    with open(os.path.join(args.outdir, "RUNINFO.txt"), "w", encoding="utf-8") as f:
        f.write("AOI LMM allocation run\n")
        f.write(f"aoi_class_csv: {args.aoi_class_csv}\n")
        f.write(f"group_manifest: {args.group_manifest}\n")
        f.write(f"group_id_col: {args.group_id_col}\n")
        f.write(f"min_rows: {args.min_rows}\n")
        f.write("group vars: " + ",".join(group_vars) + "\n")
        f.write(f"vc_scene_col: {vc_scene_col}\n")

    print("Saved:", args.outdir)


if __name__ == "__main__":
    main()
