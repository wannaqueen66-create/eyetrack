#!/usr/bin/env python3
"""Two-part modeling helpers for AOI outcomes.

Part 1: whether the AOI was visited (binary)
- visited ~ predictors + (1 | participant_id)   [approx GLMM]

Part 2: conditional outcomes given visited==1
- TTFF_ms (often skewed) -> optional log1p transform
- dwell_time_ms -> optional log1p transform
- fixation_count -> GEE Poisson/NB (clustered by participant)

This script is designed to provide reproducible, paper-friendly outputs.
It is intentionally conservative: if a model cannot be fit reliably (small N, singularities),
it will save a warning and continue.

Requires: pandas, numpy, statsmodels.
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd


def _pick_predictors(df: pd.DataFrame):
    # Heuristic: use common scene feature columns if present
    cand = [
        "table_density",
        "distance_to_table_center_m",
        "illum_lux",
        "crowding_level",
        "occlusion_ratio",
    ]
    return [c for c in cand if c in df.columns]


def _safe_write(path: str, text: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser(description="Two-part models for AOI metrics")
    ap.add_argument("--analysis_csv", required=True, help="Merged analysis table (must include participant_id, visited, TTFF_ms, dwell_time_ms, fixation_count)")
    ap.add_argument("--outdir", default="outputs_models_two_part")
    ap.add_argument("--predictors", default=None, help="Comma-separated predictor columns. If omitted, use built-in heuristic.")
    ap.add_argument("--log1p_ttff", action="store_true", help="Fit TTFF on log1p(TTFF_ms)")
    ap.add_argument("--log1p_dwell", action="store_true", help="Fit dwell on log1p(dwell_time_ms)")
    ap.add_argument("--count_family", default="nb", choices=["poisson", "nb"], help="Count model family for fixation_count (GEE)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.analysis_csv)

    # required columns
    base_req = ["participant_id", "visited", "TTFF_ms", "dwell_time_ms", "fixation_count"]
    miss = [c for c in base_req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns in analysis_csv: {miss}")

    # predictors
    if args.predictors:
        predictors = [x.strip() for x in args.predictors.split(",") if x.strip()]
    else:
        predictors = _pick_predictors(df)

    if not predictors:
        raise ValueError("No predictors found. Pass --predictors colA,colB,... or include known scene feature columns.")

    # Coerce numerics
    for c in predictors + ["visited", "TTFF_ms", "dwell_time_ms", "fixation_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---------------- Part 1: visited (binary) ----------------
    try:
        import patsy
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

        d1 = df.dropna(subset=["visited", "participant_id"] + predictors).copy()
        d1["visited"] = (d1["visited"] > 0).astype(int)
        if d1["visited"].nunique() < 2:
            raise ValueError("visited has <2 classes after filtering; cannot fit")

        formula = "visited ~ " + " + ".join(predictors)
        # Random intercept by participant_id via variance components
        vc = {"participant": "0 + C(participant_id)"}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = BinomialBayesMixedGLM.from_formula(formula, vc_formulas=vc, data=d1)
            res = model.fit_vb()

        _safe_write(os.path.join(args.outdir, "model_visited_logit_glmm.txt"), str(res.summary()))
    except Exception as e:
        _safe_write(os.path.join(args.outdir, "model_visited_logit_glmm.txt"), f"FAILED to fit visited GLMM: {repr(e)}\n")

    # ---------------- Part 2a: TTFF given visited ----------------
    try:
        import statsmodels.formula.api as smf

        d2 = df[(df["visited"] == 1)].copy()
        d2 = d2.dropna(subset=["TTFF_ms", "participant_id"] + predictors)
        if len(d2) < 8:
            raise ValueError(f"Too few rows for TTFF model: n={len(d2)}")

        if args.log1p_ttff:
            d2["TTFF_y"] = np.log1p(d2["TTFF_ms"].clip(lower=0))
            lhs = "TTFF_y"
        else:
            lhs = "TTFF_ms"

        formula = f"{lhs} ~ " + " + ".join(predictors)
        m = smf.mixedlm(formula, data=d2, groups=d2["participant_id"])
        r = m.fit(reml=False, method="lbfgs")
        _safe_write(os.path.join(args.outdir, "model_TTFF_mixedlm.txt"), str(r.summary()))
    except Exception as e:
        _safe_write(os.path.join(args.outdir, "model_TTFF_mixedlm.txt"), f"FAILED to fit TTFF model: {repr(e)}\n")

    # ---------------- Part 2b: dwell given visited ----------------
    try:
        import statsmodels.formula.api as smf

        d3 = df[(df["visited"] == 1)].copy()
        d3 = d3.dropna(subset=["dwell_time_ms", "participant_id"] + predictors)
        if len(d3) < 8:
            raise ValueError(f"Too few rows for dwell model: n={len(d3)}")

        if args.log1p_dwell:
            d3["dwell_y"] = np.log1p(d3["dwell_time_ms"].clip(lower=0))
            lhs = "dwell_y"
        else:
            lhs = "dwell_time_ms"

        formula = f"{lhs} ~ " + " + ".join(predictors)
        m = smf.mixedlm(formula, data=d3, groups=d3["participant_id"])
        r = m.fit(reml=False, method="lbfgs")
        _safe_write(os.path.join(args.outdir, "model_dwell_mixedlm.txt"), str(r.summary()))
    except Exception as e:
        _safe_write(os.path.join(args.outdir, "model_dwell_mixedlm.txt"), f"FAILED to fit dwell model: {repr(e)}\n")

    # ---------------- Part 2c: fixation_count (count) ----------------
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from statsmodels.genmod.families import Poisson, NegativeBinomial
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.cov_struct import Exchangeable

        d4 = df[(df["visited"] == 1)].copy()
        d4 = d4.dropna(subset=["fixation_count", "participant_id"] + predictors)
        if len(d4) < 8:
            raise ValueError(f"Too few rows for count model: n={len(d4)}")

        fam = Poisson() if args.count_family == "poisson" else NegativeBinomial()
        # GEE with exchangeable correlation by participant
        formula = "fixation_count ~ " + " + ".join(predictors)
        y, X = patsy.dmatrices(formula, d4, return_type="dataframe")
        gee = GEE(y, X, groups=d4["participant_id"], cov_struct=Exchangeable(), family=fam)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = gee.fit()
        _safe_write(os.path.join(args.outdir, f"model_fixation_count_gee_{args.count_family}.txt"), str(res.summary()))
    except Exception as e:
        _safe_write(os.path.join(args.outdir, f"model_fixation_count_gee_{args.count_family}.txt"), f"FAILED to fit fixation_count model: {repr(e)}\n")

    # small manifest of what was done
    _safe_write(
        os.path.join(args.outdir, "RUNINFO.txt"),
        "Two-part AOI models\n"
        f"predictors: {predictors}\n"
        f"log1p_ttff: {args.log1p_ttff}\n"
        f"log1p_dwell: {args.log1p_dwell}\n"
        f"count_family: {args.count_family}\n",
    )

    print("Saved models to", args.outdir)


if __name__ == "__main__":
    main()
