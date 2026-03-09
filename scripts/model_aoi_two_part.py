#!/usr/bin/env python3
"""Two-part modeling helpers for AOI outcomes."""

import argparse
import os
import warnings

import numpy as np
import pandas as pd


def _pick_predictors(df: pd.DataFrame):
    cand = ["table_density", "distance_to_table_center_m", "illum_lux", "crowding_level", "occlusion_ratio"]
    return [c for c in cand if c in df.columns]


def _safe_write(path: str, text: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser(description="Two-part models for AOI metrics")
    ap.add_argument("--analysis_csv", required=True, help="Merged analysis table (must include participant_id, visited, TTFF, TFD, FC)")
    ap.add_argument("--outdir", default="outputs_models_two_part")
    ap.add_argument("--predictors", default=None, help="Comma-separated predictor columns. If omitted, use built-in heuristic.")
    ap.add_argument("--log1p_ttff", action="store_true", help="Fit TTFF on log1p(TTFF)")
    ap.add_argument("--log1p_tfd", action="store_true", help="Fit TFD on log1p(TFD)")
    ap.add_argument("--count_family", default="nb", choices=["poisson", "nb"], help="Count model family for FC (GEE)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.analysis_csv)

    if "TFD" not in df.columns and "dwell_time_ms" in df.columns:
        df["TFD"] = pd.to_numeric(df["dwell_time_ms"], errors="coerce")
    if "FC" not in df.columns and "fixation_count" in df.columns:
        df["FC"] = pd.to_numeric(df["fixation_count"], errors="coerce")

    base_req = ["participant_id", "visited", "TTFF", "TFD", "FC"]
    miss = [c for c in base_req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns in analysis_csv: {miss}")

    predictors = [x.strip() for x in args.predictors.split(",") if x.strip()] if args.predictors else _pick_predictors(df)
    if not predictors:
        raise ValueError("No predictors found. Pass --predictors colA,colB,... or include known scene feature columns.")

    for c in predictors + ["visited", "TTFF", "TFD", "FC"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # visited
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

        d1 = df.dropna(subset=["visited", "participant_id"] + predictors).copy()
        d1["visited"] = (d1["visited"] > 0).astype(int)
        if d1["visited"].nunique() < 2:
            raise ValueError("visited has <2 classes after filtering; cannot fit")
        formula = "visited ~ " + " + ".join(predictors)
        vc = {"participant": "0 + C(participant_id)"}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = BinomialBayesMixedGLM.from_formula(formula, vc_formulas=vc, data=d1).fit_vb()
        _safe_write(os.path.join(args.outdir, "model_visited_logit_glmm.txt"), str(res.summary()))
    except Exception as e:
        _safe_write(os.path.join(args.outdir, "model_visited_logit_glmm.txt"), f"FAILED to fit visited GLMM: {repr(e)}\n")

    # TTFF
    try:
        import statsmodels.formula.api as smf

        d2 = df[df["visited"] == 1].copy().dropna(subset=["TTFF", "participant_id"] + predictors)
        if len(d2) < 8:
            raise ValueError(f"Too few rows for TTFF model: n={len(d2)}")
        lhs = "TTFF"
        if args.log1p_ttff:
            d2["ttff_y"] = np.log1p(d2["TTFF"].clip(lower=0))
            lhs = "ttff_y"
        res = smf.mixedlm(f"{lhs} ~ " + " + ".join(predictors), data=d2, groups=d2["participant_id"]).fit(reml=False, method="lbfgs")
        _safe_write(os.path.join(args.outdir, "model_TTFF_mixedlm.txt"), str(res.summary()))
    except Exception as e:
        _safe_write(os.path.join(args.outdir, "model_TTFF_mixedlm.txt"), f"FAILED to fit TTFF model: {repr(e)}\n")

    # TFD
    try:
        import statsmodels.formula.api as smf

        d3 = df[df["visited"] == 1].copy().dropna(subset=["TFD", "participant_id"] + predictors)
        if len(d3) < 8:
            raise ValueError(f"Too few rows for TFD model: n={len(d3)}")
        lhs = "TFD"
        if args.log1p_tfd:
            d3["tfd_y"] = np.log1p(d3["TFD"].clip(lower=0))
            lhs = "tfd_y"
        res = smf.mixedlm(f"{lhs} ~ " + " + ".join(predictors), data=d3, groups=d3["participant_id"]).fit(reml=False, method="lbfgs")
        _safe_write(os.path.join(args.outdir, "model_TFD_mixedlm.txt"), str(res.summary()))
    except Exception as e:
        _safe_write(os.path.join(args.outdir, "model_TFD_mixedlm.txt"), f"FAILED to fit TFD model: {repr(e)}\n")

    # FC
    try:
        import patsy
        from statsmodels.genmod.cov_struct import Exchangeable
        from statsmodels.genmod.families import NegativeBinomial, Poisson
        from statsmodels.genmod.generalized_estimating_equations import GEE

        d4 = df[df["visited"] == 1].copy().dropna(subset=["FC", "participant_id"] + predictors)
        if len(d4) < 8:
            raise ValueError(f"Too few rows for FC model: n={len(d4)}")
        fam = Poisson() if args.count_family == "poisson" else NegativeBinomial()
        y, X = patsy.dmatrices("FC ~ " + " + ".join(predictors), d4, return_type="dataframe")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = GEE(y, X, groups=d4["participant_id"], cov_struct=Exchangeable(), family=fam).fit()
        _safe_write(os.path.join(args.outdir, f"model_FC_gee_{args.count_family}.txt"), str(res.summary()))
    except Exception as e:
        _safe_write(os.path.join(args.outdir, f"model_FC_gee_{args.count_family}.txt"), f"FAILED to fit FC model: {repr(e)}\n")

    _safe_write(os.path.join(args.outdir, "RUNINFO.txt"), "Two-part AOI models\n" + f"predictors: {predictors}\n")
    print("Saved models to", args.outdir)


if __name__ == "__main__":
    main()
