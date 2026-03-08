#!/usr/bin/env python3
"""Paper-oriented AOI modeling (main effects + interactions + PNG outputs).

What this script adds compared with baseline scripts:
- visited: logistic GLM (with participant fixed effect)
- TFF / TFD: MixedLM on visited==1 subset (optional log1p transform)
- FC: GEE Poisson/NB on visited==1 subset
- optional interaction terms
- tidy coefficient table CSV + BH-FDR q-values
- PNG coefficient forest plots per outcome + combined dominance chart

Input expected columns (minimum):
- participant_id, class_name, visited, TFF, TFD, FC
- predictor columns (numeric)
Legacy aliases (`TTFF_ms`, `dwell_time_ms`, `fixation_count`) are still accepted.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd


def _pick_predictors(df: pd.DataFrame) -> List[str]:
    cand = [
        "WWR",
        "Complexity",
        "table_density",
        "distance_to_table_center_m",
        "illum_lux",
        "crowding_level",
        "occlusion_ratio",
    ]
    return [c for c in cand if c in df.columns]


def _z(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    m = x.mean()
    sd = x.std(ddof=0)
    if (not np.isfinite(sd)) or sd <= 0:
        return pd.Series(np.nan, index=s.index)
    return (x - m) / sd


def _bh_fdr(pvals: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    ok = p.notna()
    if ok.sum() == 0:
        return out
    pv = p[ok].to_numpy(dtype=float)
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    q_back = np.empty_like(q)
    q_back[order] = q
    out.loc[ok] = q_back
    return out


def _coef_table(res, outcome: str, model_name: str, terms_keep: List[str], n: int) -> pd.DataFrame:
    params = pd.Series(res.params)
    bse = pd.Series(res.bse)
    pvalues = pd.Series(getattr(res, "pvalues", np.nan), index=params.index)
    conf = res.conf_int()
    if isinstance(conf, np.ndarray):
        conf = pd.DataFrame(conf, index=params.index, columns=["ci_low", "ci_high"])
    else:
        conf = conf.rename(columns={0: "ci_low", 1: "ci_high"})

    rows = []
    for term in params.index:
        if term == "Intercept":
            continue
        if term not in terms_keep:
            continue
        coef = float(params.get(term, np.nan))
        se = float(bse.get(term, np.nan))
        stat = coef / se if np.isfinite(coef) and np.isfinite(se) and se > 0 else np.nan
        rows.append(
            {
                "outcome": outcome,
                "model": model_name,
                "term": term,
                "coef": coef,
                "se": se,
                "stat": stat,
                "p": float(pvalues.get(term, np.nan)),
                "ci_low": float(conf.loc[term, "ci_low"]) if term in conf.index else np.nan,
                "ci_high": float(conf.loc[term, "ci_high"]) if term in conf.index else np.nan,
                "n": int(n),
            }
        )
    return pd.DataFrame(rows)


def _make_formula(terms: List[str], include_participant_fe: bool = False) -> str:
    rhs = " + ".join(terms) if terms else "1"
    if include_participant_fe:
        rhs += " + C(participant_id)"
    return rhs


def _plot_forest(df: pd.DataFrame, out_png: str, title: str):
    if df is None or len(df) == 0:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    d = df.copy().sort_values("coef")
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(8.2, max(3.4, 0.45 * len(d))))
    ax.errorbar(
        d["coef"],
        y,
        xerr=[d["coef"] - d["ci_low"], d["ci_high"] - d["coef"]],
        fmt="o",
        color="#2f6db3",
        ecolor="#8aaed6",
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(d["term"])
    ax.set_title(title)
    ax.set_xlabel("Coefficient (standardized predictors)")
    ax.grid(axis="x", alpha=0.25)

    # q-value labels
    for yi, (_, r) in enumerate(d.iterrows()):
        q = r.get("q")
        if pd.notna(q):
            ax.text(r["ci_high"], yi, f" q={q:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _plot_dominance(df: pd.DataFrame, out_png: str):
    if df is None or len(df) == 0:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    d = df.copy()
    d["abs_coef"] = d["coef"].abs()
    piv = d.pivot_table(index="outcome", columns="term", values="abs_coef", aggfunc="mean")
    if piv.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(piv.index))
    cols = list(piv.columns)
    width = 0.8 / max(1, len(cols))
    for i, c in enumerate(cols):
        ax.bar(x + (i - (len(cols) - 1) / 2) * width, piv[c].fillna(0).to_numpy(), width=width, label=c)

    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=15, ha="right")
    ax.set_ylabel("|coef|")
    ax.set_title("Predictor dominance by outcome (absolute standardized effects)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Paper-oriented AOI models with PNG outputs")
    ap.add_argument("--analysis_csv", required=True)
    ap.add_argument("--outdir", default="outputs_model_paper_plus")
    ap.add_argument("--class_name", default=None, help="Optional AOI class filter")
    ap.add_argument("--predictors", default=None, help="Comma-separated predictors")
    ap.add_argument("--interactions", default=None, help="Comma-separated interactions, e.g. WWR:Complexity,WWR:occlusion_ratio")
    ap.add_argument("--all_pairwise_interactions", action="store_true")
    ap.add_argument("--random_slope", default=None, help="One predictor to add random slope in MixedLM")
    ap.add_argument("--count_family", default="nb", choices=["nb", "poisson"])
    ap.add_argument("--log1p_tff", action="store_true")
    ap.add_argument("--log1p_tfd", action="store_true")
    ap.add_argument("--log1p_ttff", action="store_true", help="Legacy alias of --log1p_tff")
    ap.add_argument("--log1p_dwell", action="store_true", help="Legacy alias of --log1p_tfd")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.analysis_csv)

    if "TFF" not in df.columns and "TTFF_ms" in df.columns:
        df["TFF"] = pd.to_numeric(df["TTFF_ms"], errors="coerce")
    if "TFD" not in df.columns and "dwell_time_ms" in df.columns:
        df["TFD"] = pd.to_numeric(df["dwell_time_ms"], errors="coerce")
    if "FC" not in df.columns and "fixation_count" in df.columns:
        df["FC"] = pd.to_numeric(df["fixation_count"], errors="coerce")

    required = ["participant_id", "class_name", "visited", "TFF", "TFD", "FC"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    if args.class_name:
        df = df[df["class_name"].astype(str).str.lower() == str(args.class_name).lower()].copy()
        if len(df) == 0:
            raise ValueError(f"No rows for class_name={args.class_name}")

    if args.predictors:
        predictors = [x.strip() for x in args.predictors.split(",") if x.strip()]
    else:
        predictors = _pick_predictors(df)

    if not predictors:
        raise ValueError("No predictor columns found; pass --predictors")

    for c in predictors + ["visited", "TFF", "TFD", "FC"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    log1p_tff = bool(args.log1p_tff or args.log1p_ttff)
    log1p_tfd = bool(args.log1p_tfd or args.log1p_dwell)

    # z-scored predictors for comparability
    pred_z = []
    for p in predictors:
        zc = f"{p}_z"
        df[zc] = _z(df[p])
        pred_z.append(zc)

    interactions = []
    if args.interactions:
        for it in [x.strip() for x in args.interactions.split(",") if x.strip()]:
            a, b = [k.strip() for k in it.split(":", 1)]
            az, bz = f"{a}_z", f"{b}_z"
            if az in df.columns and bz in df.columns:
                name = f"{az}:{bz}"
                interactions.append(name)
    elif args.all_pairwise_interactions:
        for i in range(len(pred_z)):
            for j in range(i + 1, len(pred_z)):
                interactions.append(f"{pred_z[i]}:{pred_z[j]}")

    terms = pred_z + interactions

    all_rows = []

    # -------- visited (binary) --------
    try:
        import statsmodels.formula.api as smf
        import statsmodels.api as sm

        d1 = df.dropna(subset=["visited", "participant_id"] + pred_z).copy()
        d1["visited"] = (d1["visited"] > 0).astype(int)
        if d1["visited"].nunique() < 2:
            raise ValueError("visited has <2 classes")

        formula = "visited ~ " + _make_formula(terms, include_participant_fe=True)
        m1 = smf.glm(formula=formula, data=d1, family=sm.families.Binomial())
        r1 = m1.fit()
        with open(os.path.join(args.outdir, "model_visited_glm.txt"), "w", encoding="utf-8") as f:
            f.write(str(r1.summary()))

        keep_terms = [t for t in r1.params.index if t in terms]
        all_rows.append(_coef_table(r1, "visited", "GLM-Binomial+participantFE", keep_terms, n=len(d1)))
    except Exception as e:
        with open(os.path.join(args.outdir, "model_visited_glm.txt"), "w", encoding="utf-8") as f:
            f.write(f"FAILED: {repr(e)}\n")

    # -------- TFF given visited --------
    try:
        import statsmodels.formula.api as smf

        d2 = df[df["visited"] == 1].copy()
        d2 = d2.dropna(subset=["TFF", "participant_id"] + pred_z)
        if log1p_tff:
            d2["y"] = np.log1p(d2["TFF"].clip(lower=0))
        else:
            d2["y"] = d2["TFF"]
        formula = "y ~ " + _make_formula(terms)
        re_formula = "~1"
        if args.random_slope and f"{args.random_slope}_z" in d2.columns:
            re_formula = f"~{args.random_slope}_z"
        m2 = smf.mixedlm(formula=formula, data=d2, groups=d2["participant_id"], re_formula=re_formula)
        r2 = m2.fit(reml=False, method="lbfgs")
        with open(os.path.join(args.outdir, "model_tff_mixedlm.txt"), "w", encoding="utf-8") as f:
            f.write(str(r2.summary()))

        keep_terms = [t for t in r2.params.index if t in terms]
        all_rows.append(_coef_table(r2, "TFF|visited", "MixedLM", keep_terms, n=len(d2)))
    except Exception as e:
        with open(os.path.join(args.outdir, "model_tff_mixedlm.txt"), "w", encoding="utf-8") as f:
            f.write(f"FAILED: {repr(e)}\n")

    # -------- TFD given visited --------
    try:
        import statsmodels.formula.api as smf

        d3 = df[df["visited"] == 1].copy()
        d3 = d3.dropna(subset=["TFD", "participant_id"] + pred_z)
        if log1p_tfd:
            d3["y"] = np.log1p(d3["TFD"].clip(lower=0))
        else:
            d3["y"] = d3["TFD"]
        formula = "y ~ " + _make_formula(terms)
        re_formula = "~1"
        if args.random_slope and f"{args.random_slope}_z" in d3.columns:
            re_formula = f"~{args.random_slope}_z"
        m3 = smf.mixedlm(formula=formula, data=d3, groups=d3["participant_id"], re_formula=re_formula)
        r3 = m3.fit(reml=False, method="lbfgs")
        with open(os.path.join(args.outdir, "model_tfd_mixedlm.txt"), "w", encoding="utf-8") as f:
            f.write(str(r3.summary()))

        keep_terms = [t for t in r3.params.index if t in terms]
        all_rows.append(_coef_table(r3, "TFD|visited", "MixedLM", keep_terms, n=len(d3)))
    except Exception as e:
        with open(os.path.join(args.outdir, "model_tfd_mixedlm.txt"), "w", encoding="utf-8") as f:
            f.write(f"FAILED: {repr(e)}\n")

    # -------- FC given visited --------
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from statsmodels.genmod.cov_struct import Exchangeable

        d4 = df[df["visited"] == 1].copy()
        d4 = d4.dropna(subset=["FC", "participant_id"] + pred_z)
        d4["FC"] = d4["FC"].clip(lower=0)

        family = sm.families.NegativeBinomial() if args.count_family == "nb" else sm.families.Poisson()
        formula = "FC ~ " + _make_formula(terms)
        m4 = smf.gee(formula=formula, groups="participant_id", data=d4, cov_struct=Exchangeable(), family=family)
        r4 = m4.fit()
        with open(os.path.join(args.outdir, f"model_fc_gee_{args.count_family}.txt"), "w", encoding="utf-8") as f:
            f.write(str(r4.summary()))

        keep_terms = [t for t in r4.params.index if t in terms]
        all_rows.append(_coef_table(r4, "FC|visited", f"GEE-{args.count_family}", keep_terms, n=len(d4)))
    except Exception as e:
        with open(os.path.join(args.outdir, f"model_fc_gee_{args.count_family}.txt"), "w", encoding="utf-8") as f:
            f.write(f"FAILED: {repr(e)}\n")

    if all_rows:
        coef = pd.concat(all_rows, ignore_index=True)
    else:
        coef = pd.DataFrame(columns=["outcome", "model", "term", "coef", "se", "stat", "p", "ci_low", "ci_high", "n"])

    coef["q"] = _bh_fdr(coef["p"]) if len(coef) else []
    coef.to_csv(os.path.join(args.outdir, "model_coefficients_tidy.csv"), index=False)

    # readable renamed terms
    term_map = {f"{p}_z": p for p in predictors}
    coef_plot = coef.copy()
    coef_plot["term"] = coef_plot["term"].replace(term_map)
    for p1 in predictors:
        for p2 in predictors:
            if p1 == p2:
                continue
            coef_plot["term"] = coef_plot["term"].str.replace(f"{p1}_z:{p2}_z", f"{p1}:{p2}", regex=False)

    for outc, sub in coef_plot.groupby("outcome"):
        _plot_forest(sub, os.path.join(args.outdir, f"coef_forest_{outc}.png"), f"{outc} coefficients")

    _plot_dominance(coef_plot, os.path.join(args.outdir, "coef_dominance_overview.png"))

    # run info
    with open(os.path.join(args.outdir, "RUNINFO.txt"), "w", encoding="utf-8") as f:
        f.write("Paper-oriented AOI modeling\n")
        f.write(f"analysis_csv={args.analysis_csv}\n")
        f.write(f"class_name={args.class_name}\n")
        f.write(f"predictors={predictors}\n")
        f.write(f"interactions={interactions}\n")
        f.write(f"random_slope={args.random_slope}\n")
        f.write(f"count_family={args.count_family}\n")
        f.write(f"log1p_tff={log1p_tff}\n")
        f.write(f"log1p_tfd={log1p_tfd}\n")

    print("Saved to", args.outdir)


if __name__ == "__main__":
    main()
