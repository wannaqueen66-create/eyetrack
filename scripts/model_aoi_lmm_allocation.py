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
    dwell_time_ms (or TFD), TTFF, fixation_count (or FC), visited
    round, WWR, Complexity   (added by our patched batch_aoi_metrics.py)
- --group_manifest: optional CSV with columns: name(or id) + SportFreq + Experience

Outputs
-------
For each group variable (Experience, SportFreq) and each outcome, export:
- model summary txt
- fixed-effect table csv (coef / SE / z / p / 95% CI)
- random-effect variance-components csv
- model-fit csv (AIC/BIC/logLik/nobs/convergence + R²)
- key simple-effects / contrasts csv around WWR × Complexity × Group
- fixed-effect forest plot png
- per-group README for how to read the outputs

Outcomes implemented
-------------------
1) tfd_y: log1p(TFD)
2) share_logit: logit( share_of_total_fixation_duration_within_trial )  (exploratory allocation proxy)
3) ttff_y: log1p(TTFF) on visited==1
4) fc_y: log1p(FC) on visited==1 (approximate; count-GLMM is better but out of scope)

Model form (statsmodels MixedLM)
-------------------------------
Fixed:
  C(class_name) * WWR_z * Complexity_z * C(GroupVar) + C(round)
Random:
  - random intercept by participant_id
  - variance component random intercept by scene_id_raw (if present)

Effect size / R² note
---------------------
The script exports Nakagawa-style approximate marginal / conditional R² for Gaussian mixed models:
- marginal R² = Var(fixed fitted values) / total variance
- conditional R² = (Var(fixed fitted values) + random-effect variance) / total variance
where total variance = fixed + random-intercept components + residual variance.
If the random-effect structure becomes singular, the script still writes the most stable
variance decomposition it can estimate and records notes in the model-fit table.

Notes
-----
- This is a pragmatic MixedLM implementation in Python (statsmodels).
- Binary/count outcomes are still better handled via GLMM in R (lme4/glmmTMB) for final inference.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aoi_metrics import normalize_aoi_class_series, normalize_aoi_class_name


AOI_PRIORITY = ["table", "window", "equipment"]
OUTCOME_LABELS = {
    "tfd_y": "log1p(TFD)",
    "share_logit": "logit(attention share from TFD share)",
    "share_pct": "attention share percentage (TFD-based)",
    "ttff_y": "log1p(TTFF), visited==1",
    "fc_y": "log1p(FC), visited==1",
    "fc_share_logit": "logit(FC_share within trial)",
    "FC_share": "FC share within trial",
    "FC_rate": "FC rate per second",
}
CI_ALPHA_DEFAULT = 0.05
RANDOM_VAR_TOL = 1e-8


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
    x = s.copy()
    if x.dtype == object:
        u = x.astype(str).str.strip().str.upper()
        out = pd.Series(np.nan, index=x.index, dtype=float)
        out[u.isin(["C0", "0", "LOW", "L"])] = 0.0
        out[u.isin(["C1", "1", "HIGH", "H"])] = 1.0
        m = u.str.extract(r"C(?P<c>[01])", expand=True)["c"]
        out = out.fillna(pd.to_numeric(m, errors="coerce"))
        return out
    return _safe_num(x)


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ci_from_estimate(estimate: float, se: float, alpha: float = CI_ALPHA_DEFAULT) -> tuple[float, float]:
    if not (np.isfinite(estimate) and np.isfinite(se)):
        return (np.nan, np.nan)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    return (estimate - zcrit * se, estimate + zcrit * se)


def _safe_float(x) -> float:
    try:
        v = float(x)
    except Exception:
        return np.nan
    return v if np.isfinite(v) else np.nan


def _is_bad_se_or_ci(fixef_df: pd.DataFrame) -> bool:
    if fixef_df is None or fixef_df.empty:
        return False
    bad_se = (~np.isfinite(pd.to_numeric(fixef_df.get("se"), errors="coerce"))).any()
    ci_low = pd.to_numeric(fixef_df.get("ci_low"), errors="coerce")
    ci_high = pd.to_numeric(fixef_df.get("ci_high"), errors="coerce")
    bad_ci_missing = (~np.isfinite(ci_low) | ~np.isfinite(ci_high)).any()
    bad_ci_order = (ci_low > ci_high).fillna(False).any()
    return bool(bad_se or bad_ci_missing or bad_ci_order)


def _collect_stability_signals(
    res,
    fixef_df: pd.DataFrame,
    random_df: pd.DataFrame,
    captured_warnings: list[warnings.WarningMessage],
    alpha: float = CI_ALPHA_DEFAULT,
) -> dict:
    warning_texts = [str(w.message).strip() for w in captured_warnings if getattr(w, "message", None) is not None]
    warning_text = " | ".join([w for w in warning_texts if w])
    lower_warning_text = warning_text.lower()

    converged = bool(getattr(res, "converged", False))
    method = str(getattr(res, "method", "") or "")
    scale = _safe_float(getattr(res, "scale", np.nan))

    random_df_local = random_df.copy() if random_df is not None else pd.DataFrame()
    non_resid = random_df_local.loc[random_df_local.get("component", pd.Series(dtype=str)) != "residual"].copy() if len(random_df_local) else pd.DataFrame()
    random_vars = pd.to_numeric(non_resid.get("variance"), errors="coerce") if len(non_resid) else pd.Series(dtype=float)
    near_zero_random = bool(len(random_vars) and (random_vars.fillna(np.nan) <= RANDOM_VAR_TOL).all())
    any_zero_random = bool(len(random_vars) and (random_vars.fillna(np.nan) <= RANDOM_VAR_TOL).any())

    hessian_non_pd = any(
        key in lower_warning_text
        for key in ["hessian", "not positive definite", "non positive definite", "non-positive definite"]
    )
    singular_warning = any(
        key in lower_warning_text
        for key in ["singular", "boundary", "on the boundary", "random effects covariance is singular"]
    )
    grad_warning = any(
        key in lower_warning_text
        for key in ["gradient", "optimization failed", "failed to converge", "did not converge", "maximum likelihood optimization failed"]
    )
    se_or_ci_issue = _is_bad_se_or_ci(fixef_df)

    notes: list[str] = []
    severe_reasons: list[str] = []
    caution_reasons: list[str] = []

    if not converged:
        severe_reasons.append("not_converged")
        notes.append("Model result reports converged=False.")
    if hessian_non_pd:
        severe_reasons.append("hessian_non_pd")
        notes.append("Optimizer/model warnings mention Hessian not positive definite.")
    if se_or_ci_issue:
        severe_reasons.append("se_or_ci_abnormal")
        notes.append("At least one fixed effect has non-finite SE/CI or CI bounds reversed.")

    if singular_warning:
        caution_reasons.append("random_effects_singular_or_boundary")
        notes.append("Warnings suggest singular or boundary random-effects fit.")
    if near_zero_random:
        caution_reasons.append("random_effect_variance_near_zero")
        notes.append("All non-residual random-effect variance components are near zero.")
    elif any_zero_random:
        caution_reasons.append("some_random_effect_variance_near_zero")
        notes.append("At least one non-residual random-effect variance component is near zero.")
    if grad_warning and converged:
        caution_reasons.append("optimizer_warning")
        notes.append("Warnings mention gradient/optimizer issues despite converged=True.")
    if not np.isfinite(scale) or scale <= 0:
        caution_reasons.append("residual_scale_non_positive")
        notes.append("Residual scale is non-finite or non-positive.")
    if alpha != CI_ALPHA_DEFAULT:
        caution_reasons.append("ci_alpha_non_default")
        notes.append(f"CI alpha={alpha:g} differs from repo default {CI_ALPHA_DEFAULT:g}.")
    else:
        notes.append(f"CI alpha uses default {CI_ALPHA_DEFAULT:g} (95% CI).")

    if severe_reasons:
        grade = "unstable"
    elif caution_reasons:
        grade = "caution"
    else:
        grade = "stable"
        notes.append("No major convergence, Hessian, singular-fit, or SE/CI anomalies detected.")

    grade_rank = {"stable": 1, "caution": 2, "unstable": 3}[grade]
    trigger_codes = severe_reasons + [x for x in caution_reasons if x not in severe_reasons]

    return {
        "stability_grade": grade,
        "stability_grade_rank": grade_rank,
        "stability_reasons": "; ".join(trigger_codes),
        "stability_notes": " ".join(dict.fromkeys(notes)),
        "warnings_text": warning_text,
        "warning_count": len(warning_texts),
        "converged": converged,
        "hessian_non_pd": hessian_non_pd,
        "optimizer_warning": grad_warning,
        "singular_or_boundary_warning": singular_warning,
        "random_effects_all_near_zero": near_zero_random,
        "random_effects_any_near_zero": any_zero_random,
        "se_or_ci_abnormal": se_or_ci_issue,
        "ci_alpha": alpha,
        "fit_method": method,
        "scale": scale,
    }


def _rename_test_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "z" in out.columns:
        out = out.rename(columns={"z": "test_stat"})
    elif "t" in out.columns:
        out = out.rename(columns={"t": "test_stat"})
    return out


def _tidy_fixef(res) -> pd.DataFrame:
    fe_params = pd.Series(res.fe_params)
    bse_all = pd.Series(res.bse)
    bse = bse_all.reindex(fe_params.index)
    zvals = pd.Series(np.asarray(fe_params, dtype=float) / np.asarray(bse, dtype=float), index=fe_params.index)
    pvals = pd.Series(2 * stats.norm.sf(np.abs(np.asarray(zvals, dtype=float))), index=fe_params.index)

    try:
        ci = res.conf_int()
        if isinstance(ci, np.ndarray):
            ci = pd.DataFrame(ci, index=pd.Index(res.params.index), columns=["ci_low", "ci_high"])
        else:
            ci = ci.rename(columns={0: "ci_low", 1: "ci_high"})
        ci = ci.reindex(fe_params.index)
    except Exception:
        ci = pd.DataFrame(index=fe_params.index, columns=["ci_low", "ci_high"], data=np.nan)

    out = pd.DataFrame(
        {
            "term": fe_params.index,
            "coef": np.asarray(fe_params, dtype=float),
            "se": np.asarray(bse, dtype=float),
            "z": np.asarray(zvals, dtype=float),
            "p": np.asarray(pvals, dtype=float),
            "ci_low": np.asarray(ci["ci_low"], dtype=float),
            "ci_high": np.asarray(ci["ci_high"], dtype=float),
        }
    )
    out["abs_test"] = out["z"].abs()
    return _rename_test_column(out)


def _extract_random_effects(res, df_model: pd.DataFrame, group_col: str, vc_scene_col: str | None) -> pd.DataFrame:
    rows: list[dict] = []

    try:
        cov_re = res.cov_re
        if cov_re is not None and len(cov_re.index) > 0:
            for i, row_name in enumerate(cov_re.index):
                var = float(cov_re.iloc[i, i])
                rows.append(
                    {
                        "component": f"{group_col}_random_intercept",
                        "grouping_factor": group_col,
                        "term": str(row_name),
                        "variance": var,
                        "sd": math.sqrt(var) if var >= 0 else np.nan,
                        "n_levels": int(df_model[group_col].nunique(dropna=True)),
                        "source": "cov_re",
                    }
                )
    except Exception:
        pass

    try:
        vcomp = np.asarray(getattr(res, "vcomp", []), dtype=float)
    except Exception:
        vcomp = np.array([], dtype=float)

    if vcomp.size:
        if vc_scene_col and vc_scene_col in df_model.columns:
            labels = [f"{vc_scene_col}_vc_random_intercept"]
            grouping = [vc_scene_col]
            n_levels = [int(df_model[vc_scene_col].nunique(dropna=True))]
        else:
            labels = [f"vc_{i+1}" for i in range(vcomp.size)]
            grouping = ["variance_component"] * vcomp.size
            n_levels = [np.nan] * vcomp.size
        for label, grp, nlev, var in zip(labels, grouping, n_levels, vcomp):
            rows.append(
                {
                    "component": label,
                    "grouping_factor": grp,
                    "term": "intercept",
                    "variance": float(var),
                    "sd": math.sqrt(var) if np.isfinite(var) and var >= 0 else np.nan,
                    "n_levels": nlev,
                    "source": "vcomp",
                }
            )

    resid = float(getattr(res, "scale", np.nan))
    rows.append(
        {
            "component": "residual",
            "grouping_factor": "residual",
            "term": "residual",
            "variance": resid,
            "sd": math.sqrt(resid) if np.isfinite(resid) and resid >= 0 else np.nan,
            "n_levels": len(df_model),
            "source": "scale",
        }
    )

    return pd.DataFrame(rows)


def _approx_r2(res, random_df: pd.DataFrame) -> dict:
    out = {
        "r2_marginal": np.nan,
        "r2_conditional": np.nan,
        "var_fixed": np.nan,
        "var_random_total": np.nan,
        "var_residual": np.nan,
        "r2_method": "Nakagawa-style approximate Gaussian mixed-model R2 based on fixed fitted variance + random intercept variance + residual variance",
        "r2_note": "",
    }
    try:
        fitted_fixed = np.asarray(res.model.exog @ np.asarray(res.fe_params), dtype=float)
        var_fixed = float(np.nanvar(fitted_fixed, ddof=0))
    except Exception as e:
        out["r2_note"] = f"failed_fixed_variance:{repr(e)}"
        return out

    rand = 0.0
    if random_df is not None and len(random_df):
        rand = float(random_df.loc[random_df["component"] != "residual", "variance"].fillna(0).sum())
    resid = float(getattr(res, "scale", np.nan))
    total = var_fixed + rand + resid
    if np.isfinite(total) and total > 0:
        out["r2_marginal"] = var_fixed / total
        out["r2_conditional"] = (var_fixed + rand) / total
    out["var_fixed"] = var_fixed
    out["var_random_total"] = rand
    out["var_residual"] = resid
    if not np.isfinite(out["r2_marginal"]):
        out["r2_note"] = (out["r2_note"] + "; " if out["r2_note"] else "") + "non_finite_total_variance"
    return out


def _model_fit_table(
    res,
    df_model: pd.DataFrame,
    formula: str,
    outcome: str,
    group_var: str,
    subset_note: str,
    random_df: pd.DataFrame,
    stability: dict | None = None,
) -> pd.DataFrame:
    r2 = _approx_r2(res, random_df)
    row = {
        "outcome": outcome,
        "group_var": group_var,
        "formula": formula,
        "subset": subset_note,
        "nobs": int(getattr(res, "nobs", len(df_model))),
        "n_participants": int(df_model["participant_id"].nunique(dropna=True)),
        "n_scenes": int(df_model["scene_id_model"].nunique(dropna=True)) if "scene_id_model" in df_model.columns else np.nan,
        "n_aoi_classes": int(df_model["class_name"].nunique(dropna=True)),
        "aic": float(getattr(res, "aic", np.nan)),
        "bic": float(getattr(res, "bic", np.nan)),
        "logLik": float(getattr(res, "llf", np.nan)),
        "scale": float(getattr(res, "scale", np.nan)),
        "method": getattr(res, "method", ""),
        "converged": bool(getattr(res, "converged", False)),
        "r2_marginal": r2["r2_marginal"],
        "r2_conditional": r2["r2_conditional"],
        "var_fixed": r2["var_fixed"],
        "var_random_total": r2["var_random_total"],
        "var_residual": r2["var_residual"],
        "r2_method": r2["r2_method"],
        "r2_note": r2["r2_note"],
    }
    if stability:
        row.update(stability)
    return pd.DataFrame([row])


def fit_mixedlm(df: pd.DataFrame, formula: str, group_col: str, vc_scene_col: str | None = None):
    vc = None
    if vc_scene_col and vc_scene_col in df.columns:
        vc = {"scene": f"0 + C({vc_scene_col})"}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if vc:
            model = smf.mixedlm(formula, data=df, groups=df[group_col], vc_formula=vc)
        else:
            model = smf.mixedlm(formula, data=df, groups=df[group_col])
        res = model.fit(reml=False, method="lbfgs")
    return res, list(caught)


def _parse_factor_levels(term: str, factor_name: str) -> str | None:
    pat = rf"C\({re.escape(factor_name)}\)\[T\.(.*?)\]"
    m = re.search(pat, term)
    if not m:
        return None
    return m.group(1)


def _class_reference_level(df_model: pd.DataFrame) -> str:
    vals = [normalize_aoi_class_name(x) for x in df_model["class_name"].dropna().astype(str).unique().tolist()]
    vals = sorted(set(vals))
    for wanted in AOI_PRIORITY:
        if wanted in vals:
            return wanted
    return vals[0]


def _group_levels(df_model: pd.DataFrame, group_var: str) -> list[str]:
    vals = [_norm_hilo(x) for x in df_model[group_var].dropna().astype(str).tolist()]
    vals = [x for x in vals if pd.notna(x)]
    uniq = []
    for x in vals:
        if x not in uniq:
            uniq.append(x)
    preferred = [x for x in ["Low", "High"] if x in uniq]
    if len(preferred) >= 2:
        return preferred
    return sorted(uniq)


def _round_profiles(df_model: pd.DataFrame) -> list[dict]:
    if "round" not in df_model.columns or not df_model["round"].notna().any():
        return [{}]
    probs = df_model["round"].dropna().astype(int).value_counts(normalize=True).sort_index()
    out = []
    for lev, wt in probs.items():
        out.append({"round": int(lev), "weight": float(wt)})
    return out or [{}]


def _make_design_row_average(res, rows: list[dict]) -> np.ndarray:
    di = res.model.data.design_info
    mats = build_design_matrices([di], pd.DataFrame(rows), return_type="dataframe")
    x = np.asarray(mats[0], dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x = x[:, : len(res.fe_params)]
    return x.mean(axis=0)


def _contrast_from_vectors(res, lhs: np.ndarray, rhs: np.ndarray) -> tuple[float, float, float, float, float]:
    c = np.asarray(lhs, dtype=float) - np.asarray(rhs, dtype=float)
    beta = np.asarray(res.fe_params, dtype=float)
    cov = np.asarray(res.cov_params().iloc[: len(beta), : len(beta)], dtype=float)
    est = float(c @ beta)
    se2 = float(c @ cov @ c)
    se = math.sqrt(se2) if se2 >= 0 else np.nan
    z = est / se if np.isfinite(se) and se > 0 else np.nan
    p = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
    ci_low, ci_high = _ci_from_estimate(est, se)
    return est, se, z, p, ci_low, ci_high


def _aoi_levels(df_model: pd.DataFrame) -> list[str]:
    vals = [normalize_aoi_class_name(x) for x in df_model["class_name"].dropna().astype(str).unique().tolist()]
    vals = list(dict.fromkeys(vals))
    ordered = [x for x in AOI_PRIORITY if x in vals]
    ordered += [x for x in sorted(vals) if x not in ordered]
    return ordered


def _key_wwr_values(df_model: pd.DataFrame) -> list[tuple[float, float, str]]:
    pairs = (
        df_model[["WWR", "WWR_z"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["WWR", "WWR_z"])
    )
    out = []
    for _, row in pairs.iterrows():
        out.append((float(row["WWR"]), float(row["WWR_z"]), f"WWR{int(row['WWR'])}"))
    return out


def _key_complexity_values(df_model: pd.DataFrame) -> list[tuple[float, str]]:
    pairs = (
        df_model[["Complexity_num", "Complexity_label"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["Complexity_num", "Complexity_label"])
    )
    return [(float(r["Complexity_num"]), str(r["Complexity_label"])) for _, r in pairs.iterrows()]


def _build_profile_rows(df_model: pd.DataFrame, class_name: str, group_var: str, group_level: str, wwr_z: float, complexity_z: float) -> list[dict]:
    rows = []
    for rp in _round_profiles(df_model):
        row = {
            "class_name": class_name,
            group_var: group_level,
            "WWR_z": float(wwr_z),
            "Complexity_z": float(complexity_z),
            "weight": rp.get("weight", 1.0),
        }
        if "round" in rp:
            row["round"] = rp["round"]
        rows.append(row)
    return rows


def _average_vector(res, rows: list[dict]) -> np.ndarray:
    weights = np.asarray([float(r.get("weight", 1.0)) for r in rows], dtype=float)
    weights = weights / weights.sum()
    rows_clean = [{k: v for k, v in r.items() if k != "weight"} for r in rows]
    di = res.model.data.design_info
    mats = build_design_matrices([di], pd.DataFrame(rows_clean), return_type="dataframe")
    x = np.asarray(mats[0], dtype=float)
    x = x[:, : len(res.fe_params)]
    return np.average(x, axis=0, weights=weights)


def _build_contrasts(res, df_model: pd.DataFrame, outcome: str, group_var: str) -> pd.DataFrame:
    group_levels = _group_levels(df_model, group_var)
    if len(group_levels) < 2:
        return pd.DataFrame()
    aoi_levels = _aoi_levels(df_model)
    wwr_levels = _key_wwr_values(df_model)
    comp_levels = _key_complexity_values(df_model)
    rows: list[dict] = []

    for aoi in aoi_levels:
        for comp_num, comp_label in comp_levels:
            for wwr_raw, wwr_z, wwr_label in wwr_levels:
                lhs = _average_vector(res, _build_profile_rows(df_model, aoi, group_var, group_levels[-1], wwr_z, comp_num))
                rhs = _average_vector(res, _build_profile_rows(df_model, aoi, group_var, group_levels[0], wwr_z, comp_num))
                est, se, z, p, lo, hi = _contrast_from_vectors(res, lhs, rhs)
                rows.append(
                    {
                        "contrast_family": "group_simple_effect",
                        "outcome": outcome,
                        "group_var": group_var,
                        "aoi": aoi,
                        "Complexity": comp_label,
                        "WWR": int(round(wwr_raw)),
                        "contrast_label": f"{group_levels[-1]} - {group_levels[0]} | {aoi}, {wwr_label}, {comp_label}",
                        "estimate": est,
                        "se": se,
                        "z": z,
                        "p": p,
                        "ci_low": lo,
                        "ci_high": hi,
                    }
                )

    if len(wwr_levels) >= 2:
        low_wwr_raw, low_wwr_z, low_wwr_label = wwr_levels[0]
        high_wwr_raw, high_wwr_z, high_wwr_label = wwr_levels[-1]
        for aoi in aoi_levels:
            for comp_num, comp_label in comp_levels:
                for g in group_levels:
                    lhs = _average_vector(res, _build_profile_rows(df_model, aoi, group_var, g, high_wwr_z, comp_num))
                    rhs = _average_vector(res, _build_profile_rows(df_model, aoi, group_var, g, low_wwr_z, comp_num))
                    est, se, z, p, lo, hi = _contrast_from_vectors(res, lhs, rhs)
                    rows.append(
                        {
                            "contrast_family": "wwr_simple_effect",
                            "outcome": outcome,
                            "group_var": group_var,
                            "aoi": aoi,
                            "Complexity": comp_label,
                            "WWR": np.nan,
                            "contrast_label": f"{high_wwr_label} - {low_wwr_label} | {aoi}, {comp_label}, {g}",
                            "group_level": g,
                            "estimate": est,
                            "se": se,
                            "z": z,
                            "p": p,
                            "ci_low": lo,
                            "ci_high": hi,
                        }
                    )

    if len(comp_levels) >= 2:
        low_comp_num, low_comp_label = comp_levels[0]
        high_comp_num, high_comp_label = comp_levels[-1]
        for aoi in aoi_levels:
            for wwr_raw, wwr_z, wwr_label in wwr_levels:
                for g in group_levels:
                    lhs = _average_vector(res, _build_profile_rows(df_model, aoi, group_var, g, wwr_z, high_comp_num))
                    rhs = _average_vector(res, _build_profile_rows(df_model, aoi, group_var, g, wwr_z, low_comp_num))
                    est, se, z, p, lo, hi = _contrast_from_vectors(res, lhs, rhs)
                    rows.append(
                        {
                            "contrast_family": "complexity_simple_effect",
                            "outcome": outcome,
                            "group_var": group_var,
                            "aoi": aoi,
                            "Complexity": np.nan,
                            "WWR": int(round(wwr_raw)),
                            "contrast_label": f"{high_comp_label} - {low_comp_label} | {aoi}, {wwr_label}, {g}",
                            "group_level": g,
                            "estimate": est,
                            "se": se,
                            "z": z,
                            "p": p,
                            "ci_low": lo,
                            "ci_high": hi,
                        }
                    )

    out = pd.DataFrame(rows)
    if len(out):
        out["abs_test"] = out["z"].abs()
        out = _rename_test_column(out)
    return out


def _forest_plot(fixef_df: pd.DataFrame, out_png: Path, title: str, stability_grade: str | None = None):
    if fixef_df.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    dfp = fixef_df.copy()
    dfp = dfp.sort_values(["abs_test", "p"], ascending=[False, True]).head(18).iloc[::-1].copy()
    dfp["effect_ci_label"] = dfp.apply(
        lambda r: f"b={float(r['coef']):.2f} [{float(r['ci_low']):.2f}, {float(r['ci_high']):.2f}]" if pd.notna(r['coef']) and pd.notna(r['ci_low']) and pd.notna(r['ci_high']) else "",
        axis=1,
    )
    fig_h = max(4.0, 0.45 * len(dfp) + 1.8)
    xmin = float(np.nanmin(dfp["ci_low"])) if dfp["ci_low"].notna().any() else -1.0
    xmax = float(np.nanmax(dfp["ci_high"])) if dfp["ci_high"].notna().any() else 1.0
    span = max(xmax - xmin, 1.0)
    fig, ax = plt.subplots(figsize=(11.5, fig_h))
    ax.axvline(0, color="#888888", linewidth=1.0, linestyle="--")
    y = np.arange(len(dfp))
    colors = np.where(dfp["p"].fillna(1.0) < 0.05, "#2C7FB8", "#9E9E9E")
    ax.hlines(y, dfp["ci_low"], dfp["ci_high"], color=colors, linewidth=2)
    ax.scatter(dfp["coef"], y, color=colors, s=30, zorder=3)
    for yi, (_, row) in enumerate(dfp.iterrows()):
        if row["effect_ci_label"]:
            xpos = float(row["ci_high"]) + 0.03 * span
            ax.text(xpos, yi, row["effect_ci_label"], va="center", ha="left", fontsize=7, color="#333333")
    ax.set_xlim(xmin - 0.08 * span, xmax + 0.55 * span)
    ax.set_yticks(y)
    ax.set_yticklabels(dfp["term"])
    ax.set_xlabel("Coefficient estimate (95% CI)")
    if stability_grade:
        ax.set_title(f"{title} | stability: {stability_grade}")
        dfp["stability_grade"] = stability_grade
    else:
        ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    dfp.to_csv(out_png.with_name(out_png.stem + "_data.csv"), index=False, encoding="utf-8-sig")
    plt.close(fig)


def _write_group_readme(gdir: Path, group_var: str):
    text = f"""AOI allocation LMM report: {group_var}

Files in this folder
--------------------
- model_stability_summary.csv
  One-row-per-model stability triage table. Read this first.
- group_size_summary_{group_var}.csv
  Sample-size summary by group (participants / rows / scenes). Use this to explain unbalanced designs.
- README_model_stability.txt
  Explains the grading rules and the recommended reading order.
- model_<outcome>.txt
  Raw statsmodels MixedLM summary for audit / troubleshooting.
- fixef_<outcome>.csv
  Fixed-effect table with coefficient, SE, Wald z, p, and 95% CI.
- ranef_<outcome>.csv
  Random-effect variance components (participant intercept, scene variance component if available, residual variance).
- model_fit_<outcome>.csv
  Model fit information including AIC, BIC, logLik, nobs, convergence, approximate marginal/conditional R², and embedded stability fields.
- contrasts_<outcome>.csv
  Key simple effects around WWR × Complexity × {group_var}, exported as linear contrasts from the fixed-effect covariance matrix.
- forest_fixef_<outcome>.png
  Forest plot of the strongest fixed effects by |z|, with inline effect/CI labels and a stability tag in the title.
- forest_fixef_<outcome>_data.csv
  Companion table for the forest plot (same terms/order and rendered labels).

How to interpret
----------------
1. Start with model_stability_summary.csv and identify whether each outcome is stable / caution / unstable.
2. For stable models, then inspect model_fit_<outcome>.csv and fixef_<outcome>.csv.
3. Use contrasts_<outcome>.csv for reviewer-facing simple-effects reporting around the target interaction.
4. Use ranef_<outcome>.csv to report variance decomposition / random intercept components.
5. If a model is caution/unstable, treat it as supplementary unless you can justify the warnings.

Stability grades
----------------
- stable: no major convergence, Hessian, singular-fit, or SE/CI anomalies detected.
- caution: fit converged, but warnings suggest singular/boundary random effects, near-zero random variances, or other softer optimizer concerns.
- unstable: non-converged, Hessian non-PD, or SE/CI output is abnormal/non-finite.

R² definition
-------------
Approximate Nakagawa-style Gaussian mixed-model R²:
- marginal R² = Var(Xβ) / [Var(Xβ) + ΣVar(random intercepts) + Var(residual)]
- conditional R² = [Var(Xβ) + ΣVar(random intercepts)] / [Var(Xβ) + ΣVar(random intercepts) + Var(residual)]

Contrast note
-------------
Contrasts are estimated from the fixed-effect design matrix while averaging over observed round proportions when round is included.
This keeps round as a nuisance adjustment rather than forcing a single arbitrary round level.
"""
    (gdir / "README_LMM_report.txt").write_text(text, encoding="utf-8")


def _write_stability_readme(outdir: Path):
    text = f"""Model stability grading for AOI allocation LMM outputs

Purpose
-------
This file helps you decide which model outputs are safe to use as the main statistical narrative,
and which ones should be treated as supplementary / cautionary.

Recommended reading order
-------------------------
1. Open each group folder's model_stability_summary.csv first.
2. Prioritize rows with stability_grade=stable as main-result candidates.
3. Treat stability_grade=caution as usable with explicit warning language.
4. Treat stability_grade=unstable as supplementary / diagnostic unless re-fit or alternative modeling resolves the issue.

Current grading logic
---------------------
- stable:
  - converged=True
  - no Hessian non-PD warning
  - no abnormal SE/CI output
  - no singular/boundary random-effects warning
- caution:
  - converged=True, but at least one softer warning exists, such as:
    - singular/boundary random-effects fit
    - some/all random-effect variance components near zero
    - optimizer/gradient warning despite converged=True
    - CI alpha not using the default {CI_ALPHA_DEFAULT:g}
- unstable:
  - converged=False, or
  - Hessian reported as non-positive-definite, or
  - fixed-effect SE/CI output is non-finite / malformed

Important caveat
----------------
These grades are triage labels for the current Python MixedLM outputs only. They do not replace substantive judgment,
and they do not make transformed LMMs equivalent to a dedicated GLMM for binary/count outcomes.

Count / binary note
-------------------
For visited / count-like outcomes, a GLMM in R (e.g. lme4/glmmTMB) is still preferable for final confirmatory inference.
"""
    (outdir / "README_model_stability.txt").write_text(text, encoding="utf-8")


def _write_group_size_summary(df: pd.DataFrame, outdir: Path, group_var: str):
    rows = []
    d = df.copy()
    d[group_var] = d[group_var].apply(_norm_hilo)
    d = d.dropna(subset=[group_var])
    if len(d) == 0:
        return

    for gv, sub in d.groupby(group_var, dropna=False):
        row = {
            'group_var': group_var,
            'group_value': gv,
            'n_rows': int(len(sub)),
            'n_participants': int(sub['participant_id'].nunique(dropna=True)),
            'n_scenes': int(sub['scene_id_model'].nunique(dropna=True)) if 'scene_id_model' in sub.columns else np.nan,
            'n_aoi_classes': int(sub['class_name'].nunique(dropna=True)),
        }
        if 'visited' in sub.columns:
            row['visited_rate_over_rows'] = float(pd.to_numeric(sub['visited'], errors='coerce').fillna(0).mean())
        rows.append(row)

    pdf = (
        d[['participant_id', group_var]]
        .dropna()
        .drop_duplicates()
        .groupby(group_var, dropna=False)
        .size()
        .rename('participant_count')
        .reset_index()
    )
    gdf = pd.DataFrame(rows).merge(pdf, left_on='group_value', right_on=group_var, how='left')
    if group_var in gdf.columns:
        gdf = gdf.drop(columns=[group_var])
    gdf.to_csv(outdir / f'group_size_summary_{group_var}.csv', index=False, encoding='utf-8-sig')


def _prepare_data(args) -> tuple[pd.DataFrame, list[str], str | None]:
    df = pd.read_csv(args.aoi_class_csv, encoding="utf-8-sig")

    if "participant_id" not in df.columns:
        raise SystemExit("aoi_class_csv missing participant_id")

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

    if "class_name" in df.columns:
        df["class_name"] = normalize_aoi_class_series(df["class_name"])

    need = ["participant_id", "class_name"]
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"aoi_class_csv missing required col: {c}")

    vc_scene_col = "scene_id_raw" if "scene_id_raw" in df.columns else ("scene_id" if "scene_id" in df.columns else None)
    if vc_scene_col:
        df["scene_id_model"] = df[vc_scene_col].astype(str)
    else:
        df["scene_id_model"] = "scene"

    if "trial_excluded" in df.columns:
        df["trial_excluded"] = _safe_num(df["trial_excluded"]).fillna(0).astype(int)
        df = df[df["trial_excluded"] == 0].copy()

    if "WWR" not in df.columns or "Complexity" not in df.columns:
        raise SystemExit("Missing WWR/Complexity columns. Re-run batch_aoi_metrics.py after latest patch.")

    dwell_col = _pick_col(df, "TFD", "dwell_time_ms")
    ttff_col = _pick_col(df, "TTFF")
    fc_col = _pick_col(df, "FC", "fixation_count")

    df["WWR"] = _safe_num(df["WWR"])
    df["Complexity_num"] = _complexity_to_num(df["Complexity"])
    df["Complexity_label"] = df["Complexity_num"].map({0.0: "C0", 1.0: "C1"}).fillna(df["Complexity"].astype(str))
    df["WWR_z"] = _z(df["WWR"])
    df["Complexity_z"] = _z(df["Complexity_num"])

    if "round" in df.columns:
        round_num = _safe_num(df["round"])
        df["round"] = np.where(round_num.notna(), round_num.astype(float), np.nan)
    else:
        df["round"] = np.nan

    if "visited" in df.columns:
        df["visited"] = _safe_num(df["visited"]).fillna(0).astype(int)
    else:
        df["visited"] = 1

    if dwell_col:
        df["TFD"] = _safe_num(df[dwell_col])
        df["tfd_y"] = np.log1p(df["TFD"].clip(lower=0))
        key_cols = ["participant_id", "scene_id_model"]
        tot = df.groupby(key_cols, dropna=False)["TFD"].sum(min_count=1).rename("TFD_total_ms")
        df = df.merge(tot.reset_index(), on=key_cols, how="left")
        eps = 1e-6
        df["share"] = df["TFD"] / df["TFD_total_ms"]
        df["share"] = df["share"].clip(lower=0, upper=1)
        df["share_logit"] = np.log((df["share"] + eps) / (1 - df["share"] + eps))

    if ttff_col:
        df["TTFF"] = _safe_num(df[ttff_col])
        df["ttff_y"] = np.log1p(df["TTFF"].clip(lower=0))

    if fc_col:
        df["FC"] = _safe_num(df[fc_col])
        df["fc_y"] = np.log1p(df["FC"].clip(lower=0))

    if "share_pct" in df.columns:
        df["share_pct"] = _safe_num(df["share_pct"]).clip(lower=0, upper=100)
    if "FC_share" in df.columns:
        df["FC_share"] = _safe_num(df["FC_share"]).clip(lower=0, upper=1)
        eps = 1e-6
        df["fc_share_logit"] = np.log((df["FC_share"] + eps) / (1 - df["FC_share"] + eps))
    if "FC_rate" in df.columns:
        df["FC_rate"] = _safe_num(df["FC_rate"]).clip(lower=0)

    group_vars = [gv for gv in ["Experience", "SportFreq"] if gv in df.columns]
    if not group_vars:
        raise SystemExit("No group variables found (Experience/SportFreq). Pass --group_manifest or ensure columns exist.")
    return df, group_vars, vc_scene_col


def main():
    ap = argparse.ArgumentParser(description="AOI allocation LMM (Experience & SportFreq) with fuller report export")
    ap.add_argument("--aoi_class_csv", required=True)
    ap.add_argument("--group_manifest", default=None)
    ap.add_argument("--group_id_col", default="name", help="ID column in group_manifest (default: name)")
    ap.add_argument("--outdir", default="outputs_aoi_lmm")
    ap.add_argument("--min_rows", type=int, default=40)
    args = ap.parse_args()

    outdir = _ensure_dir(args.outdir)
    df, group_vars, vc_scene_col = _prepare_data(args)

    outcomes = [
        ("share_logit", "All rows (allocation proxy, TFD share)"),
        ("share_pct", "All rows (allocation share %, TFD-based)"),
        ("fc_share_logit", "All rows (allocation proxy, FC share)"),
        ("FC_share", "All rows (FC share within trial)"),
        ("FC_rate", "All rows (FC per second)"),
        ("tfd_y", "All rows"),
        ("ttff_y", "visited==1"),
        ("fc_y", "visited==1"),
    ]

    runinfo = [
        "AOI LMM allocation run",
        f"aoi_class_csv: {args.aoi_class_csv}",
        f"group_manifest: {args.group_manifest}",
        f"group_id_col: {args.group_id_col}",
        f"min_rows: {args.min_rows}",
        "group vars: " + ",".join(group_vars),
        f"vc_scene_col: {vc_scene_col}",
        f"stability_ci_alpha_default: {CI_ALPHA_DEFAULT}",
        f"stability_random_var_tol: {RANDOM_VAR_TOL}",
        "outputs: model_stability_summary / fixef / ranef / model_fit / contrasts / forest plot / raw summary txt",
    ]

    for gv in group_vars:
        gdir = _ensure_dir(outdir / f"groupvar_{gv}")
        _write_group_readme(gdir, gv)
        _write_stability_readme(gdir)
        _write_group_size_summary(df, gdir, gv)
        stability_rows: list[dict] = []

        for ycol, subset_note in outcomes:
            if ycol not in df.columns:
                continue

            d = df.copy()
            d = d.dropna(subset=[ycol, "WWR_z", "Complexity_z", "class_name", gv, "participant_id"])
            if subset_note == "visited==1":
                d = d[d["visited"] == 1].copy()
            d = d.copy()
            d[gv] = d[gv].apply(_norm_hilo)
            d = d.dropna(subset=[gv])

            base_summary = {
                "group_var": gv,
                "outcome": ycol,
                "outcome_label": OUTCOME_LABELS.get(ycol, ycol),
                "subset": subset_note,
                "n": int(len(d)),
                "formula": "",
                "stability_grade": "unstable",
                "stability_grade_rank": 3,
                "stability_reasons": "not_run",
                "stability_notes": "Model was not fit.",
            }

            if len(d) < args.min_rows:
                (gdir / f"model_{ycol}.txt").write_text(
                    f"SKIP: too few rows for outcome={ycol} (n={len(d)}; min_rows={args.min_rows})\n",
                    encoding="utf-8",
                )
                base_summary.update(
                    {
                        "stability_reasons": "too_few_rows",
                        "stability_notes": f"Skipped because n={len(d)} < min_rows={args.min_rows}.",
                        "warning_count": 0,
                    }
                )
                stability_rows.append(base_summary)
                continue

            formula = f"{ycol} ~ C(class_name) * WWR_z * Complexity_z * C({gv})"
            if d["round"].notna().any():
                formula += " + C(round)"
            base_summary["formula"] = formula

            try:
                res, caught_warnings = fit_mixedlm(d, formula, group_col="participant_id", vc_scene_col="scene_id_model")

                fixef = _tidy_fixef(res)
                ranef = _extract_random_effects(res, d, group_col="participant_id", vc_scene_col=vc_scene_col)
                stability = _collect_stability_signals(res, fixef, ranef, caught_warnings)

                (gdir / f"model_{ycol}.txt").write_text(
                    "\n".join(
                        [
                            f"Outcome: {ycol}",
                            f"Outcome label: {OUTCOME_LABELS.get(ycol, ycol)}",
                            f"Subset: {subset_note}",
                            f"Formula: {formula}",
                            f"n={len(d)}",
                            f"vc_scene_col={vc_scene_col}",
                            f"stability_grade={stability['stability_grade']}",
                            f"stability_reasons={stability['stability_reasons']}",
                            f"warning_count={stability['warning_count']}",
                            "Warnings:",
                            stability["warnings_text"] or "(none)",
                            "",
                            str(res.summary()),
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )

                fixef.insert(0, "stability_grade", stability["stability_grade"])
                fixef.insert(0, "group_var", gv)
                fixef.insert(0, "outcome", ycol)
                fixef.insert(0, "n", int(len(d)))
                fixef.to_csv(gdir / f"fixef_{ycol}.csv", index=False, encoding="utf-8-sig")

                ranef.insert(0, "stability_grade", stability["stability_grade"])
                ranef.insert(0, "group_var", gv)
                ranef.insert(0, "outcome", ycol)
                ranef.to_csv(gdir / f"ranef_{ycol}.csv", index=False, encoding="utf-8-sig")

                fitdf = _model_fit_table(res, d, formula, ycol, gv, subset_note, ranef, stability=stability)
                fitdf.to_csv(gdir / f"model_fit_{ycol}.csv", index=False, encoding="utf-8-sig")

                contrasts = _build_contrasts(res, d, ycol, gv)
                if len(contrasts):
                    contrasts.insert(0, "stability_grade", stability["stability_grade"])
                    contrasts.to_csv(gdir / f"contrasts_{ycol}.csv", index=False, encoding="utf-8-sig")

                _forest_plot(
                    fixef,
                    gdir / f"forest_fixef_{ycol}.png",
                    f"{gv} | {ycol} fixed effects",
                    stability_grade=stability["stability_grade"],
                )

                stability_row = base_summary.copy()
                stability_row.update(stability)
                stability_row.update(
                    {
                        "n_participants": int(d["participant_id"].nunique(dropna=True)),
                        "n_scenes": int(d["scene_id_model"].nunique(dropna=True)) if "scene_id_model" in d.columns else np.nan,
                        "n_aoi_classes": int(d["class_name"].nunique(dropna=True)),
                        "aic": float(getattr(res, "aic", np.nan)),
                        "bic": float(getattr(res, "bic", np.nan)),
                        "logLik": float(getattr(res, "llf", np.nan)),
                    }
                )
                stability_rows.append(stability_row)
            except Exception as e:
                (gdir / f"model_{ycol}.txt").write_text(
                    "\n".join(
                        [
                            f"FAILED outcome={ycol}",
                            f"Subset: {subset_note}",
                            f"Formula: {formula}",
                            f"n={len(d)}",
                            repr(e),
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                base_summary.update(
                    {
                        "stability_reasons": "fit_failed",
                        "stability_notes": repr(e),
                        "warning_count": 0,
                    }
                )
                stability_rows.append(base_summary)

        if stability_rows:
            stab_df = pd.DataFrame(stability_rows)
            sort_cols = [c for c in ["stability_grade_rank", "outcome"] if c in stab_df.columns]
            if sort_cols:
                stab_df = stab_df.sort_values(sort_cols, ascending=[True] * len(sort_cols))
            stab_df.to_csv(gdir / "model_stability_summary.csv", index=False, encoding="utf-8-sig")

    (outdir / "RUNINFO.txt").write_text("\n".join(runinfo) + "\n", encoding="utf-8")
    print("Saved:", outdir)


if __name__ == "__main__":
    main()
