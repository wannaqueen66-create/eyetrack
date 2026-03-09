from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


TRIAL_COL_RE = re.compile(r"^(trial\d{2})_(.+)$", flags=re.IGNORECASE)


def safe_num(s):
    return pd.to_numeric(s, errors="coerce")


def normalize_condition_token(value) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    u = s.upper().replace(" ", "")
    m = re.search(r"WWR?(15|45|75).*?C([01])", u)
    if m:
        return f"WWR{m.group(1)}_C{m.group(2)}"
    m = re.search(r"C([01]).*?W(?:WR)?(15|45|75)", u)
    if m:
        return f"WWR{m.group(2)}_C{m.group(1)}"
    m = re.search(r"W(?:WR)?(15|45|75).*?C([01])", u)
    if m:
        return f"WWR{m.group(1)}_C{m.group(2)}"
    return None


def _norm_complexity(value) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip().upper()
    if not s:
        return None
    if s in {"0", "C0", "LOW", "L"}:
        return "C0"
    if s in {"1", "C1", "HIGH", "H"}:
        return "C1"
    m = re.search(r"C([01])", s)
    return f"C{m.group(1)}" if m else None


def _norm_round(value) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan
    m = re.search(r"([12])", s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def _norm_order_sort(value) -> tuple[int, str]:
    if pd.isna(value):
        return (1, "")
    s = str(value).strip()
    if not s:
        return (1, "")
    m = re.search(r"\d+", s)
    if m:
        return (0, f"{int(m.group(0)):06d}")
    return (0, s.upper())


def _first_nonempty(*values):
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def _candidate_strings(*values) -> list[str]:
    out: list[str] = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s:
            out.append(s)
    return list(dict.fromkeys(out))


def extract_manifest_trials(gm: pd.DataFrame, id_col: str = "name") -> pd.DataFrame:
    if id_col not in gm.columns:
        raise ValueError(f"group_manifest missing id col: {id_col}")

    gm = gm.copy()
    gm["participant_id"] = gm[id_col].astype(str).str.strip()

    prefixes = sorted({m.group(1) for c in gm.columns for m in [TRIAL_COL_RE.match(c)] if m})
    rows: list[dict] = []
    for _, row in gm.iterrows():
        participant_id = str(row["participant_id"]).strip()
        order_group = row["Order"] if "Order" in gm.columns else np.nan
        for prefix in prefixes:
            digits = re.search(r"(\d+)$", prefix, flags=re.IGNORECASE)
            trial_index = int(digits.group(1)) if digits else np.nan
            trial_scene = row.get(f"{prefix}_scene")
            trial_pos = row.get(f"{prefix}_Pos")
            trial_key = row.get(f"{prefix}_key")
            trial_label = row.get(f"{prefix}_label")
            trial_round = row.get(f"{prefix}_Round")
            trial_cond = row.get(f"{prefix}_Cond")
            trial_complexity = row.get(f"{prefix}_Complexity")
            trial_wwr = row.get(f"{prefix}_WWR")
            if all(pd.isna(v) or not str(v).strip() for v in [trial_scene, trial_pos, trial_key, trial_label, trial_cond, trial_complexity, trial_wwr]):
                continue

            condition_id = normalize_condition_token(_first_nonempty(trial_cond, trial_scene, trial_key, trial_label))
            complexity = _norm_complexity(_first_nonempty(trial_complexity, condition_id, trial_scene, trial_key, trial_label))
            wwr = safe_num(pd.Series([_first_nonempty(trial_wwr, condition_id, trial_scene, trial_key, trial_label)])).iloc[0]
            if pd.isna(wwr) and condition_id:
                m = re.search(r"WWR(15|45|75)", condition_id, flags=re.IGNORECASE)
                wwr = float(m.group(1)) if m else np.nan
            round_index = _norm_round(trial_round)
            if pd.isna(round_index) and pd.notna(trial_index):
                round_index = float(((int(trial_index) - 1) // 6) + 1)

            scene_label = _first_nonempty(trial_scene, trial_label, condition_id, trial_key, trial_pos)
            label_bits = []
            if pd.notna(order_group) and str(order_group).strip():
                label_bits.append(f"O{str(order_group).strip()}")
            if pd.notna(trial_index):
                label_bits.append(f"T{int(trial_index):02d}")
            if pd.notna(round_index):
                label_bits.append(f"R{int(round_index)}")
            if scene_label:
                label_bits.append(str(scene_label).strip())

            order_bucket, order_key = _norm_order_sort(order_group)
            rows.append({
                "participant_id": participant_id,
                "order_group": order_group,
                "order_sort_bucket": order_bucket,
                "order_sort_key": order_key,
                "trial_prefix": prefix,
                "scene_order": trial_index,
                "scene_pos": trial_pos,
                "scene_key": trial_key,
                "scene_label": scene_label,
                "scene_round": round_index,
                "round": round_index,
                "condition_id": condition_id,
                "WWR_manifest": float(wwr) if pd.notna(wwr) else np.nan,
                "Complexity_manifest": complexity,
                "scene_manifest": trial_scene,
                "scene_label_raw": trial_label,
                "scene_display": " ".join(label_bits).strip(),
                "scene_view_id": "|".join(label_bits).strip() if label_bits else None,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out


def _unique_exact_lookup(manifest_long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, r in manifest_long.iterrows():
        vals = _candidate_strings(r.get("scene_manifest"), r.get("scene_pos"), r.get("scene_key"), r.get("scene_label_raw"), r.get("scene_label"))
        for v in vals:
            rows.append({"participant_id": r["participant_id"], "match_value": v, **r.to_dict()})
    if not rows:
        return pd.DataFrame()
    lk = pd.DataFrame(rows)
    counts = lk.groupby(["participant_id", "match_value"]).size().rename("n").reset_index()
    lk = lk.merge(counts, on=["participant_id", "match_value"], how="left")
    lk = lk[lk["n"] == 1].drop(columns=["n"])
    return lk


def _unique_cond_round_lookup(manifest_long: pd.DataFrame) -> pd.DataFrame:
    if manifest_long.empty or "condition_id" not in manifest_long.columns:
        return pd.DataFrame()
    cols = [c for c in ["participant_id", "condition_id", "round", "order_group", "order_sort_bucket", "order_sort_key", "scene_order", "scene_pos", "scene_key", "scene_label", "scene_display", "scene_view_id", "WWR_manifest", "Complexity_manifest"] if c in manifest_long.columns]
    lk = manifest_long[cols].dropna(subset=["condition_id"]).copy()
    if lk.empty or "round" not in lk.columns:
        return pd.DataFrame()
    counts = lk.groupby(["participant_id", "condition_id", "round"]).size().rename("n").reset_index()
    lk = lk.merge(counts, on=["participant_id", "condition_id", "round"], how="left")
    lk = lk[lk["n"] == 1].drop(columns=["n"])
    return lk


def attach_manifest_trial_metadata(df: pd.DataFrame, gm: pd.DataFrame, id_col: str = "name", scene_col: str = "scene_id") -> pd.DataFrame:
    manifest_long = extract_manifest_trials(gm, id_col=id_col)
    if manifest_long.empty or df.empty:
        return df.copy()

    out = df.copy()
    out["participant_id"] = out["participant_id"].astype(str).str.strip()
    out[scene_col] = out[scene_col].astype(str).str.strip()

    exact_lookup = _unique_exact_lookup(manifest_long)
    matched = out.copy()
    meta_cols = [
        "order_group", "order_sort_bucket", "order_sort_key", "scene_order", "scene_pos", "scene_key",
        "scene_label", "scene_display", "scene_view_id", "round", "condition_id", "WWR_manifest", "Complexity_manifest",
    ]
    for c in meta_cols:
        if c not in matched.columns:
            matched[c] = np.nan

    if not exact_lookup.empty:
        merge_cols = ["participant_id", scene_col]
        tmp = exact_lookup.rename(columns={"match_value": scene_col})
        tmp = tmp[[c for c in merge_cols + meta_cols if c in tmp.columns]].drop_duplicates()
        matched = matched.merge(tmp, on=merge_cols, how="left", suffixes=("", "__m1"))
        for c in meta_cols:
            mc = f"{c}__m1"
            if mc in matched.columns:
                matched[c] = matched[c].where(matched[c].notna(), matched[mc])
                matched = matched.drop(columns=[mc])

    if "condition_id" in matched.columns and "round" in matched.columns:
        cond_round_lookup = _unique_cond_round_lookup(manifest_long)
        if not cond_round_lookup.empty:
            need = matched[matched["scene_order"].isna()].copy()
            if not need.empty:
                need = need.merge(cond_round_lookup, on=["participant_id", "condition_id", "round"], how="left", suffixes=("", "__m2"))
                for c in meta_cols:
                    mc = f"{c}__m2"
                    if mc in need.columns:
                        need[c] = need[c].where(need[c].notna(), need[mc])
                        need = need.drop(columns=[mc])
                keep_cols = [c for c in matched.columns if c in need.columns]
                matched = pd.concat([matched[matched["scene_order"].notna()], need[keep_cols]], ignore_index=True)

    matched["scene_order"] = safe_num(matched["scene_order"])
    matched["round"] = safe_num(matched["round"])
    matched["WWR_manifest"] = safe_num(matched["WWR_manifest"])

    if "WWR" in matched.columns:
        matched["WWR"] = safe_num(matched["WWR"])
        matched["WWR"] = matched["WWR"].where(matched["WWR"].notna(), matched["WWR_manifest"])
    elif "WWR_manifest" in matched.columns:
        matched["WWR"] = matched["WWR_manifest"]

    if "Complexity" in matched.columns:
        matched["Complexity"] = matched["Complexity"].where(matched["Complexity"].notna(), matched["Complexity_manifest"])
    elif "Complexity_manifest" in matched.columns:
        matched["Complexity"] = matched["Complexity_manifest"]

    if scene_col != "scene_id" and "scene_id" in matched.columns and matched["scene_id"].dtype != object:
        matched["scene_id"] = matched["scene_id"].astype(str)

    return matched
