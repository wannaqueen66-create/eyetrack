import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _invert_map(required_to_candidates: Dict[str, List[str]]) -> Dict[str, str]:
    """Return mapping from candidate column name -> required column name."""
    inv: Dict[str, str] = {}
    for req, candidates in required_to_candidates.items():
        for c in candidates:
            if c and c not in inv:
                inv[c] = req
    return inv


def load_columns_map(path: Optional[str] = None) -> Dict[str, List[str]]:
    """Load required->candidates JSON. If path is None, uses configs/columns_default.json."""
    if path is None:
        path = str(Path(__file__).resolve().parents[1] / 'configs' / 'columns_default.json')
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    # normalize values to list
    out: Dict[str, List[str]] = {}
    for k, v in d.items():
        if isinstance(v, list):
            out[k] = [str(x) for x in v]
        else:
            out[k] = [str(v)]
    return out


def resolve_columns(df_columns: Iterable[str], required_to_candidates: Dict[str, List[str]]) -> Dict[str, str]:
    """Resolve required column names to actual columns present in the dataframe.

    Returns dict: required_name -> actual_name
    """
    df_cols = list(df_columns)
    df_set = set(df_cols)

    resolved: Dict[str, str] = {}
    for req, candidates in required_to_candidates.items():
        if req in df_set:
            resolved[req] = req
            continue
        hit = None
        for c in candidates:
            if c in df_set:
                hit = c
                break
        if hit is not None:
            resolved[req] = hit
    return resolved


def rename_df_columns_inplace(df, required_to_candidates: Dict[str, List[str]]) -> Dict[str, str]:
    """Rename df columns so that required names exist when possible.

    Returns the resolved mapping required->actual(before rename).
    """
    resolved = resolve_columns(df.columns, required_to_candidates)
    # Build rename mapping actual->required for those where actual != required
    actual_to_required = {}
    for req, actual in resolved.items():
        if actual != req:
            actual_to_required[actual] = req
    if actual_to_required:
        df.rename(columns=actual_to_required, inplace=True)
    return resolved


def missing_required(df_columns: Iterable[str], required: List[str]) -> List[str]:
    cols = set(df_columns)
    return [c for c in required if c not in cols]
