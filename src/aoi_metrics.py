import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class PolygonAOI:
    class_name: str
    polygon_id: int
    points: List[Tuple[float, float]]


def load_aoi_json(path: str) -> List[PolygonAOI]:
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    out = []
    classes = d.get('aoi_classes', {})
    for cls, polys in classes.items():
        for i, p in enumerate(polys, start=1):
            pts = p.get('points', [])
            out.append(PolygonAOI(cls, i, [(float(x), float(y)) for x, y in pts]))
    return out


def point_in_poly(x: np.ndarray, y: np.ndarray, poly: List[Tuple[float, float]]) -> np.ndarray:
    # ray casting
    n = len(poly)
    inside = np.zeros_like(x, dtype=bool)
    px = np.array([p[0] for p in poly])
    py = np.array([p[1] for p in poly])
    j = n - 1
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        intersect = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi)
        inside ^= intersect
        j = i
    return inside


def compute_metrics(df: pd.DataFrame, aois: List[PolygonAOI]):
    x = pd.to_numeric(df['Gaze Point X[px]'], errors='coerce').to_numpy()
    y = pd.to_numeric(df['Gaze Point Y[px]'], errors='coerce').to_numpy()
    t = pd.to_numeric(df['Recording Time Stamp[ms]'], errors='coerce').to_numpy()
    fix_idx = pd.to_numeric(df.get('Fixation Index'), errors='coerce')
    fix_dur = pd.to_numeric(df.get('Fixation Duration[ms]'), errors='coerce')
    t0 = np.nanmin(t)

    per_poly_rows = []
    per_class_rows = []

    class_to_masks: Dict[str, List[np.ndarray]] = {}

    for a in aois:
        mask = point_in_poly(x, y, a.points) & np.isfinite(x) & np.isfinite(y)
        class_to_masks.setdefault(a.class_name, []).append(mask)

        sub = df[mask]
        dwell = pd.to_numeric(sub.get('Fixation Duration[ms]'), errors='coerce').dropna().sum()
        fcount = pd.to_numeric(sub.get('Fixation Index'), errors='coerce').dropna().nunique()
        ttff = (pd.to_numeric(sub.get('Recording Time Stamp[ms]'), errors='coerce').min() - t0) if len(sub) else np.nan

        per_poly_rows.append({
            'class_name': a.class_name,
            'polygon_id': a.polygon_id,
            'samples': int(mask.sum()),
            'dwell_time_ms': float(dwell) if pd.notna(dwell) else np.nan,
            'fixation_count': int(fcount) if pd.notna(fcount) else 0,
            'TTFF_ms': float(ttff) if pd.notna(ttff) else np.nan,
        })

    for cls, masks in class_to_masks.items():
        union = np.logical_or.reduce(masks)
        sub = df[union]
        dwell = pd.to_numeric(sub.get('Fixation Duration[ms]'), errors='coerce').dropna().sum()
        fcount = pd.to_numeric(sub.get('Fixation Index'), errors='coerce').dropna().nunique()
        ttff = (pd.to_numeric(sub.get('Recording Time Stamp[ms]'), errors='coerce').min() - t0) if len(sub) else np.nan
        per_class_rows.append({
            'class_name': cls,
            'polygon_count': len(masks),
            'samples': int(union.sum()),
            'dwell_time_ms': float(dwell) if pd.notna(dwell) else np.nan,
            'fixation_count': int(fcount) if pd.notna(fcount) else 0,
            'TTFF_ms': float(ttff) if pd.notna(ttff) else np.nan,
        })

    return pd.DataFrame(per_poly_rows), pd.DataFrame(per_class_rows)
