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


def _bbox_mask(x: np.ndarray, y: np.ndarray, poly: List[Tuple[float, float]]) -> np.ndarray:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def point_in_poly(x: np.ndarray, y: np.ndarray, poly: List[Tuple[float, float]], boundary_eps: float = 1e-6) -> np.ndarray:
    """Vectorized point-in-polygon (even–odd / ray casting) with bbox pre-filter.

    Boundary rule (recommended for papers / reproducibility):
    - Points on polygon edges are treated as INSIDE, using a small tolerance `boundary_eps`.

    Notes:
    - Coordinates are in the same pixel coordinate system as the background image.
    - `boundary_eps` is mainly for numerical robustness; in typical eye-tracking data,
      exact-on-edge points are rare.
    """

    n = len(poly)
    if n < 3:
        return np.zeros_like(x, dtype=bool)

    # bbox prefilter for speed
    bbox = _bbox_mask(x, y, poly)
    inside = np.zeros_like(x, dtype=bool)
    if not np.any(bbox):
        return inside

    px = np.array([p[0] for p in poly], dtype=float)
    py = np.array([p[1] for p in poly], dtype=float)

    # Only compute for candidate points
    xx = x[bbox].astype(float)
    yy = y[bbox].astype(float)

    # 1) Boundary check: consider points on any edge as inside
    on_edge = np.zeros_like(xx, dtype=bool)
    j = n - 1
    for i in range(n):
        x1, y1 = px[j], py[j]
        x2, y2 = px[i], py[i]

        # Segment bbox test (fast)
        minx, maxx = (x1, x2) if x1 <= x2 else (x2, x1)
        miny, maxy = (y1, y2) if y1 <= y2 else (y2, y1)
        seg_bbox = (xx >= (minx - boundary_eps)) & (xx <= (maxx + boundary_eps)) & (yy >= (miny - boundary_eps)) & (yy <= (maxy + boundary_eps))

        # Collinearity via cross product
        dx = x2 - x1
        dy = y2 - y1
        cross = (xx - x1) * dy - (yy - y1) * dx
        col = np.abs(cross) <= boundary_eps

        on_edge |= seg_bbox & col
        j = i

    # 2) Ray casting (even–odd rule)
    ins = np.zeros_like(xx, dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        intersect = ((yi > yy) != (yj > yy)) & (xx < (xj - xi) * (yy - yi) / ((yj - yi) + 1e-12) + xi)
        ins ^= intersect
        j = i

    ins |= on_edge
    inside[bbox] = ins
    return inside


def _dwell_time(sub: pd.DataFrame, mode: str = 'row') -> float:
    """Compute dwell time in ms.

    mode:
      - 'row': sum Fixation Duration[ms] per row (legacy / may double-count)
      - 'fixation': deduplicate by Fixation Index (recommended)
    """
    if sub is None or len(sub) == 0:
        return np.nan

    if mode not in ('row', 'fixation'):
        raise ValueError("mode must be 'row' or 'fixation'")

    if 'Fixation Duration[ms]' not in sub.columns:
        return np.nan

    if mode == 'row' or ('Fixation Index' not in sub.columns):
        d = pd.to_numeric(sub.get('Fixation Duration[ms]'), errors='coerce').dropna().sum()
        return float(d) if pd.notna(d) else np.nan

    # fixation mode
    tmp = sub[['Fixation Index', 'Fixation Duration[ms]']].copy()
    tmp['Fixation Index'] = pd.to_numeric(tmp['Fixation Index'], errors='coerce')
    tmp['Fixation Duration[ms]'] = pd.to_numeric(tmp['Fixation Duration[ms]'], errors='coerce')
    tmp = tmp.dropna(subset=['Fixation Index', 'Fixation Duration[ms]'])
    if len(tmp) == 0:
        return np.nan

    # Use max duration per fixation index (robust if duration repeats across rows)
    per_fix = tmp.groupby('Fixation Index', as_index=False)['Fixation Duration[ms]'].max()
    return float(per_fix['Fixation Duration[ms]'].sum())


def compute_metrics(df: pd.DataFrame, aois: List[PolygonAOI], dwell_mode: str = 'row'):
    required = ['Gaze Point X[px]', 'Gaze Point Y[px]', 'Recording Time Stamp[ms]']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    x = pd.to_numeric(df['Gaze Point X[px]'], errors='coerce').to_numpy()
    y = pd.to_numeric(df['Gaze Point Y[px]'], errors='coerce').to_numpy()
    t = pd.to_numeric(df['Recording Time Stamp[ms]'], errors='coerce').to_numpy()
    if not np.isfinite(t).any():
        t0 = np.nan
    else:
        t0 = np.nanmin(t)

    per_poly_rows = []
    per_class_rows = []

    class_to_masks: Dict[str, List[np.ndarray]] = {}

    finite_xy = np.isfinite(x) & np.isfinite(y)

    for a in aois:
        mask = point_in_poly(x, y, a.points) & finite_xy
        class_to_masks.setdefault(a.class_name, []).append(mask)

        sub = df[mask]
        dwell = _dwell_time(sub, mode=dwell_mode)
        fcount = pd.to_numeric(sub.get('Fixation Index'), errors='coerce').dropna().nunique() if len(sub) else 0
        if len(sub) and pd.notna(t0):
            ttff = (pd.to_numeric(sub.get('Recording Time Stamp[ms]'), errors='coerce').min() - t0)
        else:
            ttff = np.nan

        per_poly_rows.append({
            'class_name': a.class_name,
            'polygon_id': a.polygon_id,
            'samples': int(mask.sum()),
            'dwell_time_ms': float(dwell) if pd.notna(dwell) else np.nan,
            'fixation_count': int(fcount) if pd.notna(fcount) else 0,
            'TTFF_ms': float(ttff) if pd.notna(ttff) else np.nan,
        })

    for cls, masks in class_to_masks.items():
        union = np.logical_or.reduce(masks) if masks else np.zeros_like(x, dtype=bool)
        sub = df[union]
        dwell = _dwell_time(sub, mode=dwell_mode)
        fcount = pd.to_numeric(sub.get('Fixation Index'), errors='coerce').dropna().nunique() if len(sub) else 0
        if len(sub) and pd.notna(t0):
            ttff = (pd.to_numeric(sub.get('Recording Time Stamp[ms]'), errors='coerce').min() - t0)
        else:
            ttff = np.nan
        per_class_rows.append({
            'class_name': cls,
            'polygon_count': len(masks),
            'samples': int(union.sum()),
            'dwell_time_ms': float(dwell) if pd.notna(dwell) else np.nan,
            'fixation_count': int(fcount) if pd.notna(fcount) else 0,
            'TTFF_ms': float(ttff) if pd.notna(ttff) else np.nan,
        })

    return pd.DataFrame(per_poly_rows), pd.DataFrame(per_class_rows)
