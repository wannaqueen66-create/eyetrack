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
    """Load AOIs (polygons) from aoi.json.

    This function returns only polygon definitions for backward compatibility.
    Use `load_aoi_json_meta()` if you also need image/tool metadata.
    """
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    out = []
    classes = d.get('aoi_classes', {})
    for cls, polys in classes.items():
        for i, p in enumerate(polys, start=1):
            pts = p.get('points', [])
            out.append(PolygonAOI(cls, i, [(float(x), float(y)) for x, y in pts]))
    return out


def load_aoi_json_meta(path: str) -> dict:
    """Load metadata from aoi.json (image width/height, tool info, etc.)."""
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    meta = {}
    if isinstance(d, dict):
        if isinstance(d.get('tool'), dict):
            meta['tool'] = d.get('tool')
        if isinstance(d.get('image'), dict):
            meta['image'] = d.get('image')
    return meta


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


def _build_fixation_table(sub: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-fixation table within an AOI subset.

    Columns in return (when available):
      - Fixation Index
      - first_ts
      - duration_ms
    """
    if sub is None or len(sub) == 0 or ('Fixation Index' not in sub.columns):
        return pd.DataFrame(columns=['Fixation Index', 'first_ts', 'duration_ms'])

    tmp = pd.DataFrame({
        'Fixation Index': pd.to_numeric(sub.get('Fixation Index'), errors='coerce'),
        'first_ts': pd.to_numeric(sub.get('Recording Time Stamp[ms]'), errors='coerce'),
        'duration_ms': pd.to_numeric(sub.get('Fixation Duration[ms]'), errors='coerce') if 'Fixation Duration[ms]' in sub.columns else np.nan,
    })
    tmp = tmp.dropna(subset=['Fixation Index'])
    if len(tmp) == 0:
        return pd.DataFrame(columns=['Fixation Index', 'first_ts', 'duration_ms'])

    agg = tmp.groupby('Fixation Index', as_index=False).agg(
        first_ts=('first_ts', 'min'),
        duration_ms=('duration_ms', 'max'),
    )
    return agg.sort_values(['first_ts', 'Fixation Index'], na_position='last').reset_index(drop=True)


def _compute_mpd(sub: pd.DataFrame) -> float:
    """Mean pupil diameter in AOI subset.

    Preference: mm columns > px columns.
    If both eyes exist, use row-wise mean of left/right then average across rows.
    """
    if sub is None or len(sub) == 0:
        return np.nan

    left = right = None
    if 'Pupil Diameter Left[mm]' in sub.columns or 'Pupil Diameter Right[mm]' in sub.columns:
        left = pd.to_numeric(sub.get('Pupil Diameter Left[mm]'), errors='coerce') if 'Pupil Diameter Left[mm]' in sub.columns else None
        right = pd.to_numeric(sub.get('Pupil Diameter Right[mm]'), errors='coerce') if 'Pupil Diameter Right[mm]' in sub.columns else None
    elif 'Pupil Diameter Left[px]' in sub.columns or 'Pupil Diameter Right[px]' in sub.columns:
        left = pd.to_numeric(sub.get('Pupil Diameter Left[px]'), errors='coerce') if 'Pupil Diameter Left[px]' in sub.columns else None
        right = pd.to_numeric(sub.get('Pupil Diameter Right[px]'), errors='coerce') if 'Pupil Diameter Right[px]' in sub.columns else None

    if left is None and right is None:
        return np.nan
    if left is None:
        return float(right.dropna().mean()) if right.notna().any() else np.nan
    if right is None:
        return float(left.dropna().mean()) if left.notna().any() else np.nan

    pair = pd.concat([left.rename('L'), right.rename('R')], axis=1)
    row_mean = pair.mean(axis=1, skipna=True)
    return float(row_mean.dropna().mean()) if row_mean.notna().any() else np.nan


def _compute_return_fixations(df: pd.DataFrame, mask: np.ndarray) -> int:
    """Return fixations (RF): number of re-entries after first entry.

    Operational definition in this repo:
    - Build fixation sequence ordered by fixation first timestamp.
    - Mark each fixation as in-AOI if any row of that fixation falls in AOI.
    - Count AOI entry episodes (out->in transitions).
    - RF = max(entry_episodes - 1, 0).

    This is robust and auditable, and aligns with the common "refixation / revisit"
    interpretation at AOI level.
    """
    if 'Fixation Index' not in df.columns or len(df) == 0:
        return 0

    tmp = pd.DataFrame({
        'Fixation Index': pd.to_numeric(df.get('Fixation Index'), errors='coerce'),
        'ts': pd.to_numeric(df.get('Recording Time Stamp[ms]'), errors='coerce'),
        'in_aoi': np.asarray(mask, dtype=bool),
    })
    tmp = tmp.dropna(subset=['Fixation Index'])
    if len(tmp) == 0:
        return 0

    seq = tmp.groupby('Fixation Index', as_index=False).agg(
        first_ts=('ts', 'min'),
        in_aoi=('in_aoi', 'max'),
    )
    if len(seq) == 0:
        return 0

    seq = seq.sort_values(['first_ts', 'Fixation Index'], na_position='last')
    in_flags = seq['in_aoi'].astype(bool).tolist()

    entries = 0
    prev = False
    for cur in in_flags:
        if cur and (not prev):
            entries += 1
        prev = cur

    return int(max(entries - 1, 0))


def _metric_pack(df: pd.DataFrame, sub: pd.DataFrame, mask: np.ndarray, t0: float, dwell_mode: str, dwell_empty_as_zero: bool) -> dict:
    """Compute AOI metric pack for one mask (polygon or class union)."""
    samples = int(mask.sum())
    visited = int(samples > 0)

    # TFD (legacy alias: dwell_time_ms)
    tfd = _dwell_time(sub, mode=dwell_mode)
    if (not np.isfinite(tfd)) and dwell_empty_as_zero and visited == 0:
        tfd = 0.0

    fix_tbl = _build_fixation_table(sub)
    fc = int(len(fix_tbl))

    if len(fix_tbl) and pd.notna(t0):
        ttff = float(fix_tbl['first_ts'].min() - t0)
    elif len(sub) and pd.notna(t0):
        # Fallback when fixation columns are missing: first sample timestamp
        ttff = float(pd.to_numeric(sub.get('Recording Time Stamp[ms]'), errors='coerce').min() - t0)
    else:
        ttff = np.nan

    if len(fix_tbl):
        first_row = fix_tbl.loc[fix_tbl['first_ts'].idxmin()]
        ffd = float(first_row['duration_ms']) if pd.notna(first_row['duration_ms']) else np.nan
        mfd = float(fix_tbl['duration_ms'].max()) if fix_tbl['duration_ms'].notna().any() else np.nan
    else:
        ffd = np.nan
        mfd = np.nan

    rf = _compute_return_fixations(df, mask)
    mpd = _compute_mpd(sub)

    out = {
        'samples': samples,
        'visited': visited,
        'FC': fc,
        'TTFF': float(ttff) if pd.notna(ttff) else np.nan,
        'FFD': float(ffd) if pd.notna(ffd) else np.nan,
        'TFD': float(tfd) if pd.notna(tfd) else np.nan,
        'MFD': float(mfd) if pd.notna(mfd) else np.nan,
        'RF': int(rf),
        'MPD': float(mpd) if pd.notna(mpd) else np.nan,
    }

    # Backward-compatible aliases
    out['fixation_count'] = out['FC']
    out['TTFF_ms'] = out['TTFF']
    out['dwell_time_ms'] = out['TFD']

    return out


def compute_metrics(
    df: pd.DataFrame,
    aois: List[PolygonAOI],
    dwell_mode: str = 'row',
    point_source: str = 'gaze',
    dwell_empty_as_zero: bool = False,
    trial_start_ms: float | None = None,
    trial_start_col: str | None = None,
    warn_class_overlap: bool = True,
):
    """Compute AOI metrics.

    point_source:
      - 'gaze' (default): AOI hit testing uses Gaze Point X/Y
      - 'fixation': AOI hit testing uses Fixation Point X/Y (recommended when metrics are fixation-based)

    dwell_empty_as_zero:
      - If True, return 0.0 for TFD when visited==0, instead of NaN.
        (TTFF remains NaN; FC remains 0.)
    """

    if point_source not in ('gaze', 'fixation'):
        raise ValueError("point_source must be 'gaze' or 'fixation'")

    required = ['Recording Time Stamp[ms]']
    if point_source == 'gaze':
        required += ['Gaze Point X[px]', 'Gaze Point Y[px]']
    else:
        required += ['Fixation Point X[px]', 'Fixation Point Y[px]']

    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if point_source == 'gaze':
        x = pd.to_numeric(df['Gaze Point X[px]'], errors='coerce').to_numpy()
        y = pd.to_numeric(df['Gaze Point Y[px]'], errors='coerce').to_numpy()
    else:
        x = pd.to_numeric(df['Fixation Point X[px]'], errors='coerce').to_numpy()
        y = pd.to_numeric(df['Fixation Point Y[px]'], errors='coerce').to_numpy()

    t = pd.to_numeric(df['Recording Time Stamp[ms]'], errors='coerce').to_numpy()
    # Determine t0 (trial start)
    # Priority: explicit trial_start_ms > trial_start_col > min(Recording Time Stamp)
    if trial_start_ms is not None:
        t0 = float(trial_start_ms)
    elif trial_start_col is not None and (trial_start_col in df.columns):
        t_alt = pd.to_numeric(df[trial_start_col], errors='coerce').to_numpy()
        t0 = np.nanmin(t_alt) if np.isfinite(t_alt).any() else np.nan
    else:
        t0 = np.nanmin(t) if np.isfinite(t).any() else np.nan

    per_poly_rows = []
    per_class_rows = []

    class_to_masks: Dict[str, List[np.ndarray]] = {}

    finite_xy = np.isfinite(x) & np.isfinite(y)

    for a in aois:
        mask = point_in_poly(x, y, a.points) & finite_xy
        class_to_masks.setdefault(a.class_name, []).append(mask)

        sub = df[mask]
        row = {
            'class_name': a.class_name,
            'polygon_id': a.polygon_id,
            **_metric_pack(df, sub, mask, t0, dwell_mode, dwell_empty_as_zero),
        }
        per_poly_rows.append(row)

    for cls, masks in class_to_masks.items():
        union = np.logical_or.reduce(masks) if masks else np.zeros_like(x, dtype=bool)
        sub = df[union]
        row = {
            'class_name': cls,
            'polygon_count': len(masks),
            **_metric_pack(df, sub, union, t0, dwell_mode, dwell_empty_as_zero),
        }
        per_class_rows.append(row)

    poly_df = pd.DataFrame(per_poly_rows)
    class_df = pd.DataFrame(per_class_rows)

    # Diagnostics: class overlap counts (based on union masks)
    overlap_info = []
    if warn_class_overlap and len(class_to_masks) >= 2:
        classes = list(class_to_masks.keys())
        unions = {c: (np.logical_or.reduce(class_to_masks[c]) if class_to_masks[c] else np.zeros_like(x, dtype=bool)) for c in classes}
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                ci, cj = classes[i], classes[j]
                mask_i = unions[ci]
                mask_j = unions[cj]
                cnt = int(np.logical_and(mask_i, mask_j).sum())
                if cnt > 0:
                    cnt_i = int(mask_i.sum())
                    cnt_j = int(mask_j.sum())
                    overlap_info.append({
                        "class_a": ci,
                        "class_b": cj,
                        "overlap_samples": cnt,
                        "samples_a": cnt_i,
                        "samples_b": cnt_j,
                        "overlap_ratio_a": (cnt / cnt_i) if cnt_i > 0 else None,
                        "overlap_ratio_b": (cnt / cnt_j) if cnt_j > 0 else None,
                    })
        if overlap_info:
            top = sorted(overlap_info, key=lambda d: d["overlap_samples"], reverse=True)[:10]
            print("[WARN] AOI class overlap detected (a point can belong to multiple classes). Top overlaps:")
            for d in top:
                print(f"  - {d['class_a']} × {d['class_b']}: overlap_samples={d['overlap_samples']}")

    diag = {
        "t0_ms": float(t0) if pd.notna(t0) else None,
        "point_source": point_source,
        "trial_start_ms": float(trial_start_ms) if trial_start_ms is not None else None,
        "trial_start_col": trial_start_col,
        "warn_class_overlap": bool(warn_class_overlap),
        "class_overlap": overlap_info,
        "rf_definition": "RF = number of AOI re-entry episodes after first entry (based on fixation sequence).",
    }

    # attach to DataFrame attrs so callers can export to run_config.json
    poly_df.attrs["diagnostics"] = diag
    class_df.attrs["diagnostics"] = diag

    return poly_df, class_df
