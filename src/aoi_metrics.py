import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


AOI_CLASS_ALIASES = {
    'pingpong_table': 'table',
    'ping pong table': 'table',
    'table': 'table',
    'tables': 'table',
    'window': 'window',
    'windows': 'window',
    'equipment': 'equipment',
    'apparatus': 'equipment',
    'device': 'equipment',
    'facility': 'equipment',
    'instrument': 'equipment',
}


def normalize_aoi_class_name(name: str) -> str:
    s = '' if name is None else str(name).strip()
    low = s.lower()
    if low in AOI_CLASS_ALIASES:
        return AOI_CLASS_ALIASES[low]
    if 'pingpong_table' in low or low == 'table' or low == 'tables':
        return 'table'
    if 'window' in low:
        return 'window'
    if any(k in low for k in ['equipment', 'apparatus', 'device', 'facility', 'instrument']):
        return 'equipment'
    return low or s


def normalize_aoi_class_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_aoi_class_name)


@dataclass
class PolygonAOI:
    class_name: str
    polygon_id: int
    points: List[Tuple[float, float]]


def load_aoi_json(path: str) -> List[PolygonAOI]:
    """Load AOIs (polygons) from aoi.json."""
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    out = []
    classes = d.get('aoi_classes', {})
    for cls, polys in classes.items():
        cls_norm = normalize_aoi_class_name(cls)
        for i, p in enumerate(polys, start=1):
            pts = p.get('points', [])
            out.append(PolygonAOI(cls_norm, i, [(float(x), float(y)) for x, y in pts]))
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
    """Vectorized point-in-polygon (even–odd / ray casting) with bbox pre-filter."""
    n = len(poly)
    if n < 3:
        return np.zeros_like(x, dtype=bool)

    bbox = _bbox_mask(x, y, poly)
    inside = np.zeros_like(x, dtype=bool)
    if not np.any(bbox):
        return inside

    px = np.array([p[0] for p in poly], dtype=float)
    py = np.array([p[1] for p in poly], dtype=float)

    xx = x[bbox].astype(float)
    yy = y[bbox].astype(float)

    on_edge = np.zeros_like(xx, dtype=bool)
    j = n - 1
    for i in range(n):
        x1, y1 = px[j], py[j]
        x2, y2 = px[i], py[i]
        minx, maxx = (x1, x2) if x1 <= x2 else (x2, x1)
        miny, maxy = (y1, y2) if y1 <= y2 else (y2, y1)
        seg_bbox = (xx >= (minx - boundary_eps)) & (xx <= (maxx + boundary_eps)) & (yy >= (miny - boundary_eps)) & (yy <= (maxy + boundary_eps))
        dx = x2 - x1
        dy = y2 - y1
        cross = (xx - x1) * dy - (yy - y1) * dx
        col = np.abs(cross) <= boundary_eps
        on_edge |= seg_bbox & col
        j = i

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


def _parse_hhmmss_ms_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    dt = pd.to_timedelta(s, errors='coerce')
    return pd.Series(dt.dt.total_seconds() * 1000.0, index=series.index, dtype=float)


def _dwell_time(sub: pd.DataFrame, mode: str = 'row') -> float:
    if sub is None or len(sub) == 0:
        return np.nan
    if mode not in ('row', 'fixation'):
        raise ValueError("mode must be 'row' or 'fixation'")
    if 'Fixation Duration[ms]' not in sub.columns:
        return np.nan
    if mode == 'row' or ('Fixation Index' not in sub.columns):
        d = pd.to_numeric(sub.get('Fixation Duration[ms]'), errors='coerce').dropna().sum()
        return float(d) if pd.notna(d) else np.nan
    tmp = sub[['Fixation Index', 'Fixation Duration[ms]']].copy()
    tmp['Fixation Index'] = pd.to_numeric(tmp['Fixation Index'], errors='coerce')
    tmp['Fixation Duration[ms]'] = pd.to_numeric(tmp['Fixation Duration[ms]'], errors='coerce')
    tmp = tmp.dropna(subset=['Fixation Index', 'Fixation Duration[ms]'])
    if len(tmp) == 0:
        return np.nan
    per_fix = tmp.groupby('Fixation Index', as_index=False)['Fixation Duration[ms]'].max()
    return float(per_fix['Fixation Duration[ms]'].sum())


def _build_fixation_table(sub: pd.DataFrame) -> pd.DataFrame:
    if sub is None or len(sub) == 0 or ('Fixation Index' not in sub.columns):
        return pd.DataFrame(columns=['Fixation Index', 'first_ts', 'first_video_ms', 'segment_id', 'duration_ms'])

    tmp = pd.DataFrame({
        'Fixation Index': pd.to_numeric(sub.get('Fixation Index'), errors='coerce'),
        'first_ts': pd.to_numeric(sub.get('Recording Time Stamp[ms]'), errors='coerce'),
        'first_video_ms': pd.to_numeric(sub.get('Video Time[ms]'), errors='coerce') if 'Video Time[ms]' in sub.columns else np.nan,
        'segment_id': pd.to_numeric(sub.get('ttff_segment_id'), errors='coerce') if 'ttff_segment_id' in sub.columns else np.nan,
        'duration_ms': pd.to_numeric(sub.get('Fixation Duration[ms]'), errors='coerce') if 'Fixation Duration[ms]' in sub.columns else np.nan,
    })
    tmp = tmp.dropna(subset=['Fixation Index'])
    if len(tmp) == 0:
        return pd.DataFrame(columns=['Fixation Index', 'first_ts', 'first_video_ms', 'segment_id', 'duration_ms'])

    agg = tmp.groupby('Fixation Index', as_index=False).agg(
        first_ts=('first_ts', 'min'),
        first_video_ms=('first_video_ms', 'min'),
        segment_id=('segment_id', 'min'),
        duration_ms=('duration_ms', 'max'),
    )
    return agg.sort_values(['first_ts', 'Fixation Index'], na_position='last').reset_index(drop=True)


def _compute_mpd(sub: pd.DataFrame) -> float:
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
    seq = tmp.groupby('Fixation Index', as_index=False).agg(first_ts=('ts', 'min'), in_aoi=('in_aoi', 'max'))
    if len(seq) == 0:
        return 0
    seq = seq.sort_values(['first_ts', 'Fixation Index'], na_position='last')
    entries = 0
    prev = False
    for cur in seq['in_aoi'].astype(bool).tolist():
        if cur and (not prev):
            entries += 1
        prev = cur
    return int(max(entries - 1, 0))


def _derive_ttff_context(df: pd.DataFrame, explicit_trial_start_ms: float | None = None, explicit_trial_start_col: str | None = None, segment_gap_ms: float = 1500.0) -> dict:
    n = len(df)
    rec_ts = pd.to_numeric(df.get('Recording Time Stamp[ms]'), errors='coerce') if 'Recording Time Stamp[ms]' in df.columns else pd.Series(np.nan, index=df.index)
    video_ms = pd.to_numeric(df.get('Video Time[ms]'), errors='coerce') if 'Video Time[ms]' in df.columns else pd.Series(np.nan, index=df.index)
    tod_ms = pd.to_numeric(df.get('Time of Day[ms]'), errors='coerce') if 'Time of Day[ms]' in df.columns else pd.Series(np.nan, index=df.index)

    seg_break = pd.Series(False, index=df.index, dtype=bool)
    reasons = []
    if n:
        seg_break.iloc[0] = True

    def _apply_break(source: pd.Series, reset_label: str, gap_label: str):
        nonlocal seg_break, reasons
        if source.isna().all() or len(source) < 2:
            return 0, 0
        dif = source.diff()
        reset = (dif < -1e-9).fillna(False)
        gap = (dif > float(segment_gap_ms)).fillna(False)
        if len(reset):
            reset.iloc[0] = False
        if len(gap):
            gap.iloc[0] = False
        seg_break |= reset | gap
        reasons.append((reset_label, reset))
        reasons.append((gap_label, gap))
        return int(reset.sum()), int(gap.sum())

    video_resets, video_gaps = _apply_break(video_ms, 'video_reset', 'video_gap')
    tod_resets, tod_gaps = _apply_break(tod_ms, 'timeofday_reset', 'timeofday_gap')

    segment_id = seg_break.cumsum().astype(int) - 1 if n else pd.Series(dtype=int)
    df_work = pd.DataFrame({
        'segment_id': segment_id,
        'rec_ts': rec_ts,
        'video_ms': video_ms,
        'tod_ms': tod_ms,
    })

    segments = []
    for sid, g in df_work.groupby('segment_id', dropna=False):
        segments.append({
            'segment_id': int(sid),
            'row_count': int(len(g)),
            'recording_start_ms': float(g['rec_ts'].dropna().min()) if g['rec_ts'].notna().any() else None,
            'video_start_ms': float(g['video_ms'].dropna().min()) if g['video_ms'].notna().any() else None,
            'timeofday_start_ms': float(g['tod_ms'].dropna().min()) if g['tod_ms'].notna().any() else None,
        })

    if explicit_trial_start_ms is not None:
        ttff_source = 'explicit_trial_start_ms'
        baseline_by_segment = {s['segment_id']: float(explicit_trial_start_ms) for s in segments}
    elif explicit_trial_start_col is not None and explicit_trial_start_col in df.columns:
        alt = pd.to_numeric(df[explicit_trial_start_col], errors='coerce')
        baseline_by_segment = {}
        for sid, g in pd.DataFrame({'segment_id': segment_id, 'alt': alt}).groupby('segment_id', dropna=False):
            baseline_by_segment[int(sid)] = float(g['alt'].dropna().min()) if g['alt'].notna().any() else np.nan
        ttff_source = f'explicit_trial_start_col:{explicit_trial_start_col}'
    elif video_ms.notna().any():
        baseline_by_segment = {s['segment_id']: (float(s['video_start_ms']) if s['video_start_ms'] is not None else np.nan) for s in segments}
        ttff_source = 'video_time_segment_start'
    elif rec_ts.notna().any():
        baseline_by_segment = {s['segment_id']: (float(s['recording_start_ms']) if s['recording_start_ms'] is not None else np.nan) for s in segments}
        ttff_source = 'recording_timestamp_segment_start_fallback'
    else:
        baseline_by_segment = {s['segment_id']: np.nan for s in segments}
        ttff_source = 'unavailable'

    warnings = []
    if not video_ms.notna().any():
        warnings.append('missing_video_time')
    if not tod_ms.notna().any():
        warnings.append('missing_time_of_day')
    if len(segments) > 1:
        warnings.append('multi_segment_detected')
    if video_resets:
        warnings.append('video_reset_detected')
    if video_gaps:
        warnings.append('video_gap_detected')
    if tod_resets:
        warnings.append('timeofday_reset_detected')
    if tod_gaps:
        warnings.append('timeofday_gap_detected')

    qc_status = 'ok'
    if ttff_source == 'recording_timestamp_segment_start_fallback':
        qc_status = 'warning'
    if ttff_source == 'unavailable':
        qc_status = 'error'
    if 'multi_segment_detected' in warnings:
        qc_status = 'warning' if qc_status == 'ok' else qc_status

    return {
        'segment_id': segment_id,
        'segment_count': int(len(segments)),
        'segments': segments,
        'ttff_source': ttff_source,
        'ttff_baseline_by_segment': baseline_by_segment,
        'ttff_warning': ';'.join(warnings) if warnings else '',
        'ttff_qc_status': qc_status,
        'video_reset_count': int(video_resets),
        'video_gap_count': int(video_gaps),
        'timeofday_reset_count': int(tod_resets),
        'timeofday_gap_count': int(tod_gaps),
        'segment_gap_ms': float(segment_gap_ms),
    }


def _metric_pack(df: pd.DataFrame, sub: pd.DataFrame, mask: np.ndarray, ttff_ctx: dict, dwell_mode: str, dwell_empty_as_zero: bool) -> dict:
    samples = int(mask.sum())
    visited = int(samples > 0)
    tfd = _dwell_time(sub, mode=dwell_mode)
    if (not np.isfinite(tfd)) and dwell_empty_as_zero and visited == 0:
        tfd = 0.0

    fix_tbl = _build_fixation_table(sub)
    fc = int(len(fix_tbl))
    ttff = np.nan
    ttff_segment_id = np.nan
    ttff_segment_start_video_ms = np.nan
    ttff_segment_start_timeofday_ms = np.nan

    if len(fix_tbl):
        first_row = fix_tbl.loc[fix_tbl['first_ts'].idxmin()]
        ttff_segment_id = first_row.get('segment_id', np.nan)
        if pd.notna(ttff_segment_id):
            sid = int(ttff_segment_id)
            baseline = ttff_ctx.get('ttff_baseline_by_segment', {}).get(sid, np.nan)
            ttff_segment_start_video_ms = next((s.get('video_start_ms') for s in ttff_ctx.get('segments', []) if s.get('segment_id') == sid), np.nan)
            ttff_segment_start_timeofday_ms = next((s.get('timeofday_start_ms') for s in ttff_ctx.get('segments', []) if s.get('segment_id') == sid), np.nan)
            if pd.notna(first_row.get('first_video_ms')) and pd.notna(baseline):
                ttff = float(first_row['first_video_ms'] - baseline)
            elif pd.notna(first_row.get('first_ts')) and pd.notna(baseline):
                ttff = float(first_row['first_ts'] - baseline)
        elif pd.notna(first_row.get('first_ts')):
            ttff = np.nan

    if len(fix_tbl):
        first_fix = fix_tbl.loc[fix_tbl['first_ts'].idxmin()]
        ffd = float(first_fix['duration_ms']) if pd.notna(first_fix['duration_ms']) else np.nan
        mfd = float(fix_tbl['duration_ms'].mean()) if fix_tbl['duration_ms'].notna().any() else np.nan
    else:
        ffd = np.nan
        mfd = np.nan

    rff = _compute_return_fixations(df, mask)
    mpd = _compute_mpd(sub)

    out = {
        'samples': samples,
        'visited': visited,
        'FC': fc,
        'TTFF': float(ttff) if pd.notna(ttff) else np.nan,
        'FFD': float(ffd) if pd.notna(ffd) else np.nan,
        'TFD': float(tfd) if pd.notna(tfd) else np.nan,
        'MFD': float(mfd) if pd.notna(mfd) else np.nan,
        'RFF': int(rff),
        'MPD': float(mpd) if pd.notna(mpd) else np.nan,
        'ttff_source': ttff_ctx.get('ttff_source'),
        'segment_count': int(ttff_ctx.get('segment_count', 0)),
        'ttff_segment_id': int(ttff_segment_id) if pd.notna(ttff_segment_id) else np.nan,
        'ttff_segment_start_video_ms': float(ttff_segment_start_video_ms) if pd.notna(ttff_segment_start_video_ms) else np.nan,
        'ttff_segment_start_timeofday_ms': float(ttff_segment_start_timeofday_ms) if pd.notna(ttff_segment_start_timeofday_ms) else np.nan,
        'ttff_warning': ttff_ctx.get('ttff_warning', ''),
        'ttff_qc_status': ttff_ctx.get('ttff_qc_status', 'ok'),
        'video_reset_count': int(ttff_ctx.get('video_reset_count', 0)),
        'video_gap_count': int(ttff_ctx.get('video_gap_count', 0)),
        'timeofday_reset_count': int(ttff_ctx.get('timeofday_reset_count', 0)),
        'timeofday_gap_count': int(ttff_ctx.get('timeofday_gap_count', 0)),
    }

    out['Fixation Count'] = out['FC']
    out['Time to First Fixation'] = out['TTFF']
    out['First Fixation Duration'] = out['FFD']
    out['Total Fixation Duration'] = out['TFD']
    out['Mean Fixation Duration'] = out['MFD']
    out['Re-fixation Frequency'] = out['RFF']
    out['Mean Pupil Diameter'] = out['MPD']
    out['fixation_count'] = out['FC']
    out['dwell_time_ms'] = out['TFD']
    out['RF'] = out['RFF']
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
    ttff_segment_gap_ms: float = 1500.0,
):
    """Compute AOI metrics with segment-aware TTFF based primarily on Video Time."""
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

    work = df.copy()
    if 'Video Time[HH:mm:ss.ms]' in work.columns and 'Video Time[ms]' not in work.columns:
        work['Video Time[ms]'] = _parse_hhmmss_ms_series(work['Video Time[HH:mm:ss.ms]'])
    if 'Time of Day[HH:mm:ss.ms]' in work.columns and 'Time of Day[ms]' not in work.columns:
        work['Time of Day[ms]'] = _parse_hhmmss_ms_series(work['Time of Day[HH:mm:ss.ms]'])

    ttff_ctx = _derive_ttff_context(
        work,
        explicit_trial_start_ms=trial_start_ms,
        explicit_trial_start_col=trial_start_col,
        segment_gap_ms=ttff_segment_gap_ms,
    )
    work['ttff_segment_id'] = ttff_ctx['segment_id']

    per_poly_rows = []
    per_class_rows = []
    class_to_masks: Dict[str, List[np.ndarray]] = {}
    finite_xy = np.isfinite(x) & np.isfinite(y)

    for a in aois:
        mask = point_in_poly(x, y, a.points) & finite_xy
        class_to_masks.setdefault(a.class_name, []).append(mask)
        sub = work[mask]
        row = {'class_name': a.class_name, 'polygon_id': a.polygon_id, **_metric_pack(work, sub, mask, ttff_ctx, dwell_mode, dwell_empty_as_zero)}
        per_poly_rows.append(row)

    for cls, masks in class_to_masks.items():
        union = np.logical_or.reduce(masks) if masks else np.zeros_like(x, dtype=bool)
        sub = work[union]
        row = {'class_name': cls, 'polygon_count': len(masks), **_metric_pack(work, sub, union, ttff_ctx, dwell_mode, dwell_empty_as_zero)}
        per_class_rows.append(row)

    poly_df = pd.DataFrame(per_poly_rows)
    class_df = pd.DataFrame(per_class_rows)

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
                        'class_a': ci,
                        'class_b': cj,
                        'overlap_samples': cnt,
                        'samples_a': cnt_i,
                        'samples_b': cnt_j,
                        'overlap_ratio_a': (cnt / cnt_i) if cnt_i > 0 else None,
                        'overlap_ratio_b': (cnt / cnt_j) if cnt_j > 0 else None,
                    })
        if overlap_info:
            top = sorted(overlap_info, key=lambda d: d['overlap_samples'], reverse=True)[:10]
            print('[WARN] AOI class overlap detected (a point can belong to multiple classes). Top overlaps:')
            for d in top:
                print(f"  - {d['class_a']} × {d['class_b']}: overlap_samples={d['overlap_samples']}")

    diag = {
        'point_source': point_source,
        'trial_start_ms': float(trial_start_ms) if trial_start_ms is not None else None,
        'trial_start_col': trial_start_col,
        'warn_class_overlap': bool(warn_class_overlap),
        'class_overlap': overlap_info,
        'rff_definition': 'RFF = number of AOI re-entry episodes after first entry (based on fixation sequence).',
        'metric_naming': {
            'FC': 'Fixation Count',
            'TTFF': 'Time to First Fixation',
            'FFD': 'First Fixation Duration',
            'MFD': 'Mean Fixation Duration',
            'MPD': 'Mean Pupil Diameter',
            'RFF': 'Re-fixation Frequency',
            'TFD': 'Total Fixation Duration',
        },
        'ttff_logic': {
            'source': ttff_ctx.get('ttff_source'),
            'segment_count': ttff_ctx.get('segment_count'),
            'segment_gap_ms': ttff_ctx.get('segment_gap_ms'),
            'warnings': ttff_ctx.get('ttff_warning', '').split(';') if ttff_ctx.get('ttff_warning') else [],
            'qc_status': ttff_ctx.get('ttff_qc_status'),
            'segments': ttff_ctx.get('segments', []),
            'video_reset_count': ttff_ctx.get('video_reset_count', 0),
            'video_gap_count': ttff_ctx.get('video_gap_count', 0),
            'timeofday_reset_count': ttff_ctx.get('timeofday_reset_count', 0),
            'timeofday_gap_count': ttff_ctx.get('timeofday_gap_count', 0),
        },
    }

    poly_df.attrs['diagnostics'] = diag
    class_df.attrs['diagnostics'] = diag
    return poly_df, class_df
