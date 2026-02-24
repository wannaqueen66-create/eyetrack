import pandas as pd
import numpy as np

from src.aoi_metrics import PolygonAOI, compute_metrics


def test_dwell_row_vs_fixation():
    # One fixation appears on two rows with same duration (common exporter behavior)
    df = pd.DataFrame({
        'Recording Time Stamp[ms]': [0, 10, 20, 30],
        'Gaze Point X[px]': [5, 5, 5, 200],
        'Gaze Point Y[px]': [5, 5, 5, 200],
        'Fixation Index': [1, 1, 2, 3],
        'Fixation Duration[ms]': [100, 100, 50, 999],
    })

    aoi = [PolygonAOI('A', 1, [(0, 0), (10, 0), (10, 10), (0, 10)])]

    poly_row, cls_row = compute_metrics(df, aoi, dwell_mode='row')
    poly_fix, cls_fix = compute_metrics(df, aoi, dwell_mode='fixation')

    # In AOI we have rows 0,1,2. Row mode sums 100+100+50=250
    assert int(poly_row.loc[0, 'samples']) == 3
    assert int(poly_row.loc[0, 'fixation_count']) == 2
    assert float(poly_row.loc[0, 'dwell_time_ms']) == 250.0

    # Fixation mode dedups by Fixation Index: max(100) + max(50) = 150
    assert float(poly_fix.loc[0, 'dwell_time_ms']) == 150.0

    # TTFF should be 0 (first AOI hit at t=0)
    assert float(poly_fix.loc[0, 'TTFF_ms']) == 0.0


def test_bbox_prefilter_no_points():
    df = pd.DataFrame({
        'Recording Time Stamp[ms]': [0, 1, 2],
        'Gaze Point X[px]': [100, 101, 102],
        'Gaze Point Y[px]': [100, 101, 102],
        'Fixation Point X[px]': [100, 101, 102],
        'Fixation Point Y[px]': [100, 101, 102],
        'Fixation Index': [1, 2, 3],
        'Fixation Duration[ms]': [10, 20, 30],
    })
    aoi = [PolygonAOI('A', 1, [(0, 0), (10, 0), (10, 10), (0, 10)])]

    poly, cls = compute_metrics(df, aoi, dwell_mode='fixation', point_source='gaze')
    assert int(poly.loc[0, 'samples']) == 0
    assert np.isnan(poly.loc[0, 'dwell_time_ms'])
    assert np.isnan(poly.loc[0, 'TTFF_ms'])

    poly2, cls2 = compute_metrics(df, aoi, dwell_mode='fixation', point_source='fixation')
    assert int(poly2.loc[0, 'samples']) == 0


def test_dwell_empty_as_zero():
    df = pd.DataFrame({
        'Recording Time Stamp[ms]': [0, 1, 2],
        'Gaze Point X[px]': [100, 101, 102],
        'Gaze Point Y[px]': [100, 101, 102],
        'Fixation Index': [1, 2, 3],
        'Fixation Duration[ms]': [10, 20, 30],
    })
    aoi = [PolygonAOI('A', 1, [(0, 0), (10, 0), (10, 10), (0, 10)])]
    poly, cls = compute_metrics(df, aoi, dwell_mode='fixation', dwell_empty_as_zero=True)
    assert int(poly.loc[0, 'visited']) == 0
    assert float(poly.loc[0, 'dwell_time_ms']) == 0.0
