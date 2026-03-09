import pandas as pd
import numpy as np

from src.aoi_metrics import PolygonAOI, compute_metrics


def test_ttff_segmented_by_video_time_reset():
    df = pd.DataFrame({
        'Recording Time Stamp[ms]': [0, 100, 200, 10000, 10100, 10200],
        'Gaze Point X[px]': [999, 999, 999, 999, 999, 5],
        'Gaze Point Y[px]': [999, 999, 999, 999, 999, 5],
        'Video Time[HH:mm:ss.ms]': ['00:00:00.000', '00:00:00.100', '00:00:00.200', '00:00:00.000', '00:00:00.100', '00:00:00.200'],
        'Time of Day[HH:mm:ss.ms]': ['12:00:00.000', '12:00:00.100', '12:00:00.200', '12:00:10.000', '12:00:10.100', '12:00:10.200'],
        'Fixation Index': [1, 2, 3, 4, 5, 6],
        'Fixation Duration[ms]': [10, 10, 10, 10, 10, 10],
    })
    aoi = [PolygonAOI('A', 1, [(0, 0), (10, 0), (10, 10), (0, 10)])]
    poly, _ = compute_metrics(df, aoi, dwell_mode='fixation')
    assert float(poly.loc[0, 'TTFF']) == 200.0
    assert int(poly.loc[0, 'segment_count']) == 2
    assert 'multi_segment_detected' in str(poly.loc[0, 'ttff_warning'])


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
    assert float(poly_fix.loc[0, 'TTFF']) == 0.0
    assert poly_fix.loc[0, 'ttff_source'] in ('recording_timestamp_segment_start_fallback', 'video_time_segment_start')


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
    assert np.isnan(poly.loc[0, 'TTFF'])

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


def test_ttff_trial_start_ms_override():
    # AOI hit occurs at t=120; if trial_start_ms=100, TTFF should be 20.
    df = pd.DataFrame({
        'Recording Time Stamp[ms]': [100, 110, 120],
        'Gaze Point X[px]': [999, 999, 5],
        'Gaze Point Y[px]': [999, 999, 5],
        'Fixation Index': [1, 2, 3],
        'Fixation Duration[ms]': [10, 10, 10],
    })
    aoi = [PolygonAOI('A', 1, [(0, 0), (10, 0), (10, 10), (0, 10)])]
    poly, cls = compute_metrics(df, aoi, dwell_mode='fixation', trial_start_ms=100)
    assert float(poly.loc[0, 'TTFF']) == 20.0
    assert poly.loc[0, 'ttff_source'] == 'explicit_trial_start_ms'
