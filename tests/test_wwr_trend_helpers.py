import importlib.util
import pathlib

import pandas as pd


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / 'model_aoi_lmm_allocation.py'
spec = importlib.util.spec_from_file_location('model_aoi_lmm_allocation', SCRIPT_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_wwr_trend_codings_midpoint_is_quadratic_extreme():
    df = pd.DataFrame({'WWR': [15, 45, 75]})
    out = mod._wwr_trend_codings(df)
    assert list(out['WWR']) == [15, 45, 75]
    assert list(out['WWR_linear']) == [-1.0, 0.0, 1.0]
    mid_q = float(out.loc[out['WWR'] == 45, 'WWR_quadratic'].iloc[0])
    edge_q = float(out.loc[out['WWR'] == 15, 'WWR_quadratic'].iloc[0])
    assert mid_q < edge_q


def test_extract_wwr_trend_tests_assigns_direction():
    fixef = pd.DataFrame([
        {'term': 'WWR_linear', 'coef': 0.4, 'se': 0.1, 'p': 0.01, 'ci_low': 0.2, 'ci_high': 0.6, 'abs_test': 4.0},
        {'term': 'WWR_quadratic', 'coef': -0.05, 'se': 0.1, 'p': 0.60, 'ci_low': -0.25, 'ci_high': 0.15, 'abs_test': 0.5},
    ])
    out = mod._extract_wwr_trend_tests(fixef, 'share_pct', 'Experience', 'main_effects', 'Main effects', 'All rows')
    assert set(out['trend_component']) == {'linear', 'quadratic'}
    assert set(out['trend_direction']) == {'linear_increase'}


def test_wwr_trend_shape_plot_smoke(tmp_path):
    coding = pd.DataFrame({
        'model_family': ['main_effects'] * 3,
        'family_title': ['Main effects'] * 3,
        'group_var': ['Experience'] * 3,
        'subset': ['All rows'] * 3,
        'outcome': ['share_pct'] * 3,
        'WWR': [15, 45, 75],
        'WWR_linear': [-1.0, 0.0, 1.0],
        'WWR_quadratic': [0.7, -1.4, 0.7],
        'wwr_order': [1, 2, 3],
        'WWR_label': ['WWR15', 'WWR45', 'WWR75'],
    })
    trend = pd.DataFrame([
        {'trend_component': 'linear', 'trend_component_label': 'Linear trend', 'coef': 0.0, 'p': 0.8, 'ci_low': -0.2, 'ci_high': 0.2, 'trend_direction': 'mid_point_peak'},
        {'trend_component': 'quadratic', 'trend_component_label': 'Quadratic trend', 'coef': -1.0, 'p': 0.01, 'ci_low': -1.5, 'ci_high': -0.5, 'trend_direction': 'mid_point_peak'},
    ])
    out_png = tmp_path / 'evidence_wwr_trend_shape_share_pct.png'
    mod._plot_wwr_trend_shape(coding, trend, out_png, 'Experience', 'share_pct', 'Main effects')
    assert out_png.exists()
    assert (tmp_path / 'evidence_wwr_trend_shape_share_pct_data.csv').exists()
