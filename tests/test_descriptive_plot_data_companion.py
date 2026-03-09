import importlib.util
import pathlib

import pandas as pd


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / 'summarize_aoi_by_condition_group.py'
spec = importlib.util.spec_from_file_location('summarize_aoi_by_condition_group', SCRIPT_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_plot_grid_writes_data_companion(tmp_path):
    df = pd.DataFrame([
        {'group_value': 'Low', 'class_name': 'table', 'WWR': 15, 'Complexity': 'C0', 'mean': 10.0, 'ci_low': 9.0, 'ci_high': 11.0},
        {'group_value': 'Low', 'class_name': 'table', 'WWR': 45, 'Complexity': 'C0', 'mean': 12.0, 'ci_low': 11.0, 'ci_high': 13.0},
        {'group_value': 'Low', 'class_name': 'table', 'WWR': 75, 'Complexity': 'C0', 'mean': 13.0, 'ci_low': 12.0, 'ci_high': 14.0},
        {'group_value': 'Low', 'class_name': 'table', 'WWR': 15, 'Complexity': 'C1', 'mean': 14.0, 'ci_low': 13.0, 'ci_high': 15.0},
        {'group_value': 'Low', 'class_name': 'table', 'WWR': 45, 'Complexity': 'C1', 'mean': 15.0, 'ci_low': 14.0, 'ci_high': 16.0},
        {'group_value': 'Low', 'class_name': 'table', 'WWR': 75, 'Complexity': 'C1', 'mean': 16.0, 'ci_low': 15.0, 'ci_high': 17.0},
        {'group_value': 'High', 'class_name': 'table', 'WWR': 15, 'Complexity': 'C0', 'mean': 11.0, 'ci_low': 10.0, 'ci_high': 12.0},
        {'group_value': 'High', 'class_name': 'table', 'WWR': 45, 'Complexity': 'C0', 'mean': 12.5, 'ci_low': 11.5, 'ci_high': 13.5},
        {'group_value': 'High', 'class_name': 'table', 'WWR': 75, 'Complexity': 'C0', 'mean': 13.5, 'ci_low': 12.5, 'ci_high': 14.5},
        {'group_value': 'High', 'class_name': 'table', 'WWR': 15, 'Complexity': 'C1', 'mean': 15.0, 'ci_low': 14.0, 'ci_high': 16.0},
        {'group_value': 'High', 'class_name': 'table', 'WWR': 45, 'Complexity': 'C1', 'mean': 16.0, 'ci_low': 15.0, 'ci_high': 17.0},
        {'group_value': 'High', 'class_name': 'table', 'WWR': 75, 'Complexity': 'C1', 'mean': 17.0, 'ci_low': 16.0, 'ci_high': 18.0},
    ])
    out_png = tmp_path / 'plot_Experience_TFD.png'
    mod.plot_grid(df, out_png, outcome='TFD', group_var='Experience', title='Smoke')
    assert out_png.exists()
    data_csv = tmp_path / 'plot_Experience_TFD_data.csv'
    assert data_csv.exists()
    saved = pd.read_csv(data_csv, encoding='utf-8-sig')
    assert 'value_label' in saved.columns
    assert len(saved) == len(df)
