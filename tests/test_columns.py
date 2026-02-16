import pandas as pd

from src.columns import load_columns_map, rename_df_columns_inplace


def test_rename_columns_default_map():
    df = pd.DataFrame({
        'timestamp_ms': [0, 1],
        'x': [10, 11],
        'y': [20, 21],
    })
    cmap = load_columns_map(None)
    rename_df_columns_inplace(df, cmap)
    assert 'Recording Time Stamp[ms]' in df.columns
    assert 'Gaze Point X[px]' in df.columns
    assert 'Gaze Point Y[px]' in df.columns
