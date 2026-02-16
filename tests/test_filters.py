import pandas as pd

from src.filters import filter_by_screen_and_validity


def test_filter_by_screen_bounds():
    df = pd.DataFrame({
        'Gaze Point X[px]': [-1, 0, 5, 11],
        'Gaze Point Y[px]': [0, 0, 5, 5],
    })
    out = filter_by_screen_and_validity(df, screen_w=10, screen_h=10, require_validity=False)
    assert len(out) == 2  # (0,0) and (5,5)


def test_filter_by_validity_only_if_columns_exist():
    df = pd.DataFrame({
        'Gaze Point X[px]': [1, 2],
        'Gaze Point Y[px]': [1, 2],
    })
    out = filter_by_screen_and_validity(df, screen_w=None, screen_h=None, require_validity=True)
    assert len(out) == 2


def test_filter_by_validity_when_present():
    df = pd.DataFrame({
        'Gaze Point X[px]': [1, 2],
        'Gaze Point Y[px]': [1, 2],
        'Validity Left': [1, 0],
        'Validity Right': [1, 1],
    })
    out = filter_by_screen_and_validity(df, require_validity=True)
    assert len(out) == 1
