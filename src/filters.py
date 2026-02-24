from typing import Optional

import pandas as pd


def compute_valid_mask(
    df: pd.DataFrame,
    screen_w: Optional[int] = None,
    screen_h: Optional[int] = None,
    require_validity: bool = False,
    x_col: str = 'Gaze Point X[px]',
    y_col: str = 'Gaze Point Y[px]',
) -> pd.Series:
    """Compute a boolean mask of 'valid' rows.

    Valid rows are defined as:
    - within screen bounds (if screen_w/screen_h are provided AND x_col/y_col exist)
    - AND pass Validity Left/Right == 1 (if require_validity=True AND those columns exist)

    This helper makes trial-level tracking-rate / valid-ratio auditing possible.
    """
    if len(df) == 0:
        return pd.Series([], dtype=bool)

    mask = pd.Series(True, index=df.index)

    if (screen_w is not None) and (screen_h is not None) and (x_col in df.columns) and (y_col in df.columns):
        x = pd.to_numeric(df[x_col], errors='coerce')
        y = pd.to_numeric(df[y_col], errors='coerce')
        mask &= x.between(0, screen_w) & y.between(0, screen_h)

    if require_validity and ('Validity Left' in df.columns) and ('Validity Right' in df.columns):
        mask &= (df['Validity Left'] == 1) & (df['Validity Right'] == 1)

    return mask


def filter_by_screen_and_validity(
    df: pd.DataFrame,
    screen_w: Optional[int] = None,
    screen_h: Optional[int] = None,
    require_validity: bool = False,
    x_col: str = 'Gaze Point X[px]',
    y_col: str = 'Gaze Point Y[px]',
) -> pd.DataFrame:
    """Filter rows by screen bounds and validity flags.

    Uses the same rules as `compute_valid_mask()`.
    """
    mask = compute_valid_mask(
        df,
        screen_w=screen_w,
        screen_h=screen_h,
        require_validity=require_validity,
        x_col=x_col,
        y_col=y_col,
    )
    return df[mask].copy()
