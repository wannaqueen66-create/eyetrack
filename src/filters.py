from typing import Optional

import pandas as pd


def filter_by_screen_and_validity(
    df: pd.DataFrame,
    screen_w: Optional[int] = None,
    screen_h: Optional[int] = None,
    require_validity: bool = False,
) -> pd.DataFrame:
    """Optionally filter gaze rows by screen bounds and validity flags.

    - If screen_w/screen_h are provided, keeps rows with gaze coords within [0, w] and [0, h].
    - If require_validity is True and validity columns exist, enforces Validity Left/Right == 1.

    Returns a (possibly) filtered copy.
    """
    out = df

    if (screen_w is not None) and (screen_h is not None):
        if ('Gaze Point X[px]' in out.columns) and ('Gaze Point Y[px]' in out.columns):
            x = pd.to_numeric(out['Gaze Point X[px]'], errors='coerce')
            y = pd.to_numeric(out['Gaze Point Y[px]'], errors='coerce')
            out = out[x.between(0, screen_w) & y.between(0, screen_h)].copy()

    if require_validity and ('Validity Left' in out.columns) and ('Validity Right' in out.columns):
        out = out[(out['Validity Left'] == 1) & (out['Validity Right'] == 1)].copy()

    return out
