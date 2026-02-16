import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .columns import load_columns_map, rename_df_columns_inplace, missing_required


def load_and_clean(
    csv_path: str,
    screen_w: int = 1280,
    screen_h: int = 1440,
    require_validity: bool = True,
    columns_map_path: str = None,
):
    """Load CSV and perform a minimal clean.

    - Renames common alternative column names to the internal standard names.
    - Optionally enforces Validity Left/Right == 1 if those columns exist.
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # Trim strings
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()

    # Column mapping (makes the pipeline more robust across exporters)
    cmap = load_columns_map(columns_map_path)
    rename_df_columns_inplace(df, cmap)

    # Convert numeric columns when present
    num_cols = [
        "Recording Time Stamp[ms]", "Gaze Point X[px]", "Gaze Point Y[px]",
        "Validity Left", "Validity Right",
        "Fixation Index", "Fixation Duration[ms]",
        "Saccade Duration[ms]", "Saccade Amplitude[px]",
        "Saccade Velocity Average[px/ms]", "Saccade Velocity Peak[px/ms]",
        "Blink Duration[ms]", "Gaze Velocity[px/ms]",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Required columns for coordinate filtering
    miss = missing_required(df.columns, ["Gaze Point X[px]", "Gaze Point Y[px]"])
    if miss:
        raise ValueError(f"Missing required gaze columns: {miss}. You can adjust configs/columns_default.json or pass columns_map_path.")

    mask = (
        df["Gaze Point X[px]"].between(0, screen_w)
        & df["Gaze Point Y[px]"].between(0, screen_h)
    )

    if require_validity and ("Validity Left" in df.columns) and ("Validity Right" in df.columns):
        mask = mask & (df["Validity Left"] == 1) & (df["Validity Right"] == 1)

    clean = df[mask].copy()
    return df, clean


def quality_report(df: pd.DataFrame, clean: pd.DataFrame):
    duration_s = (
        df["Recording Time Stamp[ms]"].max() - df["Recording Time Stamp[ms]"].min()
    ) / 1000
    return {
        "total_rows": len(df),
        "valid_rows_after_clean": len(clean),
        "valid_ratio_after_clean(%)": round(len(clean) / max(len(df), 1) * 100, 2),
        "duration_seconds": round(duration_s, 2),
        "mean_gaze_velocity(px/ms)": float(pd.to_numeric(df.get("Gaze Velocity[px/ms]"), errors="coerce").mean()),
    }


def plot_heatmap(clean: pd.DataFrame, out_png: str, screen_w: int, screen_h: int):
    plt.figure(figsize=(6, 8))
    sns.kdeplot(
        x=clean["Gaze Point X[px]"],
        y=clean["Gaze Point Y[px]"],
        fill=True,
        cmap="viridis",
        bw_adjust=0.7,
        thresh=0.02,
    )
    plt.xlim(0, screen_w)
    plt.ylim(screen_h, 0)
    plt.title("Gaze Heatmap")
    plt.xlabel("X (px)")
    plt.ylabel("Y (px)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_scanpath(clean: pd.DataFrame, out_png: str, screen_w: int, screen_h: int, step: int = 20):
    sample = clean.iloc[::step].copy()
    plt.figure(figsize=(6, 8))
    plt.plot(sample["Gaze Point X[px]"], sample["Gaze Point Y[px]"], linewidth=0.8, alpha=0.7)
    plt.scatter(sample["Gaze Point X[px]"], sample["Gaze Point Y[px]"], s=6, alpha=0.6)
    plt.xlim(0, screen_w)
    plt.ylim(screen_h, 0)
    plt.title("Scanpath (sampled)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def compute_aoi_metrics(clean: pd.DataFrame, aois: dict):
    def in_aoi(x, y, rect):
        x1, y1, x2, y2 = rect
        return (x >= x1) & (x < x2) & (y >= y1) & (y < y2)

    rows = []
    t0 = clean["Recording Time Stamp[ms]"].min()
    for name, rect in aois.items():
        mask = in_aoi(clean["Gaze Point X[px]"], clean["Gaze Point Y[px]"], rect)
        sub = clean[mask]
        dwell_ms = sub["Fixation Duration[ms]"].dropna().sum()
        fix_count = sub["Fixation Index"].dropna().nunique()
        ttff = (sub["Recording Time Stamp[ms]"].min() - t0) if len(sub) > 0 else np.nan
        rows.append({
            "AOI": name,
            "samples": len(sub),
            "dwell_time_ms": dwell_ms,
            "fixation_count": fix_count,
            "TTFF_ms": ttff,
        })

    return pd.DataFrame(rows)
