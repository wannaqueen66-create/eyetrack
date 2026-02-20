#!/usr/bin/env python3
"""Batch heatmaps with group aggregation + difference plots.

This script is designed for "people/group" comparisons:
  - Individual heatmap per participant
  - Aggregated heatmap per group (SportFreq, Experience, and/or 4-way cross)
  - Difference plot for binary splits (High vs Low)

Manifest CSV (recommended columns):
  - name (string) OR participant_id (string)
  - SportFreq (High/Low)  # case-insensitive
  - Experience (High/Low) # case-insensitive
  - csv_path (path, optional if you pass --csv_dir)

Outputs (default outdir=outputs_batch_groups):
  - individual/<participant_id>/heatmap.png
  - groups/SportFreq-High/heatmap.png, groups/SportFreq-Low/heatmap.png
  - groups/Experience-High/heatmap.png, groups/Experience-Low/heatmap.png
  - groups/4way/<4way_label>/heatmap.png
  - compare/SportFreq_diff.png (High vs Low + log-ratio)
  - compare/Experience_diff.png
  - compare/4way_grid.png

Notes:
  - Densities are built from a fixed grid histogram + Gaussian smoothing, so that
    cross-group differences are well-defined.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import load_and_clean, plot_heatmap

try:
    from scipy.ndimage import gaussian_filter
except Exception as e:
    raise SystemExit(
        "scipy is required for batch_heatmap_groups.py (scipy.ndimage.gaussian_filter). "
        "Please install requirements.txt. Error: " + repr(e)
    )


def norm_level(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().lower()
    if s in {"high", "h", "1", "true", "yes"}:
        return "High"
    if s in {"low", "l", "0", "false", "no"}:
        return "Low"
    return str(x).strip()


def density_from_points(xy: np.ndarray, screen_w: int, screen_h: int, bins: int = 200, sigma: float = 2.0):
    """Return a normalized 2D density image (H x W) in screen coordinate space."""
    if xy.size == 0:
        H = np.zeros((bins, bins), dtype=float)
        return H

    x = np.clip(xy[:, 0], 0, screen_w)
    y = np.clip(xy[:, 1], 0, screen_h)

    # histogram2d: first dim is x, second is y (we'll transpose later for imshow)
    H, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=[bins, bins],
        range=[[0, screen_w], [0, screen_h]],
    )

    # smooth
    H = gaussian_filter(H, sigma=sigma, mode="nearest")

    # normalize to probability density-like (sum=1)
    s = H.sum()
    if s > 0:
        H = H / s
    return H


def save_density_png(H: np.ndarray, out_png: Path, title: str, cmap: str = "viridis", vmin=None, vmax=None):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 8))
    # transpose to align axes with screen convention; origin upper
    plt.imshow(H.T, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def save_density_overlay(H: np.ndarray, out_png: Path, title: str, background_img: str, screen_w: int, screen_h: int, alpha: float = 0.55, cmap: str = "inferno", vmin=None, vmax=None):
    """Overlay density heatmap on a background image (scene).

    Note: background image is stretched to [0..screen_w]x[0..screen_h].
    """
    bg = plt.imread(background_img)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 8))
    plt.imshow(bg, extent=[0, screen_w, screen_h, 0], aspect="auto")
    plt.imshow(H.T, origin="upper", extent=[0, screen_w, screen_h, 0], cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax, aspect="auto")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def save_binary_compare(A: np.ndarray, B: np.ndarray, out_png: Path, title: str, eps: float = 1e-12):
    """Save 3-panel: A, B, log-ratio(A/B)."""
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Use the same scaling for A/B panels
    vmax = max(float(A.max()), float(B.max()), eps)

    # log ratio for interpretability
    L = np.log2((A + eps) / (B + eps))
    # symmetric limits
    lim = float(np.nanmax(np.abs(L))) if np.isfinite(L).any() else 1.0
    lim = max(lim, 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax in axes:
        ax.axis("off")

    axes[0].imshow(A.T, origin="upper", cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
    axes[0].set_title("Group A")

    axes[1].imshow(B.T, origin="upper", cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
    axes[1].set_title("Group B")

    im = axes[2].imshow(L.T, origin="upper", cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    axes[2].set_title("log2(A/B)")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Batch heatmaps by participant + group aggregation + difference plots")
    ap.add_argument("--manifest", required=True, help="CSV with columns: participant_id,SportFreq,Experience,(optional)csv_path")
    ap.add_argument("--csv_dir", default=None, help="If csv_path is omitted in manifest, find CSVs under this directory by participant_id")
    ap.add_argument("--outdir", default="outputs_batch_groups")

    ap.add_argument("--screen_w", type=int, default=1280)
    ap.add_argument("--screen_h", type=int, default=1440)

    ap.add_argument("--bins", type=int, default=200, help="Grid bins for density (default 200x200)")
    ap.add_argument("--sigma", type=float, default=2.0, help="Gaussian smoothing sigma (default 2.0)")

    ap.add_argument("--require_validity", action="store_true", help="Require Validity Left/Right == 1 (if columns exist)")
    ap.add_argument("--columns_map", default=None, help="Path to JSON mapping of required columns to candidate names")

    # Background overlay (optional)
    ap.add_argument("--background_img", default=None, help="Optional background image (png/jpg) to overlay group densities")
    ap.add_argument("--alpha", type=float, default=0.55, help="Overlay alpha (default 0.55)")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "individual").mkdir(parents=True, exist_ok=True)
    (outdir / "groups").mkdir(parents=True, exist_ok=True)
    (outdir / "compare").mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(args.manifest)

    # Accept either participant_id or name
    id_col = None
    if "participant_id" in m.columns:
        id_col = "participant_id"
    elif "name" in m.columns:
        id_col = "name"
    else:
        raise SystemExit("Manifest must contain either 'participant_id' or 'name' column")

    req = {"SportFreq", "Experience"}
    miss = req - set(m.columns)
    if miss:
        raise SystemExit(f"Manifest missing columns: {sorted(miss)}")

    has_csv_path = "csv_path" in m.columns
    if (not has_csv_path) and (not args.csv_dir):
        raise SystemExit("Manifest has no csv_path column; please pass --csv_dir so the script can locate each participant's CSV")

    csv_dir = Path(args.csv_dir) if args.csv_dir else None
    if csv_dir and (not csv_dir.exists()):
        raise SystemExit(f"csv_dir not found: {csv_dir}")

    def resolve_csv_path(participant_id: str, csv_path_value):
        # 1) explicit path in manifest
        if has_csv_path:
            v = "" if (csv_path_value is None or (isinstance(csv_path_value, float) and np.isnan(csv_path_value))) else str(csv_path_value).strip()
            if v:
                return v

        # 2) find under csv_dir
        pid = participant_id.strip()
        # Most exporters use patterns like: raw_<name>_*.csv
        # Try strict patterns first to avoid accidental multiple matches.
        patterns = [
            f"**/{pid}.csv",
            f"**/{pid}_*.csv",
            f"**/{pid}-*.csv",
            f"**/raw_{pid}.csv",
            f"**/raw_{pid}_*.csv",
            f"**/raw-{pid}.csv",
            f"**/raw-{pid}-*.csv",
            f"**/*_{pid}_*.csv",
            f"**/*_{pid}-*.csv",
            f"**/{pid}*.csv",
        ]
        matches = []
        for pat in patterns:
            matches = sorted(csv_dir.glob(pat))
            if matches:
                break
        if not matches:
            raise FileNotFoundError(f"No CSV matched participant_id={pid!r} under {csv_dir}")
        if len(matches) > 1:
            raise FileExistsError(f"Multiple CSVs matched participant_id={pid!r} under {csv_dir}: {[str(p) for p in matches[:10]]}")
        return str(matches[0])

    # Accumulate points for groups
    points_sport = {"High": [], "Low": []}
    points_exp = {"High": [], "Low": []}
    points_4 = {
        "Experience-High×SportFreq-High": [],
        "Experience-High×SportFreq-Low": [],
        "Experience-Low×SportFreq-High": [],
        "Experience-Low×SportFreq-Low": [],
    }

    rows = []

    for _, r in m.iterrows():
        pid = str(r[id_col]).strip()
        sf = norm_level(r["SportFreq"])  # High/Low
        ex = norm_level(r["Experience"])  # High/Low

        csv_path = resolve_csv_path(pid, r["csv_path"] if has_csv_path else None)

        df, clean = load_and_clean(
            csv_path,
            screen_w=args.screen_w,
            screen_h=args.screen_h,
            require_validity=args.require_validity,
            columns_map_path=args.columns_map,
        )

        # Individual heatmap (KDE) for quick inspection
        one_out = outdir / "individual" / pid
        one_out.mkdir(parents=True, exist_ok=True)
        plot_heatmap(clean, str(one_out / "heatmap.png"), args.screen_w, args.screen_h)

        xy = clean[["Gaze Point X[px]", "Gaze Point Y[px]"]].dropna().to_numpy(dtype=float)

        if sf in points_sport:
            points_sport[sf].append(xy)
        if ex in points_exp:
            points_exp[ex].append(xy)

        key4 = f"Experience-{ex}×SportFreq-{sf}"
        if key4 in points_4:
            points_4[key4].append(xy)

        rows.append({
            "participant_id": pid,
            "csv_path": csv_path,
            "SportFreq": sf,
            "Experience": ex,
            "n_samples_after_clean": int(len(clean)),
        })

    pd.DataFrame(rows).to_csv(outdir / "participants_summary.csv", index=False)

    # ---- Group aggregated heatmaps (grid density) ----
    def concat_list(lst):
        return np.concatenate(lst, axis=0) if len(lst) and any(x.size for x in lst) else np.zeros((0, 2), dtype=float)

    # SportFreq
    dens_sf = {}
    for level in ["High", "Low"]:
        xy = concat_list(points_sport[level])
        H = density_from_points(xy, args.screen_w, args.screen_h, bins=args.bins, sigma=args.sigma)
        dens_sf[level] = H
        base = outdir / "groups" / f"SportFreq-{level}"
        save_density_png(H, base / "heatmap_density.png", f"SportFreq-{level} (density)")
        if args.background_img:
            save_density_overlay(H, base / "heatmap_overlay.png", f"SportFreq-{level} (overlay)", args.background_img, args.screen_w, args.screen_h, alpha=args.alpha)

    # Experience
    dens_ex = {}
    for level in ["High", "Low"]:
        xy = concat_list(points_exp[level])
        H = density_from_points(xy, args.screen_w, args.screen_h, bins=args.bins, sigma=args.sigma)
        dens_ex[level] = H
        base = outdir / "groups" / f"Experience-{level}"
        save_density_png(H, base / "heatmap_density.png", f"Experience-{level} (density)")
        if args.background_img:
            save_density_overlay(H, base / "heatmap_overlay.png", f"Experience-{level} (overlay)", args.background_img, args.screen_w, args.screen_h, alpha=args.alpha)

    # 4-way
    dens_4 = {}
    for k in points_4.keys():
        xy = concat_list(points_4[k])
        H = density_from_points(xy, args.screen_w, args.screen_h, bins=args.bins, sigma=args.sigma)
        dens_4[k] = H
        base = outdir / "groups" / "4way" / k
        save_density_png(H, base / "heatmap_density.png", f"{k} (density)")
        if args.background_img:
            save_density_overlay(H, base / "heatmap_overlay.png", f"{k} (overlay)", args.background_img, args.screen_w, args.screen_h, alpha=args.alpha)

    # ---- Compare plots (difference) ----
    save_binary_compare(
        dens_sf["High"],
        dens_sf["Low"],
        outdir / "compare" / "SportFreq_diff.png",
        title="SportFreq: High vs Low (density + log2 ratio)",
    )
    save_binary_compare(
        dens_ex["High"],
        dens_ex["Low"],
        outdir / "compare" / "Experience_diff.png",
        title="Experience: High vs Low (density + log2 ratio)",
    )

    # 4-way grid: 2x2 (Experience rows, SportFreq cols)
    # Layout:
    #   Exp-High: [SF-High, SF-Low]
    #   Exp-Low : [SF-High, SF-Low]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    order = [
        ["Experience-High×SportFreq-High", "Experience-High×SportFreq-Low"],
        ["Experience-Low×SportFreq-High", "Experience-Low×SportFreq-Low"],
    ]
    vmax = max(float(H.max()) for H in dens_4.values()) if dens_4 else 1.0
    vmax = max(vmax, 1e-12)
    for i in range(2):
        for j in range(2):
            k = order[i][j]
            ax = axes[i, j]
            ax.axis("off")
            ax.imshow(dens_4[k].T, origin="upper", cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
            ax.set_title(k)
    fig.suptitle("4-way groups (shared scale)")
    fig.tight_layout()
    fig.savefig(outdir / "compare" / "4way_grid.png", dpi=300)
    plt.close(fig)

    print("Done. Outputs in:", str(outdir))


if __name__ == "__main__":
    main()
