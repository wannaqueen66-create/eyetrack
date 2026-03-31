#!/usr/bin/env python3
"""Publication-oriented batch heatmaps with overall + Experience aggregation.

Purpose
-------
Generate eye-tracking heatmaps with output logic narrowed to the manuscript mainline:
- Individual heatmap per participant
- One aggregated heatmap for all participants (`Overall`)
- Two aggregated heatmaps by Experience (`Experience-High`, `Experience-Low`)
- One Experience comparison panel

Visual goal
-----------
The rendering is tuned to resemble the visual character commonly seen in Tobii /
Tobii Pro Lab heatmaps rather than a stylized academic redesign:
- strong hotspot emphasis
- blue/green/yellow/red attention ramp
- direct scene overlay
- soft tails and bright centers
- minimal decorative framing
"""

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import font_manager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import load_and_clean

try:
    from scipy.ndimage import gaussian_filter
except Exception as e:
    raise SystemExit(
        "scipy is required for batch_heatmap_groups.py (scipy.ndimage.gaussian_filter). "
        "Please install requirements.txt. Error: " + repr(e)
    )


PLOT_DPI = 600
DEFAULT_FIGSIZE = (6.7, 7.9)
COMPARE_PANEL_SIZE = (4.2, 5.25)
TITLE_PAD = 7
TEXT_PRIMARY = "#202124"
TEXT_SECONDARY = "#667085"
OVERLAY_BG_ALPHA = 0.10


def norm_level(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().lower()
    if s in {"high", "h", "1", "true", "yes"}:
        return "High"
    if s in {"low", "l", "0", "false", "no"}:
        return "Low"
    return str(x).strip()


def ascii_safe(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s2 = re.sub(r"[^\x00-\x7F]+", "", s)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


def setup_cjk_font(prefer_names=None):
    if prefer_names is None:
        prefer_names = [
            "Noto Sans CJK SC",
            "Noto Sans CJK",
            "Source Han Sans SC",
            "Source Han Sans",
            "SimHei",
            "Microsoft YaHei",
            "PingFang SC",
        ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in prefer_names:
        if name in available:
            mpl.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            mpl.rcParams["axes.unicode_minus"] = False
            return name
    return None


def setup_publication_style(font_family: str = "DejaVu Sans"):
    mpl.rcParams.update(
        {
            "figure.dpi": PLOT_DPI,
            "savefig.dpi": PLOT_DPI,
            "font.family": font_family,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "regular",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _make_tobii_cmap():
    # Approximate Tobii-style ramp: cool tails → green/yellow → hot red/white core
    stops = [
        (0.00, (0.00, 0.10, 0.80, 0.00)),
        (0.05, (0.00, 0.20, 0.95, 0.30)),
        (0.15, (0.00, 0.60, 1.00, 0.50)),
        (0.30, (0.00, 0.90, 0.50, 0.65)),
        (0.50, (0.80, 0.95, 0.00, 0.78)),
        (0.70, (1.00, 0.60, 0.00, 0.90)),
        (0.85, (1.00, 0.25, 0.00, 0.96)),
        (1.00, (1.00, 0.00, 0.00, 1.00)),
    ]
    return mcolors.LinearSegmentedColormap.from_list("tobii_like", stops, N=256)


def get_cmap(name: str):
    n = (name or "tobii").strip().lower()
    if n in {"tobii", "tobii-pro", "prolab", "tobii_pro", "tobii_like"}:
        return _make_tobii_cmap()
    return plt.get_cmap(name)


def density_from_points(
    xy: np.ndarray,
    screen_w: int,
    screen_h: int,
    bins: int = 220,
    sigma: float = 2.6,
    weights: np.ndarray | None = None,
):
    if xy.size == 0:
        return np.zeros((bins, bins), dtype=float)

    x = np.clip(xy[:, 0], 0, screen_w)
    y = np.clip(xy[:, 1], 0, screen_h)

    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or len(w) != len(x):
            raise ValueError(f"weights must be 1D and match xy length; got weights.shape={w.shape}, xy.shape={xy.shape}")

    H, _, _ = np.histogram2d(
        x,
        y,
        bins=[bins, bins],
        range=[[0, screen_w], [0, screen_h]],
        weights=w,
    )
    H = gaussian_filter(H, sigma=sigma, mode="nearest")
    s = H.sum()
    if s > 0:
        H = H / s
    return H


def _alpha_map_from_density(H: np.ndarray, alpha: float, thresh_rel: float):
    if H.size == 0:
        return None
    mx = float(np.max(H)) if np.isfinite(H).any() else 0.0
    if mx <= 0:
        return np.zeros_like(H, dtype=float)
    Hn = np.clip(H / mx, 0, 1)
    # More Tobii-like: softer tails + quickly intensified hotspots
    A = np.clip((Hn ** 0.25) * float(alpha), 0, 1)
    if thresh_rel is not None and thresh_rel > 0:
        A = np.where(Hn >= float(thresh_rel), A, 0.0)
    return A


def _normalize_background(bg: np.ndarray) -> np.ndarray:
    arr = bg.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        rgb = arr[..., :3]
        alpha = arr[..., 3:4]
        arr = rgb * alpha + (1 - alpha)
    return np.clip(arr[..., :3], 0, 1)


def _detect_content_bounds(bg: np.ndarray, crop_pct: float = 0.08):
    """Crop a fixed percentage from all edges to remove VR panoramic dark corners.

    Args:
        bg: Background image array (H, W, C) or (H, W).
        crop_pct: Fraction of each edge to crop (default 8%).

    Returns (row_min, row_max, col_min, col_max) in pixel coordinates,
    or None if crop_pct is 0.
    """
    if crop_pct <= 0:
        return None
    h, w = bg.shape[:2]
    rmin = int(h * crop_pct)
    rmax = int(h * (1 - crop_pct))
    cmin = int(w * crop_pct)
    cmax = int(w * (1 - crop_pct))
    return (rmin, rmax, cmin, cmax)


def _style_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _meta_line(ax, left_text: str = "", right_text: str = ""):
    if not left_text and not right_text:
        return
    y = -0.035
    if left_text:
        ax.text(0.00, y, left_text, transform=ax.transAxes, ha="left", va="top", fontsize=8.6, color=TEXT_SECONDARY)
    if right_text:
        ax.text(1.00, y, right_text, transform=ax.transAxes, ha="right", va="top", fontsize=8.6, color=TEXT_SECONDARY)


def _draw_density(
    ax,
    H,
    cmap,
    title,
    background_img=None,
    screen_w=None,
    screen_h=None,
    alpha=0.88,
    thresh_rel=0.006,
    vmin=None,
    vmax=None,
    meta_left: str = "",
    meta_right: str = "",
):
    if background_img is not None:
        bg = _normalize_background(plt.imread(background_img))
        # Detect and crop black borders from VR/panoramic renders
        bounds = _detect_content_bounds(bg)
        if bounds is not None:
            rmin, rmax, cmin, cmax = bounds
            bg = bg[rmin:rmax+1, cmin:cmax+1]
            # Map pixel bounds to screen coordinates for extent
            h_orig, w_orig = plt.imread(background_img).shape[:2]
            x0 = cmin / w_orig * screen_w
            x1 = (cmax + 1) / w_orig * screen_w
            y0 = rmin / h_orig * screen_h
            y1 = (rmax + 1) / h_orig * screen_h
            extent = [x0, x1, y1, y0]
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
        else:
            extent = [0, screen_w, screen_h, 0]
        ax.imshow(bg, extent=extent, aspect="auto")
        if OVERLAY_BG_ALPHA > 0:
            ax.imshow(
                np.ones((10, 10, 3), dtype=float),
                extent=extent,
                aspect="auto",
                alpha=OVERLAY_BG_ALPHA,
            )
        A = _alpha_map_from_density(H, alpha=alpha, thresh_rel=thresh_rel)
        ax.imshow(
            H.T,
            origin="upper",
            extent=[0, screen_w, screen_h, 0],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=A.T if A is not None else alpha,
            aspect="auto",
        )
        ax.set_facecolor("white")
    else:
        ax.imshow(H.T, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    _style_axis(ax)
    if title:
        ax.set_title(title, pad=TITLE_PAD, color=TEXT_PRIMARY)
    _meta_line(ax, left_text=meta_left, right_text=meta_right)


def _add_panel_letter(ax, text: str):
    ax.text(
        0.015,
        0.985,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.20", facecolor=(0, 0, 0, 0.42), edgecolor=(1, 1, 1, 0.35), linewidth=0.6),
    )


def save_density_png(
    H: np.ndarray,
    out_png: Path,
    title: str,
    cmap,
    figsize=DEFAULT_FIGSIZE,
    vmin=None,
    vmax=None,
    meta_left: str = "",
    meta_right: str = "",
):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    _draw_density(ax, H, cmap=cmap, title=title, vmin=vmin, vmax=vmax, meta_left=meta_left, meta_right=meta_right)
    fig.savefig(out_png)
    plt.close(fig)


def save_density_overlay(
    H: np.ndarray,
    out_png: Path,
    title: str,
    background_img: str,
    screen_w: int,
    screen_h: int,
    cmap,
    alpha: float = 0.88,
    thresh_rel: float = 0.006,
    figsize=DEFAULT_FIGSIZE,
    vmin=None,
    vmax=None,
    meta_left: str = "",
    meta_right: str = "",
):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    _draw_density(
        ax,
        H,
        cmap=cmap,
        title=title,
        background_img=background_img,
        screen_w=screen_w,
        screen_h=screen_h,
        alpha=alpha,
        thresh_rel=thresh_rel,
        vmin=vmin,
        vmax=vmax,
        meta_left=meta_left,
        meta_right=meta_right,
    )
    fig.savefig(out_png)
    plt.close(fig)


def save_experience_compare(
    high: np.ndarray,
    low: np.ndarray,
    out_png: Path,
    cmap,
    title: str,
    label_high: str,
    label_low: str,
    background_img: str | None = None,
    screen_w: int | None = None,
    screen_h: int | None = None,
    alpha: float = 0.88,
    thresh_rel: float = 0.006,
    point_source_label: str = "",
    sample_meta: str = "",
):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    vmax = max(float(high.max()), float(low.max()), 1e-12)
    overlap = np.minimum(high, low)
    ovmax = max(float(overlap.max()), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(COMPARE_PANEL_SIZE[0] * 3, COMPARE_PANEL_SIZE[1]))
    fig.subplots_adjust(top=0.83, bottom=0.10, wspace=0.04)

    _draw_density(
        axes[0], high, cmap=cmap, title=label_high,
        background_img=background_img, screen_w=screen_w, screen_h=screen_h,
        alpha=alpha, thresh_rel=thresh_rel, vmin=0, vmax=vmax,
        meta_left="Relative density", meta_right="Shared scale"
    )
    _add_panel_letter(axes[0], "A")

    _draw_density(
        axes[1], low, cmap=cmap, title=label_low,
        background_img=background_img, screen_w=screen_w, screen_h=screen_h,
        alpha=alpha, thresh_rel=thresh_rel, vmin=0, vmax=vmax,
        meta_left="Relative density", meta_right="Shared scale"
    )
    _add_panel_letter(axes[1], "B")

    _draw_density(
        axes[2], overlap, cmap=plt.get_cmap("Greens"), title="Shared attention pattern",
        background_img=background_img, screen_w=screen_w, screen_h=screen_h,
        alpha=0.82, thresh_rel=thresh_rel, vmin=0, vmax=ovmax,
        meta_left="Overlap density", meta_right="Within-scene"
    )
    _add_panel_letter(axes[2], "C")

    fig.text(0.01, 0.965, title, ha="left", va="top", fontsize=12.4, color=TEXT_PRIMARY, fontweight="bold")
    sub = "Tobii-like heatmap rendering; warmer colors indicate stronger attention concentration."
    fig.text(0.01, 0.928, sub, ha="left", va="top", fontsize=9.0, color=TEXT_SECONDARY)
    extra = "  |  ".join([x for x in [point_source_label, sample_meta] if x])
    if extra:
        fig.text(0.01, 0.896, extra, ha="left", va="top", fontsize=8.7, color=TEXT_SECONDARY)

    fig.savefig(out_png)
    plt.close(fig)


def concat_list(lst):
    if not lst:
        return np.zeros((0, 2), dtype=np.float32), None
    xys = [xy for (xy, _) in lst if isinstance(xy, np.ndarray) and xy.size]
    if not xys:
        return np.zeros((0, 2), dtype=np.float32), None
    XY = np.concatenate(xys, axis=0)
    ws = [w for (xy, w) in lst if isinstance(xy, np.ndarray) and xy.size and w is not None]
    valid_count = len([1 for (xy, _) in lst if isinstance(xy, np.ndarray) and xy.size])
    W = np.concatenate(ws, axis=0) if len(ws) == valid_count and ws else None
    return XY, W


def main():
    ap = argparse.ArgumentParser(description="Batch heatmaps by participant + overall + Experience aggregation")
    ap.add_argument("--manifest", required=True, help="CSV with columns: name/participant_id, Experience, (optional) csv_path")
    ap.add_argument("--csv_dir", default=None, help="If csv_path is omitted in manifest, find CSVs under this directory")
    ap.add_argument("--outdir", default="outputs_batch_groups")

    ap.add_argument("--screen_w", type=int, default=1280)
    ap.add_argument("--screen_h", type=int, default=1440)

    ap.add_argument("--bins", type=int, default=220, help="Grid bins for density (default 220x220)")
    ap.add_argument("--sigma", type=float, default=3.0, help="Gaussian smoothing sigma (default 2.6, closer to Tobii-like diffusion)")

    ap.add_argument("--point_source", default="fixation", choices=["gaze", "fixation"], help="Point source (default: fixation, matching Tobii Pro Lab)")
    ap.add_argument("--weight", default="fixation_duration", choices=["none", "fixation_duration"], help="Weighting (default: fixation_duration, matching Tobii Pro Lab)")
    ap.add_argument("--fixation_dedup", default="index", choices=["index", "none"], help="How to deduplicate fixation points")

    ap.add_argument("--require_validity", action="store_true", help="Require Validity Left/Right == 1 when columns exist")
    ap.add_argument("--columns_map", default=None, help="Path to JSON mapping of required columns to candidate names")

    ap.add_argument("--background_img", default=None, help="Optional background image (png/jpg) to overlay all heatmaps")
    ap.add_argument("--alpha", type=float, default=0.88, help="Overlay alpha (default 0.88, closer to Tobii-like heat visibility)")
    ap.add_argument("--thresh", type=float, default=0.003, help="Relative threshold in [0,1] to hide very low-density tails")
    ap.add_argument("--cmap", default="tobii", help="Heatmap colormap (default: tobii-like)")

    ap.add_argument(
        "--title_mode",
        default="english",
        choices=["english", "raw", "none"],
        help="Figure titles: english=ASCII-only (default), raw=original, none=no titles",
    )
    ap.add_argument("--font", default="DejaVu Sans", help="Font family for figure text (default: DejaVu Sans)")
    ap.add_argument("--quiet_glyph_warning", action="store_true", help="Suppress matplotlib glyph warnings")

    ap.add_argument("--resume", action="store_true", default=True, help="Resume using cached per-participant points.npy if present")
    ap.add_argument("--no_resume", action="store_false", dest="resume", help="Disable resume cache")
    ap.add_argument("--fail_fast", action="store_true", help="Stop on first error")
    ap.add_argument("--skip_individual", action="store_true", help="Skip per-participant figure output; only produce group aggregates and comparison (much faster)")

    args = ap.parse_args()

    if args.title_mode == "raw":
        if args.font and str(args.font).lower() != "auto":
            mpl.rcParams["font.sans-serif"] = [args.font, "DejaVu Sans"]
            mpl.rcParams["axes.unicode_minus"] = False
        else:
            setup_cjk_font()

    if args.quiet_glyph_warning:
        warnings.filterwarnings("ignore", message=r"Glyph .* missing from font\(s\) .*", category=UserWarning)

    setup_publication_style(font_family=(args.font or "DejaVu Sans"))
    cmap = get_cmap(args.cmap)

    def fmt_title(s: str) -> str:
        if args.title_mode == "none":
            return ""
        if args.title_mode == "english":
            t = ascii_safe(s)
            return t if t else "Heatmap"
        return str(s)

    point_source_label = "Fixation-based density" if args.point_source == "fixation" else "Gaze-based density"
    if args.weight == "fixation_duration":
        point_source_label += " (duration-weighted)"

    outdir = Path(args.outdir)
    if not getattr(args, "skip_individual", False):
        (outdir / "individual").mkdir(parents=True, exist_ok=True)
    (outdir / "groups").mkdir(parents=True, exist_ok=True)
    (outdir / "compare").mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(args.manifest)
    if "participant_id" in m.columns:
        id_col = "participant_id"
    elif "name" in m.columns:
        id_col = "name"
    else:
        raise SystemExit("Manifest must contain either 'participant_id' or 'name' column")

    if "Experience" not in m.columns:
        raise SystemExit("Manifest must contain column: Experience")

    has_csv_path = "csv_path" in m.columns
    if (not has_csv_path) and (not args.csv_dir):
        raise SystemExit("Manifest has no csv_path column; please pass --csv_dir so the script can locate each participant's CSV")

    csv_dir = Path(args.csv_dir) if args.csv_dir else None
    if csv_dir and (not csv_dir.exists()):
        raise SystemExit(f"csv_dir not found: {csv_dir}")

    def resolve_csv_path(participant_id: str, csv_path_value):
        if has_csv_path:
            v = "" if (csv_path_value is None or (isinstance(csv_path_value, float) and np.isnan(csv_path_value))) else str(csv_path_value).strip()
            if v:
                return v

        pid = participant_id.strip()
        patterns = [
            f"**/raw_{pid}_*.csv",
            f"**/raw_{pid}.csv",
            f"**/{pid}.csv",
            f"**/{pid}_*.csv",
            f"**/{pid}-*.csv",
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
            raise FileNotFoundError(f"No CSV matched id={pid!r} under {csv_dir}")
        if len(matches) > 1:
            # Multiple CSVs for same participant (e.g. merged from multiple order groups).
            # Return all paths so the caller can concatenate data.
            return [str(p) for p in matches]
        return str(matches[0])

    points_all = []
    points_exp = {"High": [], "Low": []}
    rows = []
    errors = []

    skip_ind = getattr(args, 'skip_individual', False)

    for _, r in m.iterrows():
        pid = str(r[id_col]).strip()
        ex = norm_level(r["Experience"])

        # Individual output directory (only created when needed)
        one_out = None
        if not skip_ind:
            one_out = outdir / "individual" / pid
            one_out.mkdir(parents=True, exist_ok=True)

        cache_tag = f"{args.point_source}"
        if args.point_source == "fixation":
            cache_tag += f"_dedup-{args.fixation_dedup}"
        if args.weight != "none":
            cache_tag += f"_w-{args.weight}"

        pid_title = fmt_title(pid)

        try:
            # Try loading from cache (only when individual dirs exist)
            source_csv = None
            loaded_from_cache = False
            if not skip_ind:
                points_path = one_out / f"points_{cache_tag}.npy"
                weights_path = one_out / f"weights_{cache_tag}.npy"
                meta_path = one_out / f"meta_{cache_tag}.json"
                if args.resume and points_path.exists():
                    xy = np.load(points_path)
                    w = np.load(weights_path) if weights_path.exists() else None
                    loaded_from_cache = True

            if not loaded_from_cache:
                w = None
                csv_path = resolve_csv_path(pid, r["csv_path"] if has_csv_path else None)
                # Handle single path or list of paths (multiple CSVs per participant)
                csv_paths = csv_path if isinstance(csv_path, list) else [csv_path]
                dfs = []
                cleans = []
                for cp in csv_paths:
                    _df, _clean = load_and_clean(
                        cp,
                        screen_w=args.screen_w,
                        screen_h=args.screen_h,
                        require_validity=args.require_validity,
                        columns_map_path=args.columns_map,
                    )
                    dfs.append(_df)
                    cleans.append(_clean)
                df = pd.concat(dfs, ignore_index=True)
                clean = pd.concat(cleans, ignore_index=True)
                csv_path = csv_paths[0]  # record first path for metadata

                if args.point_source == "fixation":
                    if ("Fixation Point X[px]" not in clean.columns) or ("Fixation Point Y[px]" not in clean.columns):
                        raise ValueError("point_source=fixation but Fixation Point X/Y columns are missing")
                    sub = clean.copy()
                    if "Fixation Index" in sub.columns:
                        sub = sub[pd.to_numeric(sub["Fixation Index"], errors="coerce").notna()]
                    if args.fixation_dedup == "index" and ("Fixation Index" in sub.columns):
                        sub = sub.dropna(subset=["Fixation Index"]).drop_duplicates(subset=["Fixation Index"], keep="first")
                    xy = sub[["Fixation Point X[px]", "Fixation Point Y[px]"]].dropna().to_numpy(dtype=np.float32)
                else:
                    xy = clean[["Gaze Point X[px]", "Gaze Point Y[px]"]].dropna().to_numpy(dtype=np.float32)

                if args.weight == "fixation_duration":
                    if args.point_source != "fixation":
                        raise ValueError("weight=fixation_duration requires point_source=fixation")
                    if "Fixation Duration[ms]" not in sub.columns:
                        raise ValueError("weight=fixation_duration but Fixation Duration[ms] column is missing")
                    w = pd.to_numeric(sub.get("Fixation Duration[ms]"), errors="coerce").to_numpy(dtype=float)
                    mask_xy = sub[["Fixation Point X[px]", "Fixation Point Y[px]"]].notna().all(axis=1).to_numpy()
                    w = w[mask_xy]
                    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
                else:
                    w = None

                source_csv = csv_path

                # Save cache only when not skipping individual
                if not skip_ind:
                    if w is not None:
                        np.save(weights_path, w.astype(np.float32))
                    np.save(points_path, xy)
                    meta_path.write_text(
                        json.dumps(
                            {
                                "id": pid,
                                "csv_path": csv_path,
                                "Experience": ex,
                                "point_source": args.point_source,
                                "weight": args.weight,
                                "fixation_dedup": args.fixation_dedup if args.point_source == "fixation" else None,
                                "n_points_after_clean": int(xy.shape[0]),
                                "weight_sum": float(np.nansum(w)) if w is not None else None,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

            point_count = int(xy.shape[0])

            # Individual figure output (skipped when --skip_individual)
            if not skip_ind:
                H = density_from_points(xy, args.screen_w, args.screen_h, bins=args.bins, sigma=args.sigma, weights=w)
                meta_right = f"n = {point_count:,}"
                save_density_png(H, one_out / "heatmap_density.png", fmt_title(f"Participant density: {pid_title}"), cmap, meta_left=point_source_label, meta_right=meta_right)

                if args.background_img:
                    save_density_overlay(
                        H, one_out / "heatmap.png", fmt_title(f"Participant: {pid_title}"),
                        args.background_img, args.screen_w, args.screen_h,
                        cmap=cmap, alpha=args.alpha, thresh_rel=args.thresh,
                        meta_left=point_source_label, meta_right=meta_right,
                    )
                    save_density_overlay(
                        H, one_out / "heatmap_overlay.png", fmt_title(f"Participant overlay: {pid_title}"),
                        args.background_img, args.screen_w, args.screen_h,
                        cmap=cmap, alpha=args.alpha, thresh_rel=args.thresh,
                        meta_left=point_source_label, meta_right=meta_right,
                    )
                else:
                    save_density_png(H, one_out / "heatmap.png", fmt_title(f"Participant density: {pid_title}"), cmap, meta_left=point_source_label, meta_right=meta_right)

            rows.append(
                {
                    "id": pid,
                    "Experience": ex,
                    "csv_path": source_csv,
                    "point_source": args.point_source,
                    "weight": args.weight,
                    "fixation_dedup": args.fixation_dedup if args.point_source == "fixation" else "",
                    "n_points": point_count,
                    "weight_sum": float(np.nansum(w)) if w is not None else np.nan,
                    "status": "ok",
                }
            )

            points_all.append((xy, w))
            if ex in points_exp:
                points_exp[ex].append((xy, w))

        except Exception as e:
            err = {"id": pid, "Experience": ex, "point_source": args.point_source, "weight": args.weight, "error": repr(e)}
            errors.append(err)
            rows.append({"id": pid, "Experience": ex, "csv_path": None, "point_source": args.point_source, "weight": args.weight, "n_points": 0, "status": "error", "error": repr(e)})
            if args.fail_fast:
                raise

    pd.DataFrame(rows).to_csv(outdir / "participants_summary.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(outdir / "errors.csv", index=False)

    ok_rows = pd.DataFrame([r for r in rows if r.get("status") == "ok"])
    n_total = len(ok_rows)
    n_high = int((ok_rows["Experience"] == "High").sum()) if not ok_rows.empty else 0
    n_low = int((ok_rows["Experience"] == "Low").sum()) if not ok_rows.empty else 0

    overall_xy, overall_w = concat_list(points_all)
    overall_H = density_from_points(overall_xy, args.screen_w, args.screen_h, bins=args.bins, sigma=args.sigma, weights=overall_w)
    overall_dir = outdir / "groups" / "Overall"
    overall_dir.mkdir(parents=True, exist_ok=True)
    overall_meta_right = f"participants = {n_total}"
    save_density_png(overall_H, overall_dir / "heatmap_density.png", fmt_title("Overall participant density"), cmap)
    if args.background_img:
        save_density_overlay(overall_H, overall_dir / "heatmap.png", fmt_title("Overall participants"), args.background_img, args.screen_w, args.screen_h, cmap=cmap, alpha=args.alpha, thresh_rel=args.thresh)
        save_density_overlay(overall_H, overall_dir / "heatmap_overlay.png", fmt_title("Overall participants overlay"), args.background_img, args.screen_w, args.screen_h, cmap=cmap, alpha=args.alpha, thresh_rel=args.thresh)
    else:
        save_density_png(overall_H, overall_dir / "heatmap.png", fmt_title("Overall participant density"), cmap)

    dens_ex = {}
    for level in ["High", "Low"]:
        xy, w = concat_list(points_exp[level])
        H = density_from_points(xy, args.screen_w, args.screen_h, bins=args.bins, sigma=args.sigma, weights=w)
        dens_ex[level] = H
        base = outdir / "groups" / f"Experience-{level}"
        base.mkdir(parents=True, exist_ok=True)
        n_group = n_high if level == "High" else n_low
        group_meta_right = f"participants = {n_group}"
        save_density_png(H, base / "heatmap_density.png", fmt_title(f"Experience {level}: density"), cmap)
        if args.background_img:
            save_density_overlay(H, base / "heatmap.png", fmt_title(f"Experience {level}"), args.background_img, args.screen_w, args.screen_h, cmap=cmap, alpha=args.alpha, thresh_rel=args.thresh)
            save_density_overlay(H, base / "heatmap_overlay.png", fmt_title(f"Experience {level}: overlay"), args.background_img, args.screen_w, args.screen_h, cmap=cmap, alpha=args.alpha, thresh_rel=args.thresh)
        else:
            save_density_png(H, base / "heatmap.png", fmt_title(f"Experience {level}: density"), cmap)

    save_experience_compare(
        dens_ex["High"],
        dens_ex["Low"],
        outdir / "compare" / "Experience_comparison.png",
        cmap=cmap,
        title=fmt_title("Experience-group heatmap comparison"),
        label_high="High experience",
        label_low="Low experience",
        background_img=args.background_img,
        screen_w=args.screen_w,
        screen_h=args.screen_h,
        alpha=args.alpha,
        thresh_rel=args.thresh,
        point_source_label=point_source_label,
        sample_meta=f"High group: n = {n_high}  |  Low group: n = {n_low}",
    )

    print("Done. Outputs in:", str(outdir))
    print("Main aggregation outputs:")
    print(" - groups/Overall")
    print(" - groups/Experience-High")
    print(" - groups/Experience-Low")
    print(" - compare/Experience_comparison.png")
    if errors:
        print(f"WARNING: {len(errors)} participants had errors. See: {outdir / 'errors.csv'}")


if __name__ == "__main__":
    main()
