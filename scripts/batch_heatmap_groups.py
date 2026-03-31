#!/usr/bin/env python3
"""Publication-oriented batch heatmaps with overall + Experience aggregation.

Purpose
-------
Generate eye-tracking heatmaps that are cleaner and easier to use in manuscripts
(e.g., Building and Environment submissions). This script now keeps the output
scope intentionally narrow:

- Individual heatmap per participant
- One aggregated heatmap for all participants (`Overall`)
- Two aggregated heatmaps by Experience (`Experience-High`, `Experience-Low`)
- Experience comparison figures only

Removed from the main output line:
- SportFreq aggregation
- 4-way Experience × SportFreq aggregation
- SportFreq comparison figures

Key design choices for publication-oriented figures
--------------------------------------------------
- English-only figure titles by default
- Clean figure canvas with no axes/ticks
- Consistent density normalization within each comparison figure
- Optional background overlay for scene-based interpretation
- Supports gaze-point or fixation-point heatmaps
- Optional fixation-duration weighting

Manifest CSV
------------
Required columns:
- name OR participant_id
- Experience (High/Low)

Optional columns:
- csv_path
- SportFreq (ignored by this script if present)

Typical usage
-------------
python scripts/batch_heatmap_groups.py \
  --manifest /content/group_manifest.csv \
  --csv_dir /content/csv \
  --screen_w 1748 --screen_h 2064 \
  --background_img /content/scene.png \
  --point_source fixation \
  --weight fixation_duration \
  --outdir /content/outputs_batch_groups
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
DEFAULT_FIGSIZE = (6.5, 7.68)
COMPARE_PANEL_SIZE = (4.2, 5.2)
TITLE_PAD = 8


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
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def get_cmap(name: str):
    n = (name or "tobii").strip().lower()
    if n in {"tobii", "tobii-pro", "prolab", "tobii_pro"}:
        try:
            return plt.get_cmap("turbo")
        except Exception:
            return plt.get_cmap("jet")
    return plt.get_cmap(name)


def density_from_points(
    xy: np.ndarray,
    screen_w: int,
    screen_h: int,
    bins: int = 220,
    sigma: float = 2.2,
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
    A = (Hn ** 0.6) * float(alpha)
    if thresh_rel is not None and thresh_rel > 0:
        A = np.where(Hn >= float(thresh_rel), A, 0.0)
    return A


def _draw_density(ax, H, cmap, title, background_img=None, screen_w=None, screen_h=None, alpha=0.62, thresh_rel=0.02, vmin=None, vmax=None):
    ax.set_axis_off()
    if background_img is not None:
        bg = plt.imread(background_img)
        ax.imshow(bg, extent=[0, screen_w, screen_h, 0], aspect="auto")
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
    else:
        ax.imshow(H.T, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    if title:
        ax.set_title(title, pad=TITLE_PAD)


def save_density_png(H: np.ndarray, out_png: Path, title: str, cmap, figsize=DEFAULT_FIGSIZE, vmin=None, vmax=None):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    _draw_density(ax, H, cmap=cmap, title=title, vmin=vmin, vmax=vmax)
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
    alpha: float = 0.62,
    thresh_rel: float = 0.02,
    figsize=DEFAULT_FIGSIZE,
    vmin=None,
    vmax=None,
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
    alpha: float = 0.62,
    thresh_rel: float = 0.02,
):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    vmax = max(float(high.max()), float(low.max()), 1e-12)
    overlap = np.minimum(high, low)
    ovmax = max(float(overlap.max()), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(COMPARE_PANEL_SIZE[0] * 3, COMPARE_PANEL_SIZE[1]))
    _draw_density(
        axes[0],
        high,
        cmap=cmap,
        title=label_high,
        background_img=background_img,
        screen_w=screen_w,
        screen_h=screen_h,
        alpha=alpha,
        thresh_rel=thresh_rel,
        vmin=0,
        vmax=vmax,
    )
    _draw_density(
        axes[1],
        low,
        cmap=cmap,
        title=label_low,
        background_img=background_img,
        screen_w=screen_w,
        screen_h=screen_h,
        alpha=alpha,
        thresh_rel=thresh_rel,
        vmin=0,
        vmax=vmax,
    )
    _draw_density(
        axes[2],
        overlap,
        cmap=plt.get_cmap("Greens"),
        title="Shared attention pattern",
        background_img=background_img,
        screen_w=screen_w,
        screen_h=screen_h,
        alpha=alpha,
        thresh_rel=thresh_rel,
        vmin=0,
        vmax=ovmax,
    )
    if title:
        fig.suptitle(title, y=1.02, fontsize=12)
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
    ap.add_argument("--sigma", type=float, default=2.2, help="Gaussian smoothing sigma (default 2.2)")

    ap.add_argument("--point_source", default="gaze", choices=["gaze", "fixation"], help="Use gaze points or fixation points")
    ap.add_argument("--weight", default="none", choices=["none", "fixation_duration"], help="Optional weighting")
    ap.add_argument("--fixation_dedup", default="index", choices=["index", "none"], help="How to deduplicate fixation points")

    ap.add_argument("--require_validity", action="store_true", help="Require Validity Left/Right == 1 when columns exist")
    ap.add_argument("--columns_map", default=None, help="Path to JSON mapping of required columns to candidate names")

    ap.add_argument("--background_img", default=None, help="Optional background image (png/jpg) to overlay all heatmaps")
    ap.add_argument("--alpha", type=float, default=0.62, help="Overlay alpha (default 0.62)")
    ap.add_argument("--thresh", type=float, default=0.02, help="Relative threshold in [0,1] to hide low-density tails")
    ap.add_argument("--cmap", default="tobii", help="Heatmap colormap (default: tobii -> turbo/jet)")

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

    outdir = Path(args.outdir)
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
            raise FileExistsError(f"Multiple CSVs matched id={pid!r} under {csv_dir}: {[str(p) for p in matches[:10]]}")
        return str(matches[0])

    points_all = []
    points_exp = {"High": [], "Low": []}
    rows = []
    errors = []

    for _, r in m.iterrows():
        pid = str(r[id_col]).strip()
        ex = norm_level(r["Experience"])

        one_out = outdir / "individual" / pid
        one_out.mkdir(parents=True, exist_ok=True)

        cache_tag = f"{args.point_source}"
        if args.point_source == "fixation":
            cache_tag += f"_dedup-{args.fixation_dedup}"
        if args.weight != "none":
            cache_tag += f"_w-{args.weight}"

        points_path = one_out / f"points_{cache_tag}.npy"
        weights_path = one_out / f"weights_{cache_tag}.npy"
        meta_path = one_out / f"meta_{cache_tag}.json"

        pid_title = fmt_title(pid)

        try:
            if args.resume and points_path.exists():
                xy = np.load(points_path)
                w = np.load(weights_path) if weights_path.exists() else None
                source_csv = None
            else:
                w = None
                csv_path = resolve_csv_path(pid, r["csv_path"] if has_csv_path else None)
                df, clean = load_and_clean(
                    csv_path,
                    screen_w=args.screen_w,
                    screen_h=args.screen_h,
                    require_validity=args.require_validity,
                    columns_map_path=args.columns_map,
                )

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
                    np.save(weights_path, w.astype(np.float32))
                else:
                    w = None

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
                source_csv = csv_path

            H = density_from_points(xy, args.screen_w, args.screen_h, bins=args.bins, sigma=args.sigma, weights=w)
            save_density_png(H, one_out / "heatmap_density.png", fmt_title(f"Participant density: {pid_title}"), cmap)

            if args.background_img:
                save_density_overlay(
                    H,
                    one_out / "heatmap.png",
                    fmt_title(f"Participant: {pid_title}"),
                    args.background_img,
                    args.screen_w,
                    args.screen_h,
                    cmap=cmap,
                    alpha=args.alpha,
                    thresh_rel=args.thresh,
                )
                save_density_overlay(
                    H,
                    one_out / "heatmap_overlay.png",
                    fmt_title(f"Participant overlay: {pid_title}"),
                    args.background_img,
                    args.screen_w,
                    args.screen_h,
                    cmap=cmap,
                    alpha=args.alpha,
                    thresh_rel=args.thresh,
                )
            else:
                save_density_png(H, one_out / "heatmap.png", fmt_title(f"Participant density: {pid_title}"), cmap)

            rows.append(
                {
                    "id": pid,
                    "Experience": ex,
                    "csv_path": source_csv,
                    "point_source": args.point_source,
                    "weight": args.weight,
                    "fixation_dedup": args.fixation_dedup if args.point_source == "fixation" else "",
                    "n_points": int(xy.shape[0]),
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

    overall_xy, overall_w = concat_list(points_all)
    overall_H = density_from_points(overall_xy, args.screen_w, args.screen_h, bins=args.bins, sigma=args.sigma, weights=overall_w)
    overall_dir = outdir / "groups" / "Overall"
    overall_dir.mkdir(parents=True, exist_ok=True)
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
