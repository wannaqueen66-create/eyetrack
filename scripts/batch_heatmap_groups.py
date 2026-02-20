#!/usr/bin/env python3
"""Batch heatmaps with group aggregation + difference plots + background overlays.

Designed for people/group comparisons:
  - Individual heatmap per participant
  - Aggregated heatmap per group (SportFreq, Experience, and/or 4-way cross)
  - Difference plots for binary splits (High vs Low)

This version supports:
  - Background overlay for ALL outputs when --background_img is provided
    (individual / group / compare)
  - Tobii-like colormap (default --cmap tobii)
  - Resume/caching (saves per-participant points.npy)
  - Keep-going (default): logs errors to errors.csv and continues

Manifest CSV (recommended columns):
  - name (string) OR participant_id (string)
  - SportFreq (High/Low)  # case-insensitive
  - Experience (High/Low) # case-insensitive
  - csv_path (path, optional if you pass --csv_dir)

Typical Colab usage:
  python scripts/batch_heatmap_groups.py \
    --manifest /content/group_manifest.csv \
    --csv_dir /content/csv \
    --screen_w 1748 --screen_h 2064 \
    --background_img /content/scene.png \
    --outdir /content/outputs_batch_groups
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import warnings

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import load_and_clean

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


def setup_cjk_font(prefer_names=None):
    """Best-effort configure a CJK-capable font to avoid 'Glyph missing' warnings.

    On Colab/Linux, you can install one via:
      apt-get install -y fonts-noto-cjk

    If no preferred font is found, we keep defaults (warnings may appear, but plots still save).
    """
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


def get_cmap(name: str):
    """Return a matplotlib colormap.

    Default 'tobii' uses a blue→green→yellow→red ramp similar to common eye-tracking tools.
    We prefer 'turbo' (better than jet), fallback to 'jet'.
    """
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
    bins: int = 200,
    sigma: float = 2.0,
):
    """Return a normalized 2D density image (bins x bins) in screen coordinate space."""
    if xy.size == 0:
        return np.zeros((bins, bins), dtype=float)

    x = np.clip(xy[:, 0], 0, screen_w)
    y = np.clip(xy[:, 1], 0, screen_h)

    H, _, _ = np.histogram2d(
        x,
        y,
        bins=[bins, bins],
        range=[[0, screen_w], [0, screen_h]],
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
    # emphasize hotspots, fade tails
    A = (Hn ** 0.6) * float(alpha)
    if thresh_rel is not None and thresh_rel > 0:
        A = np.where(Hn >= float(thresh_rel), A, 0.0)
    return A


def save_density_png(H: np.ndarray, out_png: Path, title: str, cmap, vmin=None, vmax=None):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 8))
    plt.imshow(H.T, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def save_density_overlay(
    H: np.ndarray,
    out_png: Path,
    title: str,
    background_img: str,
    screen_w: int,
    screen_h: int,
    cmap,
    alpha: float = 0.55,
    thresh_rel: float = 0.02,
    vmin=None,
    vmax=None,
):
    """Overlay density heatmap on a background image (scene).

    The background is stretched to [0..screen_w]x[0..screen_h].
    """
    bg = plt.imread(background_img)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    A = _alpha_map_from_density(H, alpha=alpha, thresh_rel=thresh_rel)

    plt.figure(figsize=(6, 8))
    plt.imshow(bg, extent=[0, screen_w, screen_h, 0], aspect="auto")
    plt.imshow(
        H.T,
        origin="upper",
        extent=[0, screen_w, screen_h, 0],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=A.T if A is not None else alpha,
        aspect="auto",
    )
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def save_binary_compare(
    A: np.ndarray,
    B: np.ndarray,
    out_png: Path,
    title: str,
    cmap,
    background_img: str | None = None,
    screen_w: int | None = None,
    screen_h: int | None = None,
    alpha: float = 0.55,
    thresh_rel: float = 0.02,
    eps: float = 1e-12,
    layout: int = 4,
    overlap_mode: str = "min",
    overlap_cmap = None,
    composite: bool = False,
    reference: str = "overlap",
):
    """Save compare figure.

    layout modes:
      - 3: A, B, log2(A/B)
      - 4: A, B, overlap(A,B), log2(A/B)
      - 2 (composite): composite_diff, reference
      - 3 (composite): A, B, composite_diff

    composite_diff shows log2(A/B) as color (red/blue) with overlap strength as alpha.

    overlap_mode:
      - 'min'     : overlap = min(A,B) (common attention)
      - 'product' : overlap = sqrt(A*B) (emphasize shared hotspots)

    reference:
      - 'overlap' : show overlap heatmap
      - 'mean'    : show mean heatmap (A+B)/2
    """

    out_png.parent.mkdir(parents=True, exist_ok=True)

    vmax = max(float(A.max()), float(B.max()), eps)

    L = np.log2((A + eps) / (B + eps))
    lim = float(np.nanmax(np.abs(L))) if np.isfinite(L).any() else 1.0
    lim = max(lim, 1e-6)

    # overlap
    om = (overlap_mode or "min").strip().lower()
    if om == "product":
        O = np.sqrt((A + eps) * (B + eps))
    else:
        O = np.minimum(A, B)

    if overlap_cmap is None:
        overlap_cmap = plt.get_cmap("Greens")

    bg = None
    if background_img:
        bg = plt.imread(background_img)

    # Decide layout
    if composite:
        if layout not in (2, 3):
            layout = 2
    else:
        if layout not in (3, 4):
            layout = 4

    fig, axes = plt.subplots(1, layout, figsize=(5 * layout, 6))
    if layout == 1:
        axes = [axes]
    for ax in axes:
        ax.axis("off")

    def draw_panel(ax, H, title_text, cmap_use, vmin, vmax, alpha_map=None):
        if bg is not None:
            ax.imshow(bg, extent=[0, screen_w, screen_h, 0], aspect="auto")
            ax.imshow(
                H.T,
                origin="upper",
                extent=[0, screen_w, screen_h, 0],
                cmap=cmap_use,
                vmin=vmin,
                vmax=vmax,
                alpha=alpha_map.T if alpha_map is not None else alpha,
                aspect="auto",
            )
        else:
            ax.imshow(H.T, origin="upper", cmap=cmap_use, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title_text)

    A_alpha = _alpha_map_from_density(A, alpha=alpha, thresh_rel=thresh_rel)
    B_alpha = _alpha_map_from_density(B, alpha=alpha, thresh_rel=thresh_rel)

    def draw_composite(ax, title_text: str):
        # composite: color=log2(A/B), alpha=overlap strength
        Ln = np.clip(np.abs(L) / lim, 0, 1)
        # weight by overlap (shared attention)
        On = O / max(float(O.max()), eps) if float(O.max()) > 0 else O
        Acomp = (Ln ** 0.6) * (On ** 0.7) * float(alpha)
        if thresh_rel is not None and thresh_rel > 0:
            Acomp = np.where(On >= float(thresh_rel), Acomp, 0.0)

        if bg is not None:
            ax.imshow(bg, extent=[0, screen_w, screen_h, 0], aspect="auto")
            ax.imshow(
                L.T,
                origin="upper",
                extent=[0, screen_w, screen_h, 0],
                cmap=plt.get_cmap("RdBu_r"),
                vmin=-lim,
                vmax=lim,
                alpha=Acomp.T,
                aspect="auto",
            )
        else:
            ax.imshow(L.T, origin="upper", cmap=plt.get_cmap("RdBu_r"), vmin=-lim, vmax=lim, alpha=Acomp.T, aspect="auto")
        ax.set_title(title_text)

    def draw_reference(ax, title_text: str):
        ref = (reference or "overlap").strip().lower()
        if ref == "mean":
            R = (A + B) / 2.0
            R_alpha = _alpha_map_from_density(R, alpha=alpha, thresh_rel=thresh_rel)
            draw_panel(ax, R, title_text, cmap, 0, vmax, R_alpha)
        else:
            O_alpha = _alpha_map_from_density(O, alpha=alpha, thresh_rel=thresh_rel)
            draw_panel(ax, O, title_text, overlap_cmap, 0, max(float(O.max()), eps), O_alpha)

    if composite:
        # Composite outputs
        if layout == 2:
            draw_composite(axes[0], "Composite diff (overlap-weighted log2)")
            draw_reference(axes[1], "Reference")
        else:
            # 3-panel composite: A, B, Composite
            draw_panel(axes[0], A, "Group A", cmap, 0, vmax, A_alpha)
            draw_panel(axes[1], B, "Group B", cmap, 0, vmax, B_alpha)
            draw_composite(axes[2], "Composite diff")
    else:
        # Traditional outputs
        draw_panel(axes[0], A, "Group A", cmap, 0, vmax, A_alpha)
        draw_panel(axes[1], B, "Group B", cmap, 0, vmax, B_alpha)

        col = 2
        if layout == 4:
            O_alpha = _alpha_map_from_density(O, alpha=alpha, thresh_rel=thresh_rel)
            draw_panel(axes[col], O, "Overlap", overlap_cmap, 0, max(float(O.max()), eps), O_alpha)
            col += 1

        Ln = np.clip(np.abs(L) / lim, 0, 1)
        L_alpha = (Ln ** 0.7) * float(alpha)
        if thresh_rel is not None and thresh_rel > 0:
            L_alpha = np.where(Ln >= float(thresh_rel), L_alpha, 0.0)

        draw_panel(axes[col], L, "log2(A/B)", plt.get_cmap("RdBu_r"), -lim, lim, L_alpha)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Batch heatmaps by participant + group aggregation + difference plots")
    ap.add_argument("--manifest", required=True, help="CSV with columns: name/participant_id,SportFreq,Experience,(optional)csv_path")
    ap.add_argument("--csv_dir", default=None, help="If csv_path is omitted in manifest, find CSVs under this directory")
    ap.add_argument("--outdir", default="outputs_batch_groups")

    ap.add_argument("--screen_w", type=int, default=1280)
    ap.add_argument("--screen_h", type=int, default=1440)

    ap.add_argument("--bins", type=int, default=220, help="Grid bins for density (default 220x220)")
    ap.add_argument("--sigma", type=float, default=2.2, help="Gaussian smoothing sigma (default 2.2)")

    ap.add_argument("--require_validity", action="store_true", help="Require Validity Left/Right == 1 (if columns exist)")
    ap.add_argument("--columns_map", default=None, help="Path to JSON mapping of required columns to candidate names")

    # Background overlay
    ap.add_argument("--background_img", default=None, help="Optional background image (png/jpg) to overlay ALL heatmaps")
    ap.add_argument("--alpha", type=float, default=0.60, help="Overlay alpha (default 0.60)")
    ap.add_argument("--thresh", type=float, default=0.02, help="Relative threshold in [0,1] to hide low-density tails (default 0.02)")

    # Colors
    ap.add_argument("--cmap", default="tobii", help="Heatmap colormap (default: tobii -> turbo/jet).")
    ap.add_argument("--compare_layout", type=int, default=4, choices=[3, 4], help="Compare figure layout: 3 panels (A,B,log2) or 4 panels (A,B,overlap,log2). Default 4")
    ap.add_argument("--overlap_mode", default="min", choices=["min", "product"], help="How to compute overlap heatmap: min(A,B) or sqrt(A*B). Default min")
    ap.add_argument("--compare_composite", action="store_true", help="Use composite diff (log2 color weighted by overlap alpha)")
    ap.add_argument("--compare_reference", default="overlap", choices=["overlap", "mean"], help="Reference panel for composite layout (default overlap)")
    ap.add_argument("--font", default="auto", help="Font for titles. Use 'auto' to pick a CJK font if available (default: auto)")
    ap.add_argument("--quiet_glyph_warning", action="store_true", help="Suppress matplotlib 'Glyph missing' warnings (does not fix rendering)")

    # Resume / fault tolerance
    ap.add_argument("--resume", action="store_true", default=True, help="Resume using cached per-participant points.npy if present (default: on)")
    ap.add_argument("--no_resume", action="store_false", dest="resume", help="Disable resume cache")
    ap.add_argument("--fail_fast", action="store_true", help="Stop on first error (default: keep-going and write errors.csv)")

    args = ap.parse_args()

    cmap = get_cmap(args.cmap)

    # Font setup for Chinese names in titles
    if args.font and str(args.font).lower() != "auto":
        mpl.rcParams["font.sans-serif"] = [args.font, "DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    else:
        setup_cjk_font()

    if args.quiet_glyph_warning:
        warnings.filterwarnings("ignore", message=r"Glyph .* missing from font\(s\) .*", category=UserWarning)

    outdir = Path(args.outdir)
    (outdir / "individual").mkdir(parents=True, exist_ok=True)
    (outdir / "groups").mkdir(parents=True, exist_ok=True)
    (outdir / "compare").mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(args.manifest)

    # Accept either participant_id or name
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

    points_sport = {"High": [], "Low": []}
    points_exp = {"High": [], "Low": []}
    points_4 = {
        "Experience-High×SportFreq-High": [],
        "Experience-High×SportFreq-Low": [],
        "Experience-Low×SportFreq-High": [],
        "Experience-Low×SportFreq-Low": [],
    }

    rows = []
    errors = []

    for _, r in m.iterrows():
        pid = str(r[id_col]).strip()
        sf = norm_level(r["SportFreq"])  # High/Low
        ex = norm_level(r["Experience"])  # High/Low

        one_out = outdir / "individual" / pid
        one_out.mkdir(parents=True, exist_ok=True)
        points_path = one_out / "points.npy"
        meta_path = one_out / "meta.json"

        try:
            if args.resume and points_path.exists():
                xy = np.load(points_path)
                source_csv = None
            else:
                csv_path = resolve_csv_path(pid, r["csv_path"] if has_csv_path else None)
                df, clean = load_and_clean(
                    csv_path,
                    screen_w=args.screen_w,
                    screen_h=args.screen_h,
                    require_validity=args.require_validity,
                    columns_map_path=args.columns_map,
                )
                xy = clean[["Gaze Point X[px]", "Gaze Point Y[px]"]].dropna().to_numpy(dtype=np.float32)
                np.save(points_path, xy)
                meta_path.write_text(
                    json.dumps(
                        {
                            "id": pid,
                            "csv_path": csv_path,
                            "SportFreq": sf,
                            "Experience": ex,
                            "n_samples_after_clean": int(xy.shape[0]),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                source_csv = csv_path

            # Individual outputs: always produce density + (optional) overlay
            H = density_from_points(xy, args.screen_w, args.screen_h, bins=args.bins, sigma=args.sigma)
            save_density_png(H, one_out / "heatmap_density.png", f"{pid} (density)", cmap)

            if args.background_img:
                # canonical: heatmap.png is overlay when background is provided
                save_density_overlay(
                    H,
                    one_out / "heatmap.png",
                    f"{pid}",
                    args.background_img,
                    args.screen_w,
                    args.screen_h,
                    cmap=cmap,
                    alpha=args.alpha,
                    thresh_rel=args.thresh,
                )
                # also keep explicit name
                save_density_overlay(
                    H,
                    one_out / "heatmap_overlay.png",
                    f"{pid} (overlay)",
                    args.background_img,
                    args.screen_w,
                    args.screen_h,
                    cmap=cmap,
                    alpha=args.alpha,
                    thresh_rel=args.thresh,
                )
            else:
                # fallback: no background, heatmap.png is density
                save_density_png(H, one_out / "heatmap.png", f"{pid} (density)", cmap)

            rows.append(
                {
                    "id": pid,
                    "SportFreq": sf,
                    "Experience": ex,
                    "csv_path": source_csv,
                    "n_samples": int(xy.shape[0]),
                    "status": "ok",
                }
            )

            if sf in points_sport:
                points_sport[sf].append(xy)
            if ex in points_exp:
                points_exp[ex].append(xy)
            key4 = f"Experience-{ex}×SportFreq-{sf}"
            if key4 in points_4:
                points_4[key4].append(xy)

        except Exception as e:
            err = {"id": pid, "SportFreq": sf, "Experience": ex, "error": repr(e)}
            errors.append(err)
            rows.append({"id": pid, "SportFreq": sf, "Experience": ex, "csv_path": None, "n_samples": 0, "status": "error", "error": repr(e)})
            if args.fail_fast:
                raise

    pd.DataFrame(rows).to_csv(outdir / "participants_summary.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(outdir / "errors.csv", index=False)

    def concat_list(lst):
        return np.concatenate(lst, axis=0) if len(lst) and any(x.size for x in lst) else np.zeros((0, 2), dtype=np.float32)

    # ---- Group aggregated heatmaps ----
    dens_sf = {}
    for level in ["High", "Low"]:
        xy = concat_list(points_sport[level])
        H = density_from_points(xy, args.screen_w, args.screen_h, bins=args.bins, sigma=args.sigma)
        dens_sf[level] = H
        base = outdir / "groups" / f"SportFreq-{level}"
        base.mkdir(parents=True, exist_ok=True)
        save_density_png(H, base / "heatmap_density.png", f"SportFreq-{level} (density)", cmap)
        if args.background_img:
            save_density_overlay(H, base / "heatmap.png", f"SportFreq-{level}", args.background_img, args.screen_w, args.screen_h, cmap=cmap, alpha=args.alpha, thresh_rel=args.thresh)
            save_density_overlay(H, base / "heatmap_overlay.png", f"SportFreq-{level} (overlay)", args.background_img, args.screen_w, args.screen_h, cmap=cmap, alpha=args.alpha, thresh_rel=args.thresh)
        else:
            save_density_png(H, base / "heatmap.png", f"SportFreq-{level} (density)", cmap)

    dens_ex = {}
    for level in ["High", "Low"]:
        xy = concat_list(points_exp[level])
        H = density_from_points(xy, args.screen_w, args.screen_h, bins=args.bins, sigma=args.sigma)
        dens_ex[level] = H
        base = outdir / "groups" / f"Experience-{level}"
        base.mkdir(parents=True, exist_ok=True)
        save_density_png(H, base / "heatmap_density.png", f"Experience-{level} (density)", cmap)
        if args.background_img:
            save_density_overlay(H, base / "heatmap.png", f"Experience-{level}", args.background_img, args.screen_w, args.screen_h, cmap=cmap, alpha=args.alpha, thresh_rel=args.thresh)
            save_density_overlay(H, base / "heatmap_overlay.png", f"Experience-{level} (overlay)", args.background_img, args.screen_w, args.screen_h, cmap=cmap, alpha=args.alpha, thresh_rel=args.thresh)
        else:
            save_density_png(H, base / "heatmap.png", f"Experience-{level} (density)", cmap)

    dens_4 = {}
    for k in points_4.keys():
        xy = concat_list(points_4[k])
        H = density_from_points(xy, args.screen_w, args.screen_h, bins=args.bins, sigma=args.sigma)
        dens_4[k] = H
        base = outdir / "groups" / "4way" / k
        base.mkdir(parents=True, exist_ok=True)
        save_density_png(H, base / "heatmap_density.png", f"{k} (density)", cmap)
        if args.background_img:
            save_density_overlay(H, base / "heatmap.png", k, args.background_img, args.screen_w, args.screen_h, cmap=cmap, alpha=args.alpha, thresh_rel=args.thresh)
            save_density_overlay(H, base / "heatmap_overlay.png", f"{k} (overlay)", args.background_img, args.screen_w, args.screen_h, cmap=cmap, alpha=args.alpha, thresh_rel=args.thresh)
        else:
            save_density_png(H, base / "heatmap.png", f"{k} (density)", cmap)

    # ---- Compare plots (include background if provided) ----
    save_binary_compare(
        dens_sf["High"],
        dens_sf["Low"],
        outdir / "compare" / "SportFreq_diff.png",
        title="SportFreq: High vs Low (density + log2 ratio)",
        cmap=cmap,
        background_img=args.background_img,
        screen_w=args.screen_w,
        screen_h=args.screen_h,
        alpha=args.alpha,
        thresh_rel=args.thresh,
        layout=args.compare_layout,
        overlap_mode=args.overlap_mode,
    )

    # Additional composite views (requested):
    #  - 3-panel: A / B / Composite
    #  - 2-panel: Composite + Reference
    save_binary_compare(
        dens_sf["High"],
        dens_sf["Low"],
        outdir / "compare" / "SportFreq_composite_3panel.png",
        title="SportFreq: High vs Low (A/B + composite diff)",
        cmap=cmap,
        background_img=args.background_img,
        screen_w=args.screen_w,
        screen_h=args.screen_h,
        alpha=args.alpha,
        thresh_rel=args.thresh,
        layout=3,
        overlap_mode=args.overlap_mode,
        composite=True,
        reference=args.compare_reference,
    )
    save_binary_compare(
        dens_sf["High"],
        dens_sf["Low"],
        outdir / "compare" / "SportFreq_composite_2panel.png",
        title="SportFreq: High vs Low (composite + reference)",
        cmap=cmap,
        background_img=args.background_img,
        screen_w=args.screen_w,
        screen_h=args.screen_h,
        alpha=args.alpha,
        thresh_rel=args.thresh,
        layout=2,
        overlap_mode=args.overlap_mode,
        composite=True,
        reference=args.compare_reference,
    )
    save_binary_compare(
        dens_ex["High"],
        dens_ex["Low"],
        outdir / "compare" / "Experience_diff.png",
        title="Experience: High vs Low (density + log2 ratio)",
        cmap=cmap,
        background_img=args.background_img,
        screen_w=args.screen_w,
        screen_h=args.screen_h,
        alpha=args.alpha,
        thresh_rel=args.thresh,
        layout=args.compare_layout,
        overlap_mode=args.overlap_mode,
    )

    save_binary_compare(
        dens_ex["High"],
        dens_ex["Low"],
        outdir / "compare" / "Experience_composite_3panel.png",
        title="Experience: High vs Low (A/B + composite diff)",
        cmap=cmap,
        background_img=args.background_img,
        screen_w=args.screen_w,
        screen_h=args.screen_h,
        alpha=args.alpha,
        thresh_rel=args.thresh,
        layout=3,
        overlap_mode=args.overlap_mode,
        composite=True,
        reference=args.compare_reference,
    )
    save_binary_compare(
        dens_ex["High"],
        dens_ex["Low"],
        outdir / "compare" / "Experience_composite_2panel.png",
        title="Experience: High vs Low (composite + reference)",
        cmap=cmap,
        background_img=args.background_img,
        screen_w=args.screen_w,
        screen_h=args.screen_h,
        alpha=args.alpha,
        thresh_rel=args.thresh,
        layout=2,
        overlap_mode=args.overlap_mode,
        composite=True,
        reference=args.compare_reference,
    )

    # ---- 4-way grid (overlay background if provided) ----
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    order = [
        ["Experience-High×SportFreq-High", "Experience-High×SportFreq-Low"],
        ["Experience-Low×SportFreq-High", "Experience-Low×SportFreq-Low"],
    ]

    bg = None
    if args.background_img:
        bg = plt.imread(args.background_img)

    vmax = max(float(H.max()) for H in dens_4.values()) if dens_4 else 1.0
    vmax = max(vmax, 1e-12)

    for i in range(2):
        for j in range(2):
            k = order[i][j]
            ax = axes[i, j]
            ax.axis("off")
            if bg is not None:
                ax.imshow(bg, extent=[0, args.screen_w, args.screen_h, 0], aspect="auto")
                Aalp = _alpha_map_from_density(dens_4[k], alpha=args.alpha, thresh_rel=args.thresh)
                ax.imshow(
                    dens_4[k].T,
                    origin="upper",
                    extent=[0, args.screen_w, args.screen_h, 0],
                    cmap=cmap,
                    vmin=0,
                    vmax=vmax,
                    alpha=Aalp.T if Aalp is not None else args.alpha,
                    aspect="auto",
                )
            else:
                ax.imshow(dens_4[k].T, origin="upper", cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
            ax.set_title(k)

    fig.suptitle("4-way groups (shared scale)")
    fig.tight_layout()
    fig.savefig(outdir / "compare" / "4way_grid.png", dpi=300)
    plt.close(fig)

    # ---- 4-way stratified composite diffs (方案 2) ----
    # SportFreq within Experience-High / Experience-Low
    EH_H = dens_4["Experience-High×SportFreq-High"]
    EH_L = dens_4["Experience-High×SportFreq-Low"]
    EL_H = dens_4["Experience-Low×SportFreq-High"]
    EL_L = dens_4["Experience-Low×SportFreq-Low"]

    save_binary_compare(
        EH_H,
        EH_L,
        outdir / "compare" / "4way_SportFreq_within_ExperienceHigh_composite_2panel.png",
        title="SportFreq diff within Experience=High (composite + reference)",
        cmap=cmap,
        background_img=args.background_img,
        screen_w=args.screen_w,
        screen_h=args.screen_h,
        alpha=args.alpha,
        thresh_rel=args.thresh,
        layout=2,
        overlap_mode=args.overlap_mode,
        composite=True,
        reference=args.compare_reference,
    )
    save_binary_compare(
        EL_H,
        EL_L,
        outdir / "compare" / "4way_SportFreq_within_ExperienceLow_composite_2panel.png",
        title="SportFreq diff within Experience=Low (composite + reference)",
        cmap=cmap,
        background_img=args.background_img,
        screen_w=args.screen_w,
        screen_h=args.screen_h,
        alpha=args.alpha,
        thresh_rel=args.thresh,
        layout=2,
        overlap_mode=args.overlap_mode,
        composite=True,
        reference=args.compare_reference,
    )

    # Experience within SportFreq-High / SportFreq-Low
    save_binary_compare(
        EH_H,
        EL_H,
        outdir / "compare" / "4way_Experience_within_SportFreqHigh_composite_2panel.png",
        title="Experience diff within SportFreq=High (composite + reference)",
        cmap=cmap,
        background_img=args.background_img,
        screen_w=args.screen_w,
        screen_h=args.screen_h,
        alpha=args.alpha,
        thresh_rel=args.thresh,
        layout=2,
        overlap_mode=args.overlap_mode,
        composite=True,
        reference=args.compare_reference,
    )
    save_binary_compare(
        EH_L,
        EL_L,
        outdir / "compare" / "4way_Experience_within_SportFreqLow_composite_2panel.png",
        title="Experience diff within SportFreq=Low (composite + reference)",
        cmap=cmap,
        background_img=args.background_img,
        screen_w=args.screen_w,
        screen_h=args.screen_h,
        alpha=args.alpha,
        thresh_rel=args.thresh,
        layout=2,
        overlap_mode=args.overlap_mode,
        composite=True,
        reference=args.compare_reference,
    )

    print("Done. Outputs in:", str(outdir))
    if errors:
        print(f"WARNING: {len(errors)} participants had errors. See: {outdir / 'errors.csv'}")


if __name__ == "__main__":
    main()
