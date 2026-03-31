#!/usr/bin/env python3
"""Compose a 3×2 figure panel from per-scene Overall heatmap outputs.

This script reads pre-generated heatmap images (e.g., from batch_heatmap_groups_scenes.py)
for 6 scenes and composites them into one publication figure matching the layout:

  Row 1:  (a) C0–WWR15    (b) C0–WWR45    (c) C0–WWR75
  Row 2:  (d) C1–WWR15    (e) C1–WWR45    (f) C1–WWR75

The individual heatmap images are NOT re-rendered; they are loaded as-is (PNG/JPG).
Black borders from the original outputs are auto-cropped.

Usage
-----
python scripts/compose_scene_panel.py \
    --scene_dirs outputs_by_scene \
    --outfile figures/Fig_heatmap_6scenes.png

The script expects the output tree from batch_heatmap_groups_scenes.py:
  outputs_by_scene/<scene_folder>/groups/Overall/heatmap.png

Scene-to-position mapping is configured via --scene_order (JSON) or auto-detected
from folder names containing WWR/C0/C1 patterns.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


# -- Default scene layout (matches reference figure) --------------------------
DEFAULT_SCENE_ORDER = [
    # (row, col, panel_label, display_label, folder_name_pattern)
    (0, 0, "(a)", "C0\u2013WWR15", r"C0.*W(?:WR)?15"),
    (0, 1, "(b)", "C0\u2013WWR45", r"C0.*W(?:WR)?45"),
    (0, 2, "(c)", "C0\u2013WWR75", r"C0.*W(?:WR)?75"),
    (1, 0, "(d)", "C1\u2013WWR15", r"C1.*W(?:WR)?15"),
    (1, 1, "(e)", "C1\u2013WWR45", r"C1.*W(?:WR)?45"),
    (1, 2, "(f)", "C1\u2013WWR75", r"C1.*W(?:WR)?75"),
]

PLOT_DPI = 600
FIG_BG = "white"
LABEL_COLOR = "#202124"
LABEL_SIZE = 11


def _auto_crop(img_arr: np.ndarray, bg_thresh: int = 18) -> np.ndarray:
    """Remove near-black borders from an image array (H, W, C)."""
    if img_arr.ndim == 2:
        gray = img_arr
    else:
        gray = img_arr[..., :3].mean(axis=-1)

    if gray.max() > 1.5:
        # uint8-scale
        mask = gray > bg_thresh
    else:
        mask = gray > (bg_thresh / 255.0)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return img_arr

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img_arr[rmin : rmax + 1, cmin : cmax + 1]


def _load_image(path: Path) -> np.ndarray:
    """Load image as float32 numpy array."""
    if Image is not None:
        img = Image.open(path).convert("RGB")
        return np.asarray(img, dtype=np.float32) / 255.0
    else:
        return plt.imread(str(path))


def _find_heatmap(scene_dir: Path, prefer: str = "heatmap.png") -> Path | None:
    """Locate the Overall group heatmap image inside a scene output directory."""
    candidates = [
        scene_dir / "groups" / "Overall" / prefer,
        scene_dir / "groups" / "Overall" / "heatmap_overlay.png",
        scene_dir / "groups" / "Overall" / "heatmap_density.png",
        # fallback: any heatmap.png at top level
        scene_dir / "heatmap.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _match_scenes(scene_root: Path, order_spec):
    """Map scene folders to grid positions using regex patterns."""
    scene_dirs = sorted([d for d in scene_root.iterdir() if d.is_dir()])
    mapping = {}  # (row, col) -> (panel_label, display_label, heatmap_path)

    for row, col, panel_label, display_label, pattern in order_spec:
        regex = re.compile(pattern, re.IGNORECASE)
        matched = None
        for sd in scene_dirs:
            if regex.search(sd.name):
                hm = _find_heatmap(sd)
                if hm:
                    matched = (panel_label, display_label, hm)
                    break
        if matched is None:
            print(f"[WARN] No scene matched pattern {pattern!r} — panel {panel_label} {display_label} will be empty", file=sys.stderr)
        mapping[(row, col)] = matched

    return mapping


def main():
    ap = argparse.ArgumentParser(description="Compose 3×2 scene heatmap panel for manuscript submission")
    ap.add_argument("--scene_dirs", required=True, help="Root directory of per-scene outputs (from batch_heatmap_groups_scenes.py)")
    ap.add_argument("--outfile", default="figures/Fig_heatmap_6scenes.png", help="Output figure path")
    ap.add_argument("--scene_order", default=None, help="Optional JSON file overriding scene-to-grid mapping")
    ap.add_argument("--title", default=None, help="Optional figure super-title")
    ap.add_argument("--note", default=None, help="Optional figure note (placed below panels)")
    ap.add_argument("--crop", action="store_true", default=True, help="Auto-crop black borders (default: on)")
    ap.add_argument("--no_crop", action="store_false", dest="crop", help="Disable auto-crop")
    ap.add_argument("--dpi", type=int, default=PLOT_DPI)
    ap.add_argument("--panel_w", type=float, default=4.8, help="Width per panel in inches")
    ap.add_argument("--panel_h", type=float, default=5.6, help="Height per panel in inches")
    ap.add_argument("--wspace", type=float, default=0.04, help="Horizontal gap between panels (fraction)")
    ap.add_argument("--hspace", type=float, default=0.12, help="Vertical gap between rows (fraction)")

    args = ap.parse_args()

    scene_root = Path(args.scene_dirs)
    if not scene_root.exists():
        raise SystemExit(f"scene_dirs not found: {scene_root}")

    if args.scene_order:
        with open(args.scene_order, "r", encoding="utf-8") as f:
            order_spec = json.load(f)
    else:
        order_spec = DEFAULT_SCENE_ORDER

    mapping = _match_scenes(scene_root, order_spec)

    ncols, nrows = 3, 2
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(args.panel_w * ncols, args.panel_h * nrows),
        facecolor=FIG_BG,
    )
    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.92 if args.title else 0.96,
        bottom=0.08 if args.note else 0.03,
        wspace=args.wspace,
        hspace=args.hspace,
    )

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            entry = mapping.get((r, c))
            if entry is None:
                ax.set_facecolor(FIG_BG)
                continue

            panel_label, display_label, hm_path = entry
            img = _load_image(hm_path)
            if args.crop:
                img = _auto_crop(img)

            ax.imshow(img, aspect="auto")
            ax.set_facecolor(FIG_BG)

            # Panel label below the image
            ax.text(
                0.5,
                -0.04,
                f"{panel_label}  {display_label}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=LABEL_SIZE,
                color=LABEL_COLOR,
            )

    if args.title:
        fig.suptitle(args.title, fontsize=14, color=LABEL_COLOR, fontweight="bold", y=0.97)

    if args.note:
        fig.text(
            0.5,
            0.015,
            args.note,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#667085",
            fontstyle="italic",
            wrap=True,
        )

    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=args.dpi, facecolor=FIG_BG)
    plt.close(fig)
    print(f"Saved: {outfile}")


if __name__ == "__main__":
    main()
