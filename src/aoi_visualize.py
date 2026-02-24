"""AOI visualization helpers.

Goal: export publication-friendly AOI overlays (polygons on background image)
from aoi.json for auditability.

This module is intentionally lightweight and depends only on matplotlib + PIL
(via matplotlib imread).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _load_aoi_classes(aoi_json_path: str) -> Dict[str, List[List[Tuple[float, float]]]]:
    with open(aoi_json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    classes = d.get("aoi_classes", {}) if isinstance(d, dict) else {}

    out: Dict[str, List[List[Tuple[float, float]]]] = {}
    for cls, polys in classes.items():
        poly_list = []
        for p in polys:
            pts = p.get("points", p)  # allow legacy list-of-points
            poly_list.append([(float(x), float(y)) for x, y in pts])
        out[str(cls)] = poly_list
    return out


def find_background_for_aoi(aoi_json_path: str) -> Optional[str]:
    """Try to find a background image in the same folder as aoi.json.

    Preference order:
    1) if aoi.json has image.filename and it exists in the same directory
    2) largest image file in the directory
    """
    aoi_p = Path(aoi_json_path)
    scene_dir = aoi_p.parent

    try:
        d = json.load(open(aoi_json_path, "r", encoding="utf-8"))
        img = d.get("image", {}) if isinstance(d, dict) else {}
        fn = img.get("filename") if isinstance(img, dict) else None
        if fn:
            cand = scene_dir / str(fn)
            if cand.exists() and cand.is_file():
                return str(cand)
    except Exception:
        pass

    imgs = [p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not imgs:
        return None
    imgs.sort(key=lambda p: (p.stat().st_size, p.name))
    return str(imgs[-1])


def export_aoi_overlay_png(
    *,
    aoi_json_path: str,
    background_img: Optional[str],
    out_png: str,
    title: Optional[str] = None,
    dpi: int = 200,
) -> str:
    """Export AOI polygons overlayed on a background image.

    Parameters
    - aoi_json_path: path to aoi.json
    - background_img: path to png/jpg; if None, will auto-search same dir
    - out_png: output png path
    """
    if background_img is None:
        background_img = find_background_for_aoi(aoi_json_path)
    if background_img is None:
        raise FileNotFoundError(f"No background image found for aoi_json={aoi_json_path}")

    classes = _load_aoi_classes(aoi_json_path)

    # Simple stable palette (high-contrast). Up to many classes.
    palette = [
        "#00B5FF",  # cyan
        "#FF4D4D",  # red
        "#FFD166",  # yellow
        "#06D6A0",  # green
        "#9B5DE5",  # purple
        "#F15BB5",  # pink
        "#FEE440",  # gold
        "#00F5D4",  # teal
    ]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(plt.imread(background_img))

    for i, (cls, polys) in enumerate(sorted(classes.items(), key=lambda kv: kv[0])):
        color = palette[i % len(palette)]
        for pts in polys:
            if len(pts) < 3:
                continue
            xs = [p[0] for p in pts] + [pts[0][0]]
            ys = [p[1] for p in pts] + [pts[0][1]]
            ax.plot(xs, ys, color=color, linewidth=2)
            ax.fill(xs, ys, color=color, alpha=0.15)

    if title:
        ax.set_title(title)

    ax.axis("off")
    fig.tight_layout()

    out_p = Path(out_png)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_png
