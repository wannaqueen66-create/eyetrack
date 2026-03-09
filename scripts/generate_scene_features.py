#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from PIL import Image

from src.aoi_metrics import load_aoi_json, load_aoi_json_meta, normalize_aoi_class_name
from src.manifest_scene_order import attach_manifest_trial_metadata


IMG_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}


def parse_scene_id_tokens(scene_id: str) -> dict:
    import re

    s = str(scene_id) if scene_id is not None else ""
    rnd = None
    m = re.search(r"(?:组|group)\s*(?P<r>[12])", s, flags=re.IGNORECASE)
    if m:
        rnd = int(m.group("r"))

    wwr = None
    comp = None
    m = re.search(r"WWR(?P<wwr>15|45|75)_C(?P<c>[01])", s, flags=re.IGNORECASE)
    if m:
        wwr = int(m.group("wwr"))
        comp = f"C{m.group('c')}"
    else:
        m = re.search(r"C(?P<c>[01])W(?P<wwr>15|45|75)", s, flags=re.IGNORECASE)
        if m:
            wwr = int(m.group("wwr"))
            comp = f"C{m.group('c')}"

    return {
        "round": rnd,
        "round_label": (f"Round{rnd}" if rnd in (1, 2) else None),
        "WWR": wwr,
        "Complexity": comp,
        "condition_id": (f"WWR{wwr}_{comp}" if wwr is not None and comp is not None else None),
    }


def pick_bg(scene_dir: Path) -> Path | None:
    imgs = [p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not imgs:
        return None
    return max(imgs, key=lambda p: p.stat().st_size)


def find_aoi_json(scene_dir: Path, bg_path: Path | None, mode: str) -> Path | None:
    if mode == 'aoi_json':
        p = scene_dir / 'aoi.json'
        return p if p.exists() else None
    if mode == 'auto':
        js = sorted([p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() == '.json'])
        for p in js:
            if p.name.lower() == 'aoi.json':
                return p
        return js[0] if js else None
    if bg_path is None:
        return None
    p = scene_dir / f'{bg_path.stem}.json'
    return p if p.exists() else None


def poly_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def clamp(v: float | None, lo: float = 0.0, hi: float = 1.0) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return v
    return max(lo, min(hi, float(v)))


def image_size(bg_path: Path | None, aoi_json: Path | None) -> tuple[float | None, float | None]:
    w = h = None
    if aoi_json and aoi_json.exists():
        try:
            meta = load_aoi_json_meta(str(aoi_json))
            img = meta.get('image') or {}
            w = img.get('width')
            h = img.get('height')
        except Exception:
            pass
    if (not w or not h) and bg_path and bg_path.exists():
        try:
            with Image.open(bg_path) as im:
                w, h = im.size
        except Exception:
            pass
    return (float(w) if w else None, float(h) if h else None)


def build_scene_feature_row(scene_id: str, scene_dir: Path, aoi_json_mode: str) -> dict:
    bg = pick_bg(scene_dir)
    aoi_path = find_aoi_json(scene_dir, bg, aoi_json_mode)
    if aoi_path is None:
        raise FileNotFoundError(f"No AOI json found for scene={scene_id} mode={aoi_json_mode}")

    polys = load_aoi_json(str(aoi_path))
    width_px, height_px = image_size(bg, aoi_path)
    image_area_px = (width_px * height_px) if (width_px and height_px) else None

    class_stats: dict[str, dict[str, float]] = {}
    total_area = 0.0
    total_polygons = 0
    for p in polys:
        area = poly_area(p.points)
        total_area += area
        total_polygons += 1
        s = class_stats.setdefault(p.class_name, {"polygon_count": 0, "area_px": 0.0, "cx_num": 0.0, "cy_num": 0.0})
        s["polygon_count"] += 1
        s["area_px"] += area
        if area > 0:
            xs = [x for x, _ in p.points]
            ys = [y for _, y in p.points]
            s["cx_num"] += area * (sum(xs) / len(xs))
            s["cy_num"] += area * (sum(ys) / len(ys))

    class_count = len(class_stats)
    table = class_stats.get(normalize_aoi_class_name('table'), {"polygon_count": 0, "area_px": 0.0, "cx_num": 0.0, "cy_num": 0.0})
    table_area_px = float(table["area_px"])
    table_polygon_count = int(table["polygon_count"])
    non_table_area_px = float(max(total_area - table_area_px, 0.0))
    non_table_class_count = int(max(class_count - (1 if table_polygon_count > 0 else 0), 0))

    table_center_x_px = None
    table_center_y_px = None
    if table_area_px > 0:
        table_center_x_px = table["cx_num"] / table_area_px
        table_center_y_px = table["cy_num"] / table_area_px

    table_center_offset_ratio = None
    if width_px and height_px and table_center_x_px is not None and table_center_y_px is not None:
        dx = table_center_x_px - (width_px / 2.0)
        dy = table_center_y_px - (height_px / 2.0)
        denom = math.hypot(width_px / 2.0, height_px / 2.0)
        if denom > 0:
            table_center_offset_ratio = math.hypot(dx, dy) / denom

    parsed = parse_scene_id_tokens(scene_id)

    row = {
        "scene_id": scene_id,
        "scene_folder": str(scene_dir),
        "background_image": str(bg) if bg else None,
        "aoi_json": str(aoi_path),
        "image_width_px": width_px,
        "image_height_px": height_px,
        "image_area_px": image_area_px,
        "aoi_class_count": class_count,
        "aoi_polygon_count": total_polygons,
        "aoi_total_area_px": total_area,
        "aoi_coverage_ratio": (total_area / image_area_px) if image_area_px else None,
        "table_polygon_count": table_polygon_count,
        "table_area_px": table_area_px,
        "table_density": (table_area_px / image_area_px) if image_area_px else None,
        "table_area_ratio": (table_area_px / image_area_px) if image_area_px else None,
        "table_center_x_px": table_center_x_px,
        "table_center_y_px": table_center_y_px,
        "table_center_offset_ratio": clamp(table_center_offset_ratio),
        "non_table_aoi_area_px": non_table_area_px,
        "non_table_aoi_coverage_ratio": (non_table_area_px / image_area_px) if image_area_px else None,
        "occlusion_ratio": (non_table_area_px / image_area_px) if image_area_px else None,
        "crowding_level": clamp((non_table_class_count / max(class_count, 1)) if class_count else None),
        "non_table_class_count": non_table_class_count,
        "has_table": int(table_polygon_count > 0),
        **parsed,
    }
    return row


def main():
    ap = argparse.ArgumentParser(description='Auto-generate scene_features.csv from batch AOI outputs + scene folders')
    ap.add_argument('--aoi_class_csv', required=True, help='batch_aoi_metrics_by_class.csv (used to enumerate participant_id × scene_id rows)')
    ap.add_argument('--scenes_root', required=True, help='Root directory of scene folders')
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--group_manifest', default=None, help='Optional group_manifest.csv; copied through for helpful columns like SportFreq/Experience')
    ap.add_argument('--aoi_json_mode', default='image_stem', choices=['image_stem', 'aoi_json', 'auto'])
    args = ap.parse_args()

    aoi = pd.read_csv(args.aoi_class_csv, encoding='utf-8-sig')
    req = {'participant_id', 'scene_id'}
    miss = req - set(aoi.columns)
    if miss:
        raise SystemExit(f'aoi_class_csv missing columns: {sorted(miss)}')

    pairs = aoi[['participant_id', 'scene_id']].drop_duplicates().copy()
    pairs['participant_id'] = pairs['participant_id'].astype(str).str.strip()
    pairs['scene_id'] = pairs['scene_id'].astype(str).str.strip()

    scene_rows = []
    for scene_id in sorted(pairs['scene_id'].dropna().unique().tolist()):
        scene_dir = Path(args.scenes_root) / scene_id
        if not scene_dir.exists() or not scene_dir.is_dir():
            raise SystemExit(f'scene folder not found for scene_id={scene_id}: {scene_dir}')
        scene_rows.append(build_scene_feature_row(scene_id, scene_dir, args.aoi_json_mode))

    scene_df = pd.DataFrame(scene_rows)
    out = pairs.merge(scene_df, on='scene_id', how='left')

    if args.group_manifest and os.path.exists(args.group_manifest):
        gm = pd.read_csv(args.group_manifest, encoding='utf-8-sig')
        id_col = None
        for c in ['participant_id', 'name', 'id']:
            if c in gm.columns:
                id_col = c
                break
        if id_col:
            gm = gm.copy()
            gm['participant_id'] = gm[id_col].astype(str).str.strip()
            out = attach_manifest_trial_metadata(out, gm, id_col=id_col, scene_col='scene_id')
            keep = ['participant_id'] + [c for c in ['SportFreq', 'Experience', 'condition', 'Order'] if c in gm.columns]
            for c in gm.columns:
                if c.startswith('trial') and (c.endswith('_scene') or c.endswith('_Round') or c.endswith('_RoundLabel') or c.endswith('_Pos') or c.endswith('_key') or c.endswith('_label') or c.endswith('_Cond') or c.endswith('_Complexity') or c.endswith('_WWR')):
                    keep.append(c)
            keep = list(dict.fromkeys(keep))
            out = out.merge(gm[keep], on='participant_id', how='left')

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, encoding='utf-8-sig')
    print('Saved:', args.out_csv)
    print('Columns:', ', '.join(out.columns))


if __name__ == '__main__':
    main()
