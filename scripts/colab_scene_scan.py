#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}
NON_SCENE_PATTERNS = [
    r'^AOI输出(?:_|$)',
    r'^research_bundle(?:_|$)',
    r'^输出结果(?:_|$)',
    r'^results?(?:_|$)',
    r'^colab(?:_|$)',
    r'^__MACOSX$',
]


@dataclass
class SceneEntry:
    name: str
    path: Path
    bg_count: int
    csv_count: int
    json_count: int
    formal: bool
    skip_reason: str | None = None


def count_scene_assets(scene_dir: Path) -> tuple[int, int, int]:
    bg_count = 0
    csv_count = 0
    json_count = 0
    for child in scene_dir.iterdir():
        if not child.is_file():
            continue
        suffix = child.suffix.lower()
        if suffix in IMG_EXTS:
            bg_count += 1
        elif suffix == '.csv':
            csv_count += 1
        elif suffix == '.json':
            json_count += 1
    return bg_count, csv_count, json_count


def classify_non_scene_dir(name: str) -> str | None:
    for pattern in NON_SCENE_PATTERNS:
        if re.search(pattern, name, flags=re.IGNORECASE):
            return '明显是结果/打包目录，按 non-scene 跳过'
    return None


def scan_scene_root(root: str | os.PathLike[str]) -> dict:
    root_path = Path(root)
    entries: list[SceneEntry] = []
    for child in sorted(root_path.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        non_scene_reason = classify_non_scene_dir(child.name)
        bg_count, csv_count, json_count = count_scene_assets(child)
        formal = (bg_count > 0) and (csv_count > 0) and (json_count > 0) and (non_scene_reason is None)
        skip_reason = None
        if non_scene_reason is not None:
            skip_reason = non_scene_reason
        elif bg_count == 0:
            skip_reason = '缺少背景图'
        elif json_count == 0:
            skip_reason = '缺少 AOI JSON'
        elif csv_count == 0:
            skip_reason = '缺少被试 CSV'
        entries.append(SceneEntry(
            name=child.name,
            path=child,
            bg_count=bg_count,
            csv_count=csv_count,
            json_count=json_count,
            formal=formal,
            skip_reason=skip_reason,
        ))

    valid_entries = [e for e in entries if e.formal]
    skipped_entries = [e for e in entries if not e.formal]
    return {
        'root': str(root_path),
        'candidate_count': len(entries),
        'valid_count': len(valid_entries),
        'valid_entries': valid_entries,
        'skipped_entries': skipped_entries,
    }


def print_scan_report(root: str | os.PathLike[str], group_manifest: str | os.PathLike[str] | None = None):
    scan = scan_scene_root(root)
    print('=== 场景扫描预检 / Scene precheck ===')
    print('SCENES_ROOT:', scan['root'])
    if group_manifest is not None:
        manifest_path = Path(group_manifest)
        exists = manifest_path.exists()
        print(f"group_manifest: {'FOUND' if exists else 'MISSING'} -> {manifest_path}")
    print('候选目录数量 / Candidate directories:', scan['candidate_count'])
    print('识别出的有效场景数 / Valid scenes:', scan['valid_count'])

    if scan['valid_entries']:
        print('有效场景检查 / Valid scene checks:')
        for entry in scan['valid_entries']:
            print(
                f"  - {entry.name}: 背景图 {entry.bg_count} | CSV {entry.csv_count} | JSON {entry.json_count}"
            )
    else:
        print('有效场景检查 / Valid scene checks: 0')

    if scan['skipped_entries']:
        print('[WARN] 跳过的目录汇总 / Skipped directories summary:')
        for entry in scan['skipped_entries']:
            print(
                f"  - {entry.name}: {entry.skip_reason} "
                f"(背景图 {entry.bg_count} | CSV {entry.csv_count} | JSON {entry.json_count})"
            )
    print('=== 预检结束 / Precheck done ===')
    return scan
