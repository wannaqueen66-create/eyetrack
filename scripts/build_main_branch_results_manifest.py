#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

PRIMARY_OUTCOMES = [
    'share_pct',
    'share_logit',
    'FC_share',
    'fc_share_logit',
    'FC_rate',
    'tfd_y',
    'ttff_y',
    'fc_y',
]

EXPLORATORY_OUTCOMES = ['ffd_y', 'mfd_y', 'rff_y', 'MPD']

TRACKS = [
    ('all_sample', '00_全样本_AllSample'),
    ('after_qc', '01_QC后_AfterQC'),
]


def collect_track(track_root: Path) -> dict:
    return {
        'exists': track_root.exists(),
        'descriptive_dir': str(track_root / '01_描述性分析_Descriptive'),
        'significance_dir': str(track_root / '02_显著性分析_Significance'),
        'experience_index': str(track_root / '02_显著性分析_Significance' / 'allocation_lmm' / 'groupvar_Experience' / 'tables' / 'model_family_index.csv'),
        'experience_packet_summary': str(track_root / '02_显著性分析_Significance' / 'allocation_lmm' / 'groupvar_Experience' / 'tables' / 'three_model_packet_summary.csv'),
        'grouped_experience_png': str(track_root / '01_描述性分析_Descriptive' / 'grouped_experience' / 'png'),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='Build a lightweight manifest for a generated eyetrack main result bundle')
    ap.add_argument('--results-root', type=Path, required=True, help='Path to a generated 研究输出_YYYYMMDD_HHMMSS folder')
    args = ap.parse_args()

    manifest = {
        'results_root': str(args.results_root),
        'primary_outcomes': PRIMARY_OUTCOMES,
        'exploratory_outcomes': EXPLORATORY_OUTCOMES,
        'tracks': {},
    }
    for slug, dirname in TRACKS:
        manifest['tracks'][slug] = collect_track(args.results_root / dirname)
        manifest['tracks'][slug]['track_dirname'] = dirname

    out = args.results_root / 'main_branch_results_manifest.json'
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'generated': str(out)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
