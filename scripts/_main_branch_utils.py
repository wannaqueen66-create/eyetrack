from __future__ import annotations

import json
from pathlib import Path

TRACKS = [
    ('all_sample', '00_全样本_AllSample'),
    ('after_qc', '01_QC后_AfterQC'),
]

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


def build_results_manifest(results_root: Path) -> dict:
    manifest = {
        'results_root': str(results_root),
        'primary_outcomes': PRIMARY_OUTCOMES,
        'exploratory_outcomes': EXPLORATORY_OUTCOMES,
        'tracks': {},
    }
    for slug, dirname in TRACKS:
        track_root = results_root / dirname
        manifest['tracks'][slug] = {
            'track_dirname': dirname,
            'exists': track_root.exists(),
            'descriptive_dir': str(track_root / '01_描述性分析_Descriptive'),
            'significance_dir': str(track_root / '02_显著性分析_Significance'),
            'experience_index': str(track_root / '02_显著性分析_Significance' / 'allocation_lmm' / 'groupvar_Experience' / 'tables' / 'model_family_index.csv'),
            'experience_packet_summary': str(track_root / '02_显著性分析_Significance' / 'allocation_lmm' / 'groupvar_Experience' / 'tables' / 'three_model_packet_summary.csv'),
            'grouped_experience_png': str(track_root / '01_描述性分析_Descriptive' / 'grouped_experience' / 'png'),
        }
    return manifest


def write_results_manifest(results_root: Path) -> Path:
    out = results_root / 'main_branch_results_manifest.json'
    out.write_text(json.dumps(build_results_manifest(results_root), ensure_ascii=False, indent=2), encoding='utf-8')
    return out
