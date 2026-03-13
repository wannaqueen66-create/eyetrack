#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    REPO / 'README.md',
    REPO / 'README_zh.md',
    REPO / 'RESULTS_MAP.md',
    REPO / 'docs' / 'PROJECT_OVERVIEW.md',
    REPO / 'docs' / 'RESULTS_READING_GUIDE.md',
    REPO / 'docs' / 'RESULTS_READING_GUIDE.zh.md',
    REPO / 'docs' / 'COLAB_ZH.md',
    REPO / 'docs' / 'SIGNIFICANCE_MAINLINE.md',
    REPO / 'scripts' / 'run_mainline_bundle.py',
    REPO / 'scripts' / 'run_analysis2.py',
    REPO / 'scripts' / 'run_colab_mainline_bundle.py',
    REPO / 'scripts' / 'run_colab_one_command.py',
    REPO / 'scripts' / 'run_one_scene_bundle.py',
    REPO / 'scripts' / 'model_scene_feature_mixed.py',
    REPO / 'scripts' / 'build_manuscript_figures.py',
    REPO / 'scripts' / 'optimize_merged_batch_outputs.py',
    REPO / 'scripts' / 'model_aoi_explanatory_pack.py',
    REPO / 'scripts' / 'build_main_branch_figure_pack.py',
    REPO / 'scripts' / 'build_main_branch_writing_guide.py',
    REPO / 'scripts' / 'build_main_branch_captions.py',
    REPO / 'scripts' / 'build_main_branch_packet_summary.py',
    REPO / 'scripts' / 'build_main_branch_results_manifest.py',
    REPO / 'scripts' / 'build_main_support_docs_readme.py',
    REPO / 'scripts' / 'check_doc_consistency.py',
    REPO / 'scripts' / 'run_smoke_checks.py',
]

README_LINK_SOURCES = [REPO / 'README.md', REPO / 'README_zh.md']
LINK_RE = re.compile(r'\((\./[^)]+)\)')


def check_required_files() -> list[str]:
    failures: list[str] = []
    for path in REQUIRED_FILES:
        if not path.exists():
            failures.append(f'missing required file: {path.relative_to(REPO)}')
    return failures


def check_readme_links() -> list[str]:
    failures: list[str] = []
    for src in README_LINK_SOURCES:
        if not src.exists():
            continue
        text = src.read_text(encoding='utf-8')
        for rel in LINK_RE.findall(text):
            target = (src.parent / rel).resolve()
            if not target.exists():
                failures.append(f'broken README link: {src.relative_to(REPO)} -> {rel}')
    return failures


def run_doc_consistency() -> list[str]:
    cmd = [sys.executable, str(REPO / 'scripts' / 'check_doc_consistency.py')]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = proc.stdout.strip() or proc.stderr.strip() or 'unknown failure'
        return [f'doc consistency check failed: {msg}']
    return []


def main() -> int:
    failures: list[str] = []
    failures.extend(check_required_files())
    failures.extend(check_readme_links())
    failures.extend(run_doc_consistency())
    if failures:
        print('MAIN ENTRYPOINT CHECK FAILED')
        for item in failures:
            print(item)
        return 1
    print('MAIN ENTRYPOINT CHECK PASSED')
    print('Required files present:', len(REQUIRED_FILES))
    print('README link sources checked:', len(README_LINK_SOURCES))
    print('Doc consistency script: passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
