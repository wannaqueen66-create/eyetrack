#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]

TARGETS = [
    REPO / 'README.md',
    REPO / 'README_zh.md',
    REPO / 'RESULTS_MAP.md',
    REPO / 'docs' / 'PROJECT_OVERVIEW.md',
    REPO / 'docs' / 'RESULTS_READING_GUIDE.md',
    REPO / 'docs' / 'RESULTS_READING_GUIDE.zh.md',
]

REQUIRED_TOKENS = [
    '00_全样本_AllSample',
    '01_QC后_AfterQC',
    '01_描述性分析_Descriptive',
    '02_显著性分析_Significance',
]


def main() -> int:
    failures: list[str] = []
    for path in TARGETS:
        if not path.exists():
            failures.append(f'missing doc: {path.relative_to(REPO)}')
            continue
        text = path.read_text(encoding='utf-8')
        for token in REQUIRED_TOKENS:
            if token not in text:
                failures.append(f'{path.relative_to(REPO)} missing token: {token}')

    if failures:
        print('DOC CONSISTENCY CHECK FAILED')
        for item in failures:
            print(item)
        return 1

    print('DOC CONSISTENCY CHECK PASSED')
    for path in TARGETS:
        print(path.relative_to(REPO))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
