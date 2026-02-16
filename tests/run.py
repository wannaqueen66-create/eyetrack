#!/usr/bin/env python3
import os
import sys
import importlib.util


def _run_file(path: str):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)


def main():
    # ensure repo root on sys.path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)

    files = [
        os.path.join(repo_root, 'tests', 'test_aoi_metrics.py'),
        os.path.join(repo_root, 'tests', 'test_columns.py'),
        os.path.join(repo_root, 'tests', 'test_filters.py'),
    ]

    ok = True
    for fp in files:
        try:
            _run_file(fp)
            print(f'{os.path.basename(fp)}: OK')
        except Exception as e:
            ok = False
            print(f'{os.path.basename(fp)}: FAIL -> {e}')

    if not ok:
        sys.exit(1)
    print('All tests: OK')


if __name__ == '__main__':
    main()
