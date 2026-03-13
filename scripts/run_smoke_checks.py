#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def expect_ok(name: str, cmd: list[str], failures: list[str]) -> None:
    code, out, err = run(cmd)
    if code != 0:
        failures.append(f"{name} failed (exit={code})\nSTDOUT:\n{out}\nSTDERR:\n{err}")


def main() -> int:
    failures: list[str] = []

    expect_ok("run_analysis2_help", [sys.executable, str(REPO / 'scripts' / 'run_analysis2.py'), '--help'], failures)
    expect_ok("run_colab_one_command_help", [sys.executable, str(REPO / 'scripts' / 'run_colab_one_command.py'), '--help'], failures)
    expect_ok("build_main_branch_figure_pack_help", [sys.executable, str(REPO / 'scripts' / 'build_main_branch_figure_pack.py'), '--help'], failures)
    expect_ok("build_main_branch_writing_guide_help", [sys.executable, str(REPO / 'scripts' / 'build_main_branch_writing_guide.py'), '--help'], failures)
    expect_ok("check_doc_consistency", [sys.executable, str(REPO / 'scripts' / 'check_doc_consistency.py')], failures)
    expect_ok("check_main_entrypoints", [sys.executable, str(REPO / 'scripts' / 'check_main_entrypoints.py')], failures)

    with tempfile.TemporaryDirectory(prefix='eyetrack_smoke_') as tmpdir:
        out_dir = Path(tmpdir) / 'docs'
        expect_ok(
            'build_main_branch_writing_guide_generate',
            [sys.executable, str(REPO / 'scripts' / 'build_main_branch_writing_guide.py'), '--out-dir', str(out_dir)],
            failures,
        )
        generated = out_dir / 'MAIN_BRANCH_WRITING_GUIDE.md'
        if not generated.exists():
            failures.append('build_main_branch_writing_guide did not create MAIN_BRANCH_WRITING_GUIDE.md')

    if failures:
        print('SMOKE CHECKS FAILED')
        for item in failures:
            print('\n---\n' + item)
        return 1

    print('SMOKE CHECKS PASSED')
    print('Checked help interfaces, doc consistency, entrypoints, and writing-guide generation.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
