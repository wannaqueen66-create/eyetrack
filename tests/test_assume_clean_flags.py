# This is a lightweight "smoke" test to ensure CLI flags parse.
# It does not execute the full pipeline (no real CSV files in repo).

import subprocess
import sys
import os


def test_run_aoi_metrics_help_includes_assume_clean():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, os.path.join(repo_root, 'scripts', 'run_aoi_metrics.py'), '--help']
    out = subprocess.check_output(cmd, text=True)
    assert '--assume_clean' in out


def test_batch_aoi_metrics_help_includes_assume_clean():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, os.path.join(repo_root, 'scripts', 'batch_aoi_metrics.py'), '--help']
    out = subprocess.check_output(cmd, text=True)
    assert '--assume_clean' in out
