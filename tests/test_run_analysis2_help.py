import os
import subprocess
import sys


def test_run_analysis2_help():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, os.path.join(repo_root, 'scripts', 'run_analysis2.py'), '--help']
    out = subprocess.check_output(cmd, text=True)
    assert '--group_manifest' in out
    assert '--scenes_root' in out
    assert '--batch_class_csv' in out
    assert '--scene_features_csv' in out
    assert '--qc_exclusion_csv' in out
    assert '--out_root' in out
