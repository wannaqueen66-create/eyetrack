import subprocess
import sys
import os


def test_run_all_help():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, os.path.join(repo_root, 'scripts', 'run_all.py'), '--help']
    out = subprocess.check_output(cmd, text=True)
    assert '--input_csv' in out
    assert '--aoi_json' in out
    assert '--workdir' in out
