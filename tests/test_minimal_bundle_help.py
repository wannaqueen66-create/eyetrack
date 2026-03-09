import os
import subprocess
import sys


def test_run_minimal_aoi_bundle_help():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, os.path.join(repo_root, 'scripts', 'run_minimal_aoi_bundle.py'), '--help']
    out = subprocess.check_output(cmd, text=True)
    assert '--csv_dir' in out
    assert '--group_manifest' in out
    assert '--scene_image' in out
    assert '--aoi_json' in out
    assert '--scene_id' in out
