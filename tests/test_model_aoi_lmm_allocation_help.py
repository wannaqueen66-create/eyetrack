import os
import subprocess
import sys


def test_model_aoi_lmm_allocation_help():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, os.path.join(repo_root, 'scripts', 'model_aoi_lmm_allocation.py'), '--help']
    out = subprocess.check_output(cmd, text=True)
    assert '--aoi_class_csv' in out
    assert '--group_manifest' in out
    assert '--group_id_col' in out
    assert '--outdir' in out
    assert '--min_rows' in out
