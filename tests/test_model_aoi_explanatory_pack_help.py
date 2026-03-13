import os
import subprocess
import sys


def test_model_aoi_explanatory_pack_help():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, os.path.join(repo_root, 'scripts', 'model_aoi_explanatory_pack.py'), '--help']
    out = subprocess.check_output(cmd, text=True)
    assert '--analysis_csv' in out
    assert '--outdir' in out
    assert '--predictors' in out
    assert '--interactions' in out
