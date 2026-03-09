import importlib.util
import pathlib
import warnings

import numpy as np
import pandas as pd


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / 'model_aoi_lmm_allocation.py'
spec = importlib.util.spec_from_file_location('model_aoi_lmm_allocation', SCRIPT_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class DummyRes:
    converged = True
    method = 'lbfgs'
    scale = 1.0


def _warn(msg: str):
    with warnings.catch_warnings(record=True) as caught:
        warnings.warn(msg, RuntimeWarning)
    return caught[0]


def test_collect_stability_signals_stable():
    fixef = pd.DataFrame({'se': [0.1], 'ci_low': [0.0], 'ci_high': [0.2]})
    ranef = pd.DataFrame({'component': ['participant_id_random_intercept', 'residual'], 'variance': [0.2, 1.0]})
    out = mod._collect_stability_signals(DummyRes(), fixef, ranef, [])
    assert out['stability_grade'] == 'stable'
    assert out['se_or_ci_abnormal'] is False


def test_collect_stability_signals_caution_for_boundary_warning():
    fixef = pd.DataFrame({'se': [0.1], 'ci_low': [0.0], 'ci_high': [0.2]})
    ranef = pd.DataFrame({'component': ['participant_id_random_intercept', 'residual'], 'variance': [1e-10, 1.0]})
    out = mod._collect_stability_signals(DummyRes(), fixef, ranef, [_warn('The MLE may be on the boundary of the parameter space.')])
    assert out['stability_grade'] == 'caution'
    assert 'boundary' in out['stability_reasons'] or 'near_zero' in out['stability_reasons']


def test_collect_stability_signals_unstable_for_bad_ci_or_hessian():
    fixef = pd.DataFrame({'se': [np.nan], 'ci_low': [1.0], 'ci_high': [0.0]})
    ranef = pd.DataFrame({'component': ['participant_id_random_intercept', 'residual'], 'variance': [0.2, 1.0]})
    out = mod._collect_stability_signals(DummyRes(), fixef, ranef, [_warn('Hessian matrix at the estimated parameter values is not positive definite.')])
    assert out['stability_grade'] == 'unstable'
    assert out['hessian_non_pd'] is True
    assert out['se_or_ci_abnormal'] is True
