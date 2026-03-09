import importlib.util
import pathlib

import pandas as pd


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / 'model_aoi_lmm_allocation.py'
spec = importlib.util.spec_from_file_location('model_aoi_lmm_allocation', SCRIPT_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_evidence_plot_smoke(tmp_path):
    stab = pd.DataFrame([
        {
            'group_var': 'Experience',
            'outcome': 'share_pct',
            'outcome_label': 'attention allocation share percentage (TFD-based)',
            'stability_grade': 'stable',
            'stability_grade_rank': 1,
            'warning_count': 0,
            'stability_reasons': '',
            'stability_notes': 'ok',
            'aic': 10.0,
            'bic': 12.0,
            'logLik': -3.0,
        },
        {
            'group_var': 'Experience',
            'outcome': 'ttff_y',
            'outcome_label': 'log1p(TTFF), visited==1',
            'stability_grade': 'caution',
            'stability_grade_rank': 2,
            'warning_count': 1,
            'stability_reasons': 'boundary',
            'stability_notes': 'warn',
            'aic': 15.0,
            'bic': 17.0,
            'logLik': -5.0,
        },
    ])
    fit = pd.DataFrame([
        {
            'group_var': 'Experience',
            'outcome': 'share_pct',
            'formula': 'y ~ x',
            'subset': 'All rows',
            'nobs': 100,
            'n_participants': 10,
            'n_scenes': 6,
            'n_aoi_classes': 3,
            'aic': 10.0,
            'bic': 12.0,
            'logLik': -3.0,
            'r2_marginal': 0.2,
            'r2_conditional': 0.45,
            'var_fixed': 0.2,
            'var_random_total': 0.15,
            'var_residual': 0.65,
            'converged': True,
            'stability_grade': 'stable',
            'stability_reasons': '',
        },
        {
            'group_var': 'Experience',
            'outcome': 'ttff_y',
            'formula': 'y ~ x',
            'subset': 'visited==1',
            'nobs': 80,
            'n_participants': 10,
            'n_scenes': 6,
            'n_aoi_classes': 3,
            'aic': 15.0,
            'bic': 17.0,
            'logLik': -5.0,
            'r2_marginal': 0.12,
            'r2_conditional': 0.32,
            'var_fixed': 0.12,
            'var_random_total': 0.20,
            'var_residual': 0.68,
            'converged': True,
            'stability_grade': 'caution',
            'stability_reasons': 'boundary',
        },
    ])
    contrasts = pd.DataFrame([
        {
            'contrast_family': 'group_simple_effect',
            'outcome': 'share_pct',
            'group_var': 'Experience',
            'aoi': 'table',
            'Complexity': 'C0',
            'WWR': 15,
            'contrast_label': 'High - Low | table, WWR15, C0',
            'estimate': 0.3,
            'se': 0.1,
            'test_stat': 3.0,
            'p': 0.01,
            'ci_low': 0.1,
            'ci_high': 0.5,
        },
        {
            'contrast_family': 'wwr_simple_effect',
            'outcome': 'share_pct',
            'group_var': 'Experience',
            'aoi': 'window',
            'Complexity': 'C1',
            'WWR': None,
            'group_level': 'High',
            'contrast_label': 'WWR75 - WWR15 | window, C1, High',
            'estimate': -0.2,
            'se': 0.12,
            'test_stat': -1.8,
            'p': 0.08,
            'ci_low': -0.44,
            'ci_high': 0.04,
        },
    ])
    fixef = pd.DataFrame([
        {'term': 'C(class_name)[T.window]', 'coef': 0.25, 'se': 0.1, 'test_stat': 2.5, 'p': 0.01, 'ci_low': 0.05, 'ci_high': 0.45, 'abs_test': 2.5},
        {'term': 'WWR_z', 'coef': -0.15, 'se': 0.08, 'test_stat': -1.9, 'p': 0.06, 'ci_low': -0.31, 'ci_high': 0.01, 'abs_test': 1.9},
        {'term': 'C(class_name)[T.window]:WWR_z:C(Experience)[T.High]', 'coef': 0.4, 'se': 0.15, 'test_stat': 2.67, 'p': 0.008, 'ci_low': 0.11, 'ci_high': 0.69, 'abs_test': 2.67},
    ])

    stab['model_family'] = 'two_way_interactions'
    fit['model_family'] = 'two_way_interactions'
    contrasts['model_family'] = 'two_way_interactions'
    fixef['model_family'] = 'two_way_interactions'

    png_dir = tmp_path / 'png'
    data_dir = tmp_path / 'data'
    mod._plot_stability_overview(stab, png_dir / 'evidence_stability_overview_Experience.png', 'Experience')
    mod._plot_model_fit_overview(fit, png_dir / 'evidence_model_fit_overview_Experience.png', 'Experience')
    mod._plot_contrasts_overview(contrasts, png_dir / 'evidence_contrasts_share_pct.png', 'Experience', 'share_pct')
    mod._plot_fixef_terms_overview(fixef, png_dir / 'evidence_fixef_key_terms_share_pct.png', 'Experience', 'share_pct')

    assert (png_dir / 'evidence_stability_overview_Experience.png').exists()
    assert (data_dir / 'evidence_stability_overview_Experience_data.csv').exists()
    assert (png_dir / 'evidence_model_fit_overview_Experience.png').exists()
    assert (data_dir / 'evidence_model_fit_overview_Experience_data.csv').exists()
    assert (png_dir / 'evidence_contrasts_share_pct.png').exists()
    assert (data_dir / 'evidence_contrasts_share_pct_data.csv').exists()
    assert (png_dir / 'evidence_fixef_key_terms_share_pct.png').exists()
    assert (data_dir / 'evidence_fixef_key_terms_share_pct_data.csv').exists()
