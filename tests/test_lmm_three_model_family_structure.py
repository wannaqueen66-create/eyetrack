import importlib.util
import pathlib


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / 'model_aoi_lmm_allocation.py'
spec = importlib.util.spec_from_file_location('model_aoi_lmm_allocation', SCRIPT_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_model_family_specs_experience():
    specs = mod._model_family_specs('Experience')
    assert [s['family_key'] for s in specs] == [
        '01_main_effects',
        '02_two_way_interactions',
        '03_three_way_interaction',
    ]
    assert specs[0]['family_title'] == 'Main effects'
    assert specs[1]['export_contrasts'] is True
    assert specs[2]['formula_rhs'] == 'C(class_name) * WWR_z * Complexity_z * C(Experience)'


def test_write_model_family_index_and_packet_summary(tmp_path):
    gdir = tmp_path / 'groupvar_Experience'
    gdir.mkdir(parents=True)
    mod._write_model_family_index(gdir, 'Experience')
    assert (gdir / 'model_family_index.csv').exists()

    for family_dir in ['01_main_effects', '02_two_way_interactions', '03_three_way_interaction']:
        fdir = gdir / family_dir
        fdir.mkdir(parents=True, exist_ok=True)
        (fdir / 'model_stability_summary.csv').write_text(
            'group_var,outcome,outcome_label,subset,n,formula,stability_grade,stability_grade_rank,stability_reasons,warning_count,aic,bic,logLik\n'
            f'Experience,share_pct,share_pct,All rows,120,y~x,stable,1,,0,10,12,-3\n',
            encoding='utf-8',
        )

    mod._write_three_model_packet_summary(gdir, 'Experience')
    out = gdir / 'three_model_packet_summary.csv'
    assert out.exists()
    txt = out.read_text(encoding='utf-8-sig')
    assert 'main_effects' in txt
    assert 'two_way_interactions' in txt
    assert 'three_way_interaction' in txt
