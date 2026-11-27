import copy
import hashlib
import json

import pytest

from vemgsbm.mri_v2_compare import comparison_configs,verify_identity


def checkpoint():
    return dict(format_version=6,config=dict(online_atlas=False,frequency_split=True,
        atlas_weight=.05,data={'train':{},'val':{}},sampling={'steps':8,'solver':'heun'}),
        jetatlas={'train_bank':True},jetatlas_state={'trained_router':True})


def test_paired_configs_change_only_online_switch_and_preserve_source():
    ck=checkpoint();original=copy.deepcopy(ck)
    cfg=comparison_configs(ck)
    assert cfg['offline']==ck['config']
    assert cfg['online']=={**cfg['offline'],'online_atlas':True}
    cfg['online']['sampling']['steps']=16
    assert ck==original and cfg['offline']['sampling']['steps']==8


@pytest.mark.parametrize('field,value', [
    ('format_version',5),('next_batch',2),('jetatlas',None),('jetatlas_state',None)])
def test_untrained_or_legacy_geometry_is_rejected(field,value):
    ck=checkpoint();ck[field]=value
    with pytest.raises(ValueError,match='exp3 checkpoint'):comparison_configs(ck)


def test_online_selected_checkpoint_and_test_contract_are_rejected():
    ck=checkpoint();ck['config']['online_atlas']=True
    with pytest.raises(ValueError):comparison_configs(ck)
    ck=checkpoint();ck['config']['data']['test']={}
    with pytest.raises(ValueError,match='TRAIN/VAL'):comparison_configs(ck)


def test_inventory_identity_must_be_exact_and_exclude_test():
    rows=[dict(split='train',id='a',fingerprint='x'),dict(split='val',id='b',fingerprint='y')]
    ck=dict(data_identity=rows,manifest_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest())
    verify_identity(ck,rows)
    with pytest.raises(ValueError,match='differs'):verify_identity(ck,[*rows[:-1],{**rows[-1],'fingerprint':'other'}])
    with pytest.raises(ValueError,match='TEST'):verify_identity(ck,[*rows,dict(split='test',id='c')])


def test_shared_metadata_preparation_preserves_exp1_and_training_budget(tmp_path):
    import gzip
    import importlib.util
    from pathlib import Path
    import yaml
    path=Path(__file__).resolve().parents[1]/'tools'/'mri_v2_prepare_shared.py'
    spec=importlib.util.spec_from_file_location('prepare_shared_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    root=Path(__file__).resolve().parents[1]
    source=tmp_path/'old';source.mkdir()
    cfg=yaml.safe_load((root/'configs/mri_v2_exp3_freq_gsbm_jetatlas_train.yaml').read_text())
    cfg['run_dir']=str(source/'exp3')
    protected={**cfg,'run_dir':str(source/'exp1')}
    (source/'exp1.yaml').write_text(yaml.safe_dump(protected))
    (source/'exp3.yaml').write_text(yaml.safe_dump(cfg))
    (source/'gate_passed').touch()
    with gzip.open(source/'trainval_inventory.json.gz','wt') as stream:
        json.dump({'data':cfg['data'],'splits':{'train':[{'id':'train'}],'val':[{'id':'val'}]}},stream)
    before={p.name:p.read_bytes() for p in source.iterdir()}
    with pytest.raises(ValueError,match='separate'):module.prepare(source,source/'exp1'/'nested')
    result=module.prepare(source,tmp_path/'new')
    actual=yaml.safe_load((tmp_path/'new'/'exp3.yaml').read_text())
    assert actual['train']==cfg['train'] and actual['online_atlas'] is False
    assert result['config_only_changed_run_dir']
    assert all((source/name).read_bytes()==value for name,value in before.items())
    assert (source/'trainval_inventory.json.gz').read_bytes()==(tmp_path/'new'/'trainval_inventory.json.gz').read_bytes()
