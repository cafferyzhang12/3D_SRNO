"""Integration contracts for the four strictly TRAIN/VAL-only upgrade arms."""
import copy
from pathlib import Path

import pytest
import torch

from vemgsbm.mri_v2 import V2Objective, make_model, soft_pareto
from vemgsbm.runtime import read_config, build_models


@pytest.fixture(autouse=True)
def small_threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config(i=1):
    root=Path(__file__).resolve().parents[1]
    cfg=read_config(next((root/'configs').glob(f'mri_v2_exp{i}_*.yaml')))
    cfg['model'].update(widths=[4,8,12],blocks=[1,1,1],heads=1,refine_width=4,window_size=4)
    cfg['use_band_whitening']=False
    return cfg


def test_four_configs_share_epoch_budget_and_keep_clean_controls():
    cfgs=[config(i) for i in range(1,5)]
    for key in ('epochs', 'samples_per_epoch', 'learning_rate'):
        assert all(c['train'][key]==cfgs[0]['train'][key] for c in cfgs)
    assert all('test' not in c['data'] for c in cfgs)
    assert [c['atlas_weight'] for c in cfgs]==[0,0,.05,.05]
    assert [c['online_atlas'] for c in cfgs]==[False,False,False,True]
    for cfg in cfgs[::2]:
        assert cfg['sigma_bridge']==cfg['train']['noise_weight']==cfg['train']['rollout_weight']==0
        assert cfg['train']['batch_size']==48 and cfg['train']['workers']==6
        assert cfg['sampling']['sampler_type']=='ode' and cfg['sampling']['steps']==1
        assert cfg['model']['gradient_checkpointing']=='refinement_only'
    assert cfgs[2]['jetatlas']['lr_dim']==9


@pytest.mark.parametrize('i',[1,2])
def test_objective_real_backward_and_checkpoint_roundtrip(i):
    cfg=config(i);m=make_model(cfg)
    lr=torch.rand(2,1,3,8,8)
    target=(m.prepare_base(lr)+.01*torch.randn(2,1,3,32,32)).clamp(0,1)
    terms=V2Objective(m,cfg)(lr,target,['a','b'])
    assert all(torch.isfinite(v).all() for v in terms.values())
    assert 'start_ssim' in terms and 'start_local_std' in terms
    terms['loss'].backward()
    assert m.refinement.output.weight.grad is not None
    payload=dict(format_version=6,config=cfg,band_statistics={'mean':torch.zeros(16),'std':torch.ones(16)},model=m.state_dict())
    loaded,geometry=build_models(payload,'cpu')
    assert geometry is None
    assert torch.isfinite(loaded.restore(lr,m.prepare_base(lr),steps=8)).all()


def test_soft_floor_retains_higher_ssim_with_small_psnr_drop():
    eligible,score=soft_pareto(dict(psnr=32.3,ssim=.95,lpips=.1,fid=10.),32.552259)
    assert eligible and score[0]==.95
    assert not soft_pareto(dict(psnr=31.,ssim=.99,lpips=.1,fid=10.),32.552259)[0]


def test_v2_low_temperature_sampling_does_not_inherit_legacy_sigma_constraint():
    from vemgsbm.runtime import sampling_noise
    cfg=config(2);cfg['sampling']['sampler_type']='sde';cfg['sampling']['temperature']=.2
    assert sampling_noise(cfg)==pytest.approx(.016)
    cfg['sampling']['sampler_type']='ode'
    assert sampling_noise(cfg)==0.


@pytest.mark.parametrize('metric', ['psnr','ssim','lpips','fid'])
def test_four_metric_selection_rejects_nonfinite_values(metric):
    values=dict(psnr=32.4,ssim=.95,lpips=.1,fid=10.)
    values[metric]=float('nan')
    with pytest.raises(ValueError,match='Finite four-metric'):
        soft_pareto(values,32.552259)


def test_development_config_rejects_test_roots_and_untrained_online_atlas():
    from vemgsbm.mri_v2 import validate_config
    cfg=config()
    cfg['data']['test']={'root_hq':'forbidden-test-hq','root_lq':'forbidden-test-lq'}
    with pytest.raises(ValueError,match='TRAIN/VAL only'):
        validate_config(cfg)
    cfg=config();cfg['online_atlas']=True
    with pytest.raises(ValueError,match='trained atlas'):
        validate_config(cfg)


@pytest.mark.parametrize('split',['val','test'])
def test_geometry_fitting_rejects_nontrain_before_accessing_any_image(split):
    from vemgsbm.mri_v2_train import prepare_train_geometry
    class ForbiddenDataset:
        def __init__(self):self.split=split
        def __getitem__(self,index):
            raise AssertionError('Non-TRAIN images must never be accessed for fitting')
    with pytest.raises(ValueError,match='Only TRAIN'):
        prepare_train_geometry(None,ForbiddenDataset(),config(),'cpu','identity')


def test_hybrid_checkpoint_requires_new_format_before_loading_statistics():
    with pytest.raises(ValueError,match='format 6'):
        build_models({'format_version':5,'config':config()},'cpu')


def _prepare_tool():
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'tools'/'mri_v2_prepare.py'
    spec=importlib.util.spec_from_file_location('v2_prepare_test',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_preparation_requires_allocation_before_filesystem_access(monkeypatch,tmp_path):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError,match='Slurm allocation'):
        _prepare_tool().prepare('missing-inventory.json.gz','audit')
    assert not (tmp_path/'runs').exists()


def test_runtime_preparation_strips_test_metadata_without_opening_images(monkeypatch,tmp_path):
    import gzip
    import json
    import shutil
    import yaml
    root=Path(__file__).resolve().parents[1]
    (tmp_path/'configs').mkdir()
    for source in (root/'configs').glob('mri_v2_exp*.yaml'):
        shutil.copyfile(source,tmp_path/'configs'/source.name)
    data=copy.deepcopy(config()['data'])
    data['test']={'root_hq':'NEVER_OPEN_TEST_HQ','root_lq':'NEVER_OPEN_TEST_LQ'}
    data.setdefault('scan_cache',{})['test']='NEVER_OPEN_TEST_CACHE'
    train_row={'id':'train-case','lq':['slice.png'],'hq':['slice.png']}
    val_row={'id':'val-case','lq':['slice.png'],'hq':['slice.png']}
    inventory=tmp_path/'old.json.gz'
    with gzip.open(inventory,'wt',encoding='utf-8') as stream:
        json.dump({'version':1,'data':data,'splits':{'train':[train_row],'val':[val_row],
            'test':[{'id':'excluded-test','hq':['NEVER_OPEN_TEST_IMAGE']}]}},stream)
    monkeypatch.chdir(tmp_path);monkeypatch.setenv('SLURM_JOB_ID','unit-test-allocation')
    _prepare_tool().prepare(str(inventory),'metadata-probe')
    folder=tmp_path/'runs'/'mri'/'v2'/'metadata-probe'
    with gzip.open(folder/'trainval_inventory.json.gz','rt',encoding='utf-8') as stream:
        prepared=json.load(stream)
    assert set(prepared['splits'])=={'train','val'}
    assert prepared['splits']['train']==[train_row]
    assert prepared['splits']['val']==[val_row]
    assert 'test' not in prepared['data']
    assert 'test' not in prepared['data']['scan_cache']
    for path in folder.glob('exp*.yaml'):
        cfg=yaml.safe_load(path.read_text(encoding='utf-8'))
        assert cfg['data']==prepared['data']
        assert cfg['data']['patch_size']==[7,256,256]


@pytest.mark.parametrize('change',['base','whitening','patch','atlas','seed','manifest'])
def test_geometry_contract_invalidates_stale_train_statistics(change):
    from vemgsbm.mri_v2_train import geometry_contract
    cfg=config(3)
    before=copy.deepcopy(geometry_contract(cfg,'TRAIN-fingerprint'))
    digest='TRAIN-fingerprint'
    if change=='base':cfg['base_interpolation']='lanczos'
    elif change=='whitening':cfg['use_band_whitening']=not cfg['use_band_whitening']
    elif change=='patch':cfg['data']['patch_size']=[5,128,128]
    elif change=='atlas':cfg['jetatlas']['centers']=1024
    elif change=='seed':cfg['train']['seed']+=1
    elif change=='manifest':digest='changed-TRAIN-cohort'
    assert geometry_contract(cfg,digest)!=before


@pytest.mark.parametrize('run_id',['../escape','/absolute','bad/id'])
def test_runtime_preparation_rejects_path_like_run_ids(monkeypatch,tmp_path,run_id):
    monkeypatch.chdir(tmp_path);monkeypatch.setenv('SLURM_JOB_ID','unit-test-allocation')
    with pytest.raises(ValueError,match='run ID'):
        _prepare_tool().prepare('missing.json.gz',run_id)
    assert not (tmp_path/'runs').exists()


@pytest.mark.parametrize('sigma,bridge_weight,noise_weight,expected_passes,expected_epsilon', [
    (0., 0., .1, 1, 0),
    (0., 1., .1, 2, 0),
    (.08, 1., 0., 2, 0),
    (.08, 0., .1, 2, 1),
])
def test_v2_objective_compact_forward(sigma, bridge_weight, noise_weight, expected_passes, expected_epsilon):
    cfg = config()
    cfg.update(sigma_bridge=sigma, frequency_split=True)
    cfg['train'].update(bridge_weight=bridge_weight, noise_weight=noise_weight, rollout_weight=0.)
    model = make_model(cfg)
    objective = V2Objective(model, cfg)
    lr = torch.rand(2, 1, 3, 8, 8)
    hr = torch.rand(2, 1, 3, 32, 32)
    calls = {'endpoint': [], 'refinement': [], 'epsilon': []}
    for name in calls:
        getattr(model, name).register_forward_hook(
            lambda module, args, output, name=name: calls[name].append(output.shape[0]))
    terms = objective(lr, hr)
    terms['loss'].backward()
    assert calls['endpoint'] == [len(lr)]*expected_passes
    assert calls['refinement'] == [len(lr)]
    assert calls['epsilon'] == [len(lr)]*expected_epsilon
    assert ('noise_loss' in terms) == bool(expected_epsilon)
    assert ('bridge_haar' in terms) == ('bridge_pixel' in terms) == bool(bridge_weight)
    assert all(parameter.grad is not None for parameter in objective.parameters() if parameter.requires_grad)


def test_explicit_rollout_uses_deterministic_prefix(monkeypatch):
    from vemgsbm import frequency_bridge

    cfg = config()
    cfg.update(sigma_bridge=.08, frequency_split=True)
    cfg['train'].update(bridge_weight=0., noise_weight=0., rollout_weight=.1, rollout_steps=2)
    model = make_model(cfg)
    calls = []
    integrate = frequency_bridge.integrate_frequency_bridge

    def checked_integrate(*args, **kwargs):
        calls.append(kwargs)
        return integrate(*args, **kwargs)

    monkeypatch.setattr(frequency_bridge, 'integrate_frequency_bridge', checked_integrate)
    terms = V2Objective(model, cfg)(torch.rand(2, 1, 3, 8, 8), torch.rand(2, 1, 3, 32, 32))
    assert len(calls) == 1
    assert calls[0].get('sigma_max', 0.) == 0.
    assert 'curvature' not in calls[0]
    assert 'rollout_haar' in terms and 'rollout_pixel' in terms
    terms['loss'].backward()
    assert model.endpoint.weight.grad.norm() > 0
