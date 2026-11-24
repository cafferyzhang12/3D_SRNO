"""Deployment contracts for paired, deterministic SR refinement."""
import copy
from pathlib import Path
import pytest
import yaml
from vemgsbm.runtime import read_config


def config():
    cfg = yaml.safe_load(Path('configs/mri_haar_bridge_x4.yaml').read_text())
    cfg.update(sigma_0=0., sigma_bridge=0., infer_steps=4)
    cfg['path']['sigma'] = 0.
    cfg['sampling'].update(sigma=0., steps=4, sampler_type='ode', solver='euler')
    cfg['train'].update(objective='sr_reconstruction', rollout_steps=4)
    return cfg


def load(tmp_path, cfg):
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    return read_config(path)


def test_deterministic_sr_contract(tmp_path):
    assert load(tmp_path, config())['sigma_0'] == 0


@pytest.mark.parametrize('kind', ['source_noise','bridge_noise','grid','teacher','no_start','negative_hf'])
def test_reject_inconsistent_sr_contract(tmp_path, kind):
    cfg = copy.deepcopy(config())
    if kind == 'source_noise': cfg['sigma_0'] = .03
    if kind == 'bridge_noise':
        cfg['sigma_bridge'] = cfg['path']['sigma'] = cfg['sampling']['sigma'] = .08
    if kind == 'grid': cfg['train']['rollout_steps'] = 3
    if kind == 'teacher': cfg['straightening'] = {'teacher_checkpoint':'unused.pt'}
    if kind == 'no_start': cfg['train']['start_weight'] = 0
    if kind == 'negative_hf': cfg['train']['hf_weight'] = -.1
    with pytest.raises(ValueError): load(tmp_path, cfg)


def test_legacy_bridge_keeps_nondegenerate_source(tmp_path):
    cfg = config()
    cfg['train'].pop('objective')
    with pytest.raises(ValueError, match='positive sigma_0'): load(tmp_path, cfg)
