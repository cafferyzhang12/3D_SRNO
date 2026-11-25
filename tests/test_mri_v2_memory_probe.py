"""CPU-only budget/CLI tests; these never allocate CUDA tensors."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml


def tool():
    path=Path(__file__).resolve().parents[1]/'tools'/'mri_v2_memory_probe.py'
    spec=importlib.util.spec_from_file_location('v2_memory_probe_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def run_configs(tmp_path):
    root=Path(__file__).resolve().parents[1]
    for i in range(1,5):
        source=next((root/'configs').glob(f'mri_v2_exp{i}_*.yaml'))
        (tmp_path/f'exp{i}.yaml').write_text(source.read_text(encoding='utf-8'),encoding='utf-8')
    return tool().config_paths(tmp_path)


def test_memory_probe_parser_and_source_config_guard():
    m=tool()
    args=m.parser().parse_args(['--configs-dir','run','--output','memory.json'])
    assert args.candidates==(8,16,32,48,64) and args.world_size==4
    assert args.safety==.82
    with pytest.raises(ValueError,match='source configs'):
        m.config_paths(m.ROOT/'configs')


def test_common_budget_uses_heaviest_arm_and_keeps_global_batch():
    m=tool()
    rows=[]
    for batch in (8,16,32,48,64):
        for arm in range(1,5):
            peak=500+batch*arm*18
            rows.append(dict(batch=batch,arm=arm,status='ok',budget_peak_bytes=peak,total_bytes=5000))
    budget=m.choose_budget(rows)
    assert budget['batch_size']==48 and budget['effective_global_batch']==192
    assert budget['accumulation_steps']==1
    limited=[r for r in rows if r['batch']==8]
    assert m.choose_budget(limited)['accumulation_steps']==4
    with pytest.raises(RuntimeError,match='No common'):
        m.choose_budget(limited[:-1])


def test_cuda_unavailable_reports_truth_and_does_not_rewrite_configs(monkeypatch,tmp_path):
    paths=run_configs(tmp_path)
    before={p:p.read_bytes() for p in paths}
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    result=tool().run(tmp_path,tmp_path/'memory.json')
    assert result['status']=='unavailable' and not result['configs_changed']
    assert not any('bytes' in key for key in result)
    assert json.loads((tmp_path/'memory.json').read_text())==result
    assert all(p.read_bytes()==before[p] for p in paths)


def test_updates_only_runtime_batch_and_accumulation(tmp_path):
    m=tool();paths=run_configs(tmp_path);configs=m.read_configs(paths)
    originals=copy.deepcopy(configs)
    m.update_run_configs(paths,configs,dict(batch_size=16,accumulation_steps=2))
    assert configs==originals
    for path,original in zip(paths,originals):
        changed=yaml.safe_load(path.read_text(encoding='utf-8'))
        assert changed['train']['batch_size']==16 and changed['train']['accumulation_steps']==2
        changed['train']['batch_size']=original['train']['batch_size']
        changed['train']['accumulation_steps']=original['train']['accumulation_steps']
        assert changed==original
