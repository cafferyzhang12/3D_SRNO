"""Administrative metadata copy for independent exp3 training; opens no images."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import yaml


def prepare(source,destination):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    cfg=yaml.safe_load((source/'exp3.yaml').read_text(encoding='utf-8'))
    protected=yaml.safe_load((source/'exp1.yaml').read_text(encoding='utf-8'))
    protected_run=Path(protected['run_dir']).resolve()
    if destination==source or source in destination.parents or destination==protected_run or protected_run in destination.parents:
        raise ValueError('New study must be separate from the protected exp1 study')
    if not (source/'gate_passed').is_file():
        raise ValueError('Source study has not passed the full-size four-GPU gate')
    with gzip.open(source/'trainval_inventory.json.gz','rt',encoding='utf-8') as stream:
        inventory=json.load(stream)
    if set(inventory['splits'])!={'train','val'} or 'test' in cfg['data'] or inventory['data']!=cfg['data']:
        raise ValueError('Expected matching TRAIN/VAL-only inventory')
    if cfg.get('online_atlas') or not cfg.get('atlas_weight',0) or not cfg.get('frequency_split'):
        raise ValueError('Expected an offline exp3 training configuration')
    if cfg['train'].get('warm_start_checkpoint'):
        raise ValueError('Shared exp3 study must preserve independent from-scratch training')
    destination.mkdir(parents=True,exist_ok=False)
    original=copy.deepcopy(cfg)
    cfg['run_dir']=str(destination/'train')
    (destination/'exp3.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    shutil.copyfile(source/'trainval_inventory.json.gz',destination/'trainval_inventory.json.gz')
    receipt=dict(source_study=str(source),run_dir=cfg['run_dir'],test_used=False,
        preserved_exp1_run=str(protected_run),training_budget=cfg['train'],
        source_exp3_sha256=hashlib.sha256((source/'exp3.yaml').read_bytes()).hexdigest(),
        config_only_changed_run_dir=all(cfg[k]==v for k,v in original.items() if k!='run_dir'))
    (destination/'prepared.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-study',required=True);p.add_argument('--destination',required=True)
    a=p.parse_args();prepare(a.source_study,a.destination)
