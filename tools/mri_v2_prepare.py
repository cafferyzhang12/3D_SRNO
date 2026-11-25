"""Prepare immutable TRAIN/VAL-only runtime configs inside an allocation."""
import argparse
import copy
import gzip
import json
import os
import re
from pathlib import Path

import yaml


def prepare(inventory,run_id):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run data preparation in a Slurm allocation')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,80}',run_id):
        raise ValueError('Invalid run ID')
    root=Path('runs/mri/v2')/run_id
    with gzip.open(inventory,'rt',encoding='utf-8') as f:old=json.load(f)
    if old.get('version') != 1 or any(not old.get('splits',{}).get(s) for s in ('train','val')):
        raise ValueError('Expected an audited nonempty TRAIN/VAL inventory version 1')
    root.mkdir(parents=True,exist_ok=False)
    configs=[]
    for i in range(1,5):
        cfg=yaml.safe_load(next(Path('configs').glob(f'mri_v2_exp{i}_*.yaml')).read_text())
        data=copy.deepcopy(old['data'])
        data.pop('test',None)
        for key in ('scan_cache','legacy_scan_cache'):
            if key in data:data[key].pop('test',None)
        data['patch_size']=cfg['data']['patch_size']
        cfg['data']=data
        cfg['run_dir']=str((root/f'exp{i}').resolve())
        cfg['perceptual']['weights_dir']=str(Path(os.environ.get('MRI_PERCEPTUAL_WEIGHTS','cache/perceptual/torch')).resolve())
        path=root/f'exp{i}.yaml'
        path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
        configs.append(str(path.resolve()))
    # Merely copy TRAIN/VAL metadata. No TEST image is opened or selected.
    with gzip.open(root/'trainval_inventory.json.gz','wt',encoding='utf-8') as f:
        json.dump({'version':1,'data':data,'splits':{s:old['splits'][s] for s in ('train','val')}},f)
    receipt={'configs':configs,'inventory':str((root/'trainval_inventory.json.gz').resolve()),'test_used':False}
    (root/'prepared.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inventory',required=True);p.add_argument('--run-id',required=True)
    a=p.parse_args();prepare(a.inventory,a.run_id)
