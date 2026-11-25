"""Matched SSIM/MLS fine-tuning study with strict whole-VAL PSNR retention."""
import argparse
import copy
import json
import math
from pathlib import Path
import re

from .runtime import atomic_save, fingerprint, load_checkpoint, project_path, read_config
from .mri_x1_pipeline import write_config

VARIANTS = ('control', 'ssim', 'stratified', 'neighbor')


def prepare(baseline, inventory, bank, run_id, epochs=8):
    from .mri_inventory import load_inventory
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,80}', run_id) or epochs < 1:
        raise ValueError('Invalid run ID or epoch budget')
    baseline, bank = Path(baseline).resolve(), Path(bank).resolve()
    old = load_checkpoint(baseline)
    cfg = copy.deepcopy(old['config'])
    if (cfg.get('predict_target') != 'x1' or cfg['train'].get('objective') != 'sr_reconstruction'
            or cfg['sampling']['steps'] != 4 or cfg.get('sigma_0') != 0 or cfg.get('sigma_bridge') != 0):
        raise ValueError('Requires deterministic four-step SR reconstruction baseline')
    cases, _ = load_inventory(inventory, cfg['data'], ('train', 'val'))
    if (len(cases['train']),sum(len(c.lq) for c in cases['train']),len(cases['val']),
            sum(len(c.lq) for c in cases['val'])) != (3919,292657,10,595):
        raise ValueError('Full historical TRAIN/VAL required')
    if not bank.is_file() or not project_path(cfg['band_stats_cache']).is_file():
        raise FileNotFoundError('TRAIN bank/statistics missing')
    root = project_path('runs/mri') / run_id
    cfg['mls_regularizer']['bank'] = str(bank)
    cfg['infer_steps'] = cfg['sampling']['steps'] = 4
    cfg['train'].update(warm_start_checkpoint=str(baseline), selection_metric='ssim_psnr_guard',
        epochs=epochs,bridge_epochs=epochs,schedule_epochs=epochs,minimum_epochs=epochs,
        whole_val_patience=epochs,learning_rate=2.e-5,warmup_epochs=1,
        samples_per_epoch=262144,batch_size=64,rollout_steps=4,checkpoint_epochs=[epochs])
    paths = {}
    for name in VARIANTS:
        candidate = copy.deepcopy(cfg)
        candidate['run_dir'] = str(root/name)
        candidate['train']['ssim_weight'] = 0. if name == 'control' else .3
        candidate['mls_regularizer'].update(
            loss_sampling='structure_stratified' if name in ('stratified','neighbor') else 'uniform',
            structure_fractions=[.1,.2,.5,.2],neighbor_weight=.1 if name == 'neighbor' else 0.)
        paths[name] = str(root/f'{name}.yaml')
        write_config(paths[name],candidate)
    smoke = copy.deepcopy(candidate)
    smoke['run_dir'] = str(root/'smoke')
    smoke['train'].update(epochs=1,bridge_epochs=1,samples_per_epoch=1024,minimum_epochs=1,
                          whole_val_patience=1,checkpoint_epochs=[1])
    write_config(root/'smoke.yaml',smoke)
    atomic_save(dict(baseline=str(baseline),baseline_sha256=fingerprint(baseline),
        inventory=str(Path(inventory).resolve()),inventory_sha256=fingerprint(inventory),
        variants=paths,epochs=epochs,selection='max whole-VAL SSIM with PSNR >= frozen baseline; fallback baseline',
        test_used_for_selection=False),root/'study.json',json_file=True)


def choose(rows, baseline):
    """No dB tolerance: do not silently relax the requested PSNR constraint."""
    required=('psnr','ssim')
    for row in [baseline,*rows]:
        if any(not math.isfinite(row[k]) for k in required):
            raise ValueError('Nonfinite selection metrics')
    eligible=[r for r in rows if r['psnr'] >= baseline['psnr'] and r['ssim'] > baseline['ssim']]
    return max(eligible,key=lambda r:(r['ssim'],r['psnr'])) if eligible else baseline


def check_smoke(config):
    from .mri_x1_pipeline import check_sr_smoke
    check_sr_smoke(config)
    cfg=read_config(config)
    folder=project_path(cfg['run_dir'])/'bridge'
    row=json.loads((folder/'metrics.jsonl').read_text().splitlines()[-1])
    for prefix in ('start','rollout','intermediate'):
        for key in ('ssim_loss','mls_local_mse','mls_neighbor_mse','mls_sample_fraction_texture'):
            if not math.isfinite(row[f'{prefix}_{key}']):
                raise ValueError('Nonfinite structural smoke terms')
        if row[f'{prefix}_ssim_items'] != 8 or row[f'{prefix}_mls_neighbor_pairs'] <= 0:
            raise ValueError('Structural smoke did not exercise subset/neighbor supervision')
    atomic_save(dict(passed=True,record=row),folder/'structural_smoke_gate.json',json_file=True)


def select(study):
    root=Path(study).resolve().parent
    spec=json.loads(Path(study).read_text())
    if fingerprint(spec['baseline']) != spec['baseline_sha256']:
        raise ValueError('Frozen baseline changed')
    rows=[]
    baseline=None
    for name,path in spec['variants'].items():
        cfg=read_config(path)
        folder=project_path(cfg['run_dir'])/'bridge'
        history=[json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
        if history[-1]['epoch'] != spec['epochs']:
            raise ValueError('Every matched ablation must complete its budget')
        ck=load_checkpoint(folder/'best.pt')
        anchor=ck['baseline_validation']
        b=dict(name='baseline',checkpoint=spec['baseline'],psnr=anchor['val_pooled_psnr_db'],ssim=anchor['val_mean_slice_ssim'])
        if baseline is not None and (abs(b['psnr']-baseline['psnr'])>1.e-8 or abs(b['ssim']-baseline['ssim'])>1.e-10):
            raise ValueError('Ablations disagree on the fixed baseline')
        baseline=b
        # best.pt may intentionally retain the epoch-zero warm-start fallback.
        if ck.get('selection_origin') == 'warm_start_fallback':
            score=anchor
        else:
            score=next(r for r in history if r['epoch']==ck['epoch'])
        rows.append(dict(name=name,checkpoint=str(folder/'best.pt'),epoch=ck['epoch'],
                         psnr=score['val_pooled_psnr_db'],ssim=score['val_mean_slice_ssim']))
    selected=choose(rows,baseline)
    result=dict(baseline=baseline,ablations=rows,selected=selected,
        improved=selected['name']!='baseline',constraint='VAL PSNR >= baseline, SSIM > baseline',
        test_used_for_selection=False)
    atomic_save(result,root/'selection.json',json_file=True)
    (root/'selected_checkpoint.txt').write_text(selected['checkpoint']+'\n')
    print(json.dumps(result),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    for key in ('baseline','inventory','bank','run-id'):p.add_argument('--'+key,required=True)
    p.add_argument('--epochs',type=int,default=8)
    p=sub.add_parser('select');p.add_argument('--study',required=True)
    p=sub.add_parser('check-smoke');p.add_argument('--config',required=True)
    args=vars(ap.parse_args());command=args.pop('command')
    {'prepare':prepare,'select':select,'check-smoke':check_smoke}[command](**args)


if __name__=='__main__':main()
