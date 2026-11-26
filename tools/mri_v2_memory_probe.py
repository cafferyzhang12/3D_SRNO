"""A800 allocation-only full-size MRI v2 forward/backward memory calibration.

Each candidate runs in a fresh process so OOM cannot contaminate later trials.
All four run YAMLs receive the same microbatch/accumulation after successful
calibration. Source configs are never rewritten. Synthetic data checks memory
and finite gradients; it does not estimate MRI quality or peak VAL memory.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

import yaml


ROOT=Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES=(8,16,32,48,64)


def config_paths(directory):
    directory=Path(directory).resolve()
    if directory == (ROOT/'configs').resolve():
        raise ValueError('Pass prepared run YAMLs; source configs may not be calibrated in place')
    paths=[directory/f'exp{i}.yaml' for i in range(1,5)]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError('Expected exp1.yaml through exp4.yaml in --configs-dir')
    return paths


def read_configs(paths):
    configs=[yaml.safe_load(path.read_text(encoding='utf-8')) for path in paths]
    for cfg in configs:
        if (cfg['model']['architecture'] != 'HybridWaveletNAFNet25D'
            or cfg['model']['widths'] != [64,128,192]
            or cfg['data']['patch_size'] != [7,256,256]
            or not cfg['model'].get('gradient_checkpointing',False)
            or cfg['train'].get('amp_dtype') != 'bfloat16'):
            raise ValueError('Probe requires formal widths64/128/192, checkpoint bf16 and 7x256x256 HR')
        if cfg['loss'].get('lpips',0):
            raise ValueError('Probe currently supports the formal LPIPS-reporting-only training budget')
    for key in ('epochs','samples_per_epoch','learning_rate'):
        if len({cfg['train'][key] for cfg in configs}) != 1:
            raise ValueError(f'Four ablations differ in optimization budget: {key}')
    return configs


def choose_budget(rows,candidates=DEFAULT_CANDIDATES,safety=.82,world_size=4):
    """Largest measured common batch with global update batch in [128,256]."""
    eligible=[]
    for batch in candidates:
        trial=[row for row in rows if row['batch']==batch]
        if len(trial)!=4 or {row['arm'] for row in trial}!={1,2,3,4}:
            continue
        if all(row.get('status')=='ok' and row['budget_peak_bytes'] <= safety*row['total_bytes'] for row in trial):
            accumulation=max(1,math.ceil(128/(world_size*batch)))
            effective=world_size*batch*accumulation
            if 128 <= effective <= 256:
                eligible.append((batch,accumulation,effective))
    if not eligible:
        raise RuntimeError('No common measured microbatch fits the 82% GPU memory ceiling and global batch128-256')
    batch,accumulation,effective=max(eligible)
    return dict(batch_size=batch,accumulation_steps=accumulation,effective_global_batch=effective,
                world_size=world_size,safety_fraction=safety)


def update_run_configs(paths,configs,budget):
    # Validate all mutations before replacing any run configuration.
    written=[]
    try:
        for path,cfg in zip(paths,configs):
            updated=copy.deepcopy(cfg)
            updated['train']['batch_size']=budget['batch_size']
            updated['train']['accumulation_steps']=budget['accumulation_steps']
            with tempfile.NamedTemporaryFile('w',dir=path.parent,encoding='utf-8',delete=False,suffix='.yaml.tmp') as stream:
                yaml.safe_dump(updated,stream,sort_keys=False)
                written.append((Path(stream.name),path))
        for source,destination in written:
            source.replace(destination)
    finally:
        for source,_ in written:
            source.unlink(missing_ok=True)


def tiny_atlas_payload(cfg):
    """Small synthetic TRAIN bank with full d8/PCA32 geometry and fold separation."""
    import torch
    from vemgsbm.jetatlas import JetAtlasRegularizer
    torch.manual_seed(43)
    rows=torch.randn(5*96,135)*.15
    cases=[f'probe-TRAIN-{i}' for i in range(5) for _ in range(96)]
    settings=copy.deepcopy(cfg['jetatlas'])
    settings.update(centers=4,projection_dim=32,intrinsic=8,folds=5,neighbors=256)
    atlas=JetAtlasRegularizer.fit_from_bank(rows,cases,settings,split='train')
    return atlas.payload()


def worker(config,batch,output,atlas_file):
    import torch
    from vemgsbm.jetatlas import JetAtlasRegularizer
    from vemgsbm.mri_v2 import V2Objective,make_model
    from vemgsbm.mri_perceptual import load_extractors
    from vemgsbm.training_state import ModelEMA
    torch.set_num_threads(4)
    cfg=yaml.safe_load(Path(config).read_text(encoding='utf-8'))
    device=torch.device('cuda',0)
    torch.cuda.set_device(device)
    torch.manual_seed(43)
    torch.cuda.manual_seed_all(43)
    total=torch.cuda.get_device_properties(device).total_memory
    torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter()
    result=dict(batch=batch,status='error',total_bytes=total,gpu=torch.cuda.get_device_name(device),
                lr_shape=[batch,1,7,64,64],hr_shape=[batch,1,7,256,256],amp='bfloat16',checkpointing=True,
                torch_version=str(torch.__version__),cuda_version=torch.version.cuda)
    phase='setup'
    try:
        model=make_model(cfg).to(device).train()
        model.set_band_statistics(dict(mean=torch.zeros(16),std=torch.ones(16)*.1))
        atlas=None
        if cfg.get('atlas_weight',0):
            atlas=JetAtlasRegularizer.from_payload(torch.load(atlas_file,map_location='cpu',weights_only=True)).to(device)
        # Training holds EMA and both pretrained reporting extractors persistently.
        ema=ModelEMA(model,cfg['train'].get('ema_decay',.999))
        paired,extractor=load_extractors(cfg['perceptual']['weights_dir'],device)
        paired.eval();extractor.eval()
        objective=V2Objective(model,cfg,atlas)
        parameters=[p for p in objective.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=cfg['train']['learning_rate'],weight_decay=cfg['train'].get('weight_decay',1.e-4))
        # Reserve the missing full-atlas buffers/router parameters and DDP buckets.
        # Query tensors retain full d8/PCA32/top3/site budgets; chart fit uses four
        # synthetic charts solely to bound fitting cost in this memory diagnostic.
        ddp_reserve=sum(p.numel()*p.element_size() for p in parameters)
        atlas_reserve=0
        if atlas is not None:
            wanted=int(cfg['jetatlas'].get('centers',512))
            for fitted,router in zip(atlas.atlases,atlas.routers):
                per_chart=sum(v.numel()*v.element_size() for v in fitted.buffers())/fitted.chart_count
                missing=max(0,wanted-fitted.chart_count)
                # Full router output weights/grad and Adam's two moments.
                hidden=int(cfg['jetatlas'].get('router_hidden',64))
                atlas_reserve+=math.ceil(missing*(per_chart+4*4*(hidden+1)))
        reserve=torch.empty(ddp_reserve+atlas_reserve,device=device,dtype=torch.uint8)
        # Structured deterministic target plus a recoverable HR perturbation.
        lr=torch.rand(batch,1,7,64,64,device=device)
        with torch.no_grad():
            base=model.prepare_base(lr)
            yy=torch.linspace(-1,1,256,device=device)[None,None,None,:,None]
            xx=torch.linspace(-1,1,256,device=device)[None,None,None,None,:]
            hr=(base+.015*torch.sin(27*xx)*torch.cos(23*yy)).clamp(0,1)
        ids=[f'probe-TRAIN-{i%5}' for i in range(batch)]
        phase='forward'
        with torch.autocast('cuda',dtype=torch.bfloat16):
            terms=objective(lr,hr,ids)
            loss=terms['loss']
        torch.cuda.synchronize(device)
        result['forward_peak_bytes']=torch.cuda.max_memory_allocated(device)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite forward loss')
        phase='backward'
        loss.backward()
        phase='optimizer'
        norm=torch.nn.utils.clip_grad_norm_(parameters,cfg['train'].get('clip_grad_norm',1.),error_if_nonfinite=True)
        optimizer.step();ema.update(model)
        torch.cuda.synchronize(device)
        allocated=torch.cuda.max_memory_allocated(device)
        reserved=torch.cuda.max_memory_reserved(device)
        result.update(status='ok',loss=float(loss.detach()),gradient_norm=float(norm),
            peak_allocated_bytes=allocated,peak_reserved_bytes=reserved,budget_peak_bytes=max(allocated,reserved),
            ddp_bucket_reserve_bytes=ddp_reserve,full_atlas_extra_reserve_bytes=atlas_reserve,
            elapsed_seconds=time.perf_counter()-started)
        # Keep the reserved memory alive until all synchronized measurements finish.
        result['persistent_reserved_tensor_bytes']=reserve.numel()
    except torch.OutOfMemoryError:
        result.update(status='oom',peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),budget_peak_bytes=total+1,
            elapsed_seconds=time.perf_counter()-started)
    except Exception as exc:
        result.update(status='error',error=f'{type(exc).__name__}: {exc}',budget_peak_bytes=total+1,
            failure_phase=phase,traceback=traceback.format_exc(),
            elapsed_seconds=time.perf_counter()-started)
    Path(output).write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


def run(configs_dir,output,*,candidates=DEFAULT_CANDIDATES,safety=.82,world_size=4):
    import torch
    paths=config_paths(configs_dir)
    configs=read_configs(paths)
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():
        result=dict(status='unavailable',reason='CUDA unavailable; no GPU memory values inferred',
                    configured_runs=[str(p) for p in paths],configs_changed=False)
        output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result
    if os.name=='posix' and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run CUDA calibration through sbatch on an allocated A800 node')
    if 'A800' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Formal calibration requires an allocated NVIDIA A800')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('Formal calibration requires bf16 support')
    if not 0 < safety <= .82 or world_size!=4 or any(type(b) is not int or b<1 for b in candidates):
        raise ValueError('Use safety<=.82, four ranks and positive candidate batches')
    rows=[]
    total=torch.cuda.get_device_properties(0).total_memory
    # Probe 32 first. Expand to 48/64 only when the measured trajectory can fit;
    # falling back through16/8 avoids repeated predictable OOM allocations.
    order=[b for b in (32,16,8,48,64) if b in candidates]
    order.extend(b for b in candidates if b not in order)
    with tempfile.TemporaryDirectory(dir=output.parent,prefix='memory_probe_') as temp:
        atlas_file=Path(temp)/'tiny_train_atlas.pt'
        torch.save(tiny_atlas_payload(configs[2]),atlas_file)
        for batch in order:
            # Once a larger common batch fits, smaller trials add no information.
            if any(all(any(r['batch']==larger and r['arm']==arm and r['status']=='ok'
                           and r['budget_peak_bytes']<=safety*r['total_bytes'] for r in rows)
                       for arm in range(1,5)) for larger in candidates if larger>batch):
                continue
            if batch>32:
                previous=[r for r in rows if r['status']=='ok' and r['batch']<batch]
                by_arm={arm:[r for r in previous if r['arm']==arm] for arm in range(1,5)}
                # Conservatively extrapolate total live memory, including fixed
                # components, rather than assigning an invented measured peak.
                predicted=max((max(r['budget_peak_bytes']/r['batch'] for r in arm_rows)*batch
                               for arm_rows in by_arm.values() if arm_rows),default=math.inf)
                if any(not arm_rows for arm_rows in by_arm.values()) or predicted>.9*total:
                    for arm in range(1,5):
                        rows.append(dict(arm=arm,batch=batch,status='skipped_predicted_oom',total_bytes=total,
                                         prediction_only_bytes=None if not math.isfinite(predicted) else predicted))
                    continue
            for arm,path in enumerate(paths,1):
                trial_file=Path(temp)/f'exp{arm}_b{batch}.json'
                command=[sys.executable,str(Path(__file__).resolve()),'--worker','--config',str(path),
                         '--batch',str(batch),'--output',str(trial_file),'--atlas-file',str(atlas_file)]
                completed=subprocess.run(command,cwd=ROOT,check=False,capture_output=True,text=True)
                if trial_file.exists():
                    row=json.loads(trial_file.read_text(encoding='utf-8'))
                else:
                    row=dict(batch=batch,status='error',total_bytes=total,budget_peak_bytes=total+1,
                             error=f'Worker exit {completed.returncode}: {completed.stderr[-2000:]}')
                row['arm']=arm
                rows.append(row)
                print(json.dumps(row,allow_nan=False),flush=True)
                output.write_text(json.dumps(dict(status='running',trials=rows,configs_changed=False),indent=2),encoding='utf-8')
                if row['status']=='error':
                    raise RuntimeError(f'Memory probe worker error for exp{arm}, batch{batch}: {row.get("error")}')
    budget=choose_budget(rows,candidates,safety,world_size)
    update_run_configs(paths,configs,budget)
    result=dict(status='ok',trials=rows,budget=budget,configs_changed=True,
        limitation='Synthetic forward/backward plus optimizer/EMA/extractor/DDP/full-atlas reserves; VAL inference peak measured separately')
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configs-dir');p.add_argument('--output',required=True)
    p.add_argument('--candidates',type=lambda s:tuple(int(v) for v in s.split(',')),default=DEFAULT_CANDIDATES)
    p.add_argument('--safety',type=float,default=.82);p.add_argument('--world-size',type=int,default=4)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--config',help=argparse.SUPPRESS);p.add_argument('--batch',type=int,help=argparse.SUPPRESS)
    p.add_argument('--atlas-file',help=argparse.SUPPRESS)
    return p


def main():
    a=parser().parse_args()
    if a.worker:
        result=worker(a.config,a.batch,a.output,a.atlas_file)
    else:
        if a.configs_dir is None:parser().error('--configs-dir is required')
        result=run(a.configs_dir,a.output,candidates=a.candidates,safety=a.safety,world_size=a.world_size)
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
