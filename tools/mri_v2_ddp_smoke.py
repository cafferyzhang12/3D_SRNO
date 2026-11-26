"""Allocated four-A800, full-size two-update startup gate for all MRI v2 arms."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from vemgsbm.distributed import initialize_distributed
from vemgsbm.mri_v2 import V2Objective, make_model
from vemgsbm.jetatlas import JetAtlasRegularizer
from vemgsbm.runtime import read_config, atomic_save
from vemgsbm.training_state import ModelEMA
from vemgsbm.mri_perceptual import load_extractors
from mri_v2_memory_probe import tiny_atlas_payload


def run(configs_dir,output):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Four-GPU startup checks require a Slurm allocation')
    context=initialize_distributed('cuda')
    try:
        if context.world_size!=4:
            raise RuntimeError('Use torchrun with four ranks on allocated A800s')
        torch.set_num_threads(4)
        records=[]
        for arm in range(1,5):
            cfg=read_config(Path(configs_dir)/f'exp{arm}.yaml')
            torch.manual_seed(cfg['train']['seed'])
            model=make_model(cfg).to(context.device).train()
            model.set_band_statistics(dict(mean=torch.zeros(16),std=torch.full((16,),.1)))
            atlas=(JetAtlasRegularizer.from_payload(tiny_atlas_payload(cfg)).to(context.device)
                   if cfg.get('atlas_weight',0) else None)
            paired,extractor=load_extractors(cfg['perceptual']['weights_dir'],context.device)
            ema=ModelEMA(model,cfg['train']['ema_decay'])
            objective=V2Objective(model,cfg,atlas)
            wrapped=DDP(objective,device_ids=[context.local_rank],find_unused_parameters=True)
            params=[p for p in objective.parameters() if p.requires_grad]
            optimizer=torch.optim.AdamW(params,lr=cfg['train']['learning_rate'])
            accumulation=cfg['train'].get('accumulation_steps',1)
            batch=cfg['train']['batch_size']
            torch.manual_seed(cfg['train']['seed']+context.rank+100)
            lr=torch.rand(batch,1,7,64,64,device=context.device)
            with torch.no_grad():
                base=model.prepare_base(lr)
                x=torch.linspace(-1,1,256,device=context.device)[None,None,None,None,:]
                hr=(base+.01*torch.sin(21*x)).clamp(0,1)
            ids=[f'probe-TRAIN-{i%5}' for i in range(batch)]
            torch.cuda.reset_peak_memory_stats(context.device)
            losses=[]
            for update in range(2):
                optimizer.zero_grad(set_to_none=True)
                for _ in range(accumulation):
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        terms=wrapped(lr,hr,ids)
                        loss=terms['loss']/accumulation
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite distributed startup loss')
                    loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                optimizer.step();ema.update(model)
                losses.append(float(terms['loss'].detach()))
            torch.cuda.synchronize(context.device)
            # Check synchronized parameters after real backward/all-reduce/Adam updates.
            checksum=torch.stack([sum(p.detach().double().sum() for p in params),
                                  sum(p.detach().double().square().sum() for p in params)])
            lower,upper=checksum.clone(),checksum.clone()
            torch.distributed.all_reduce(lower,op=torch.distributed.ReduceOp.MIN)
            torch.distributed.all_reduce(upper,op=torch.distributed.ReduceOp.MAX)
            if not torch.allclose(lower,upper,rtol=1.e-10,atol=1.e-10):
                raise RuntimeError('Parameters diverged between DDP ranks')
            local=dict(rank=context.rank,losses=losses,gradient_norm=float(norm),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(context.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(context.device),parameters_synced=True)
            rows=[None]*4
            torch.distributed.all_gather_object(rows,local)
            records.append(dict(arm=arm,batch_per_gpu=batch,accumulation_steps=accumulation,
                effective_global_batch=batch*4*accumulation,updates=2,ranks=rows))
            context.main_call(lambda:atomic_save(dict(status='running',arms=records),output,json_file=True))
            if context.is_main:print(json.dumps(records[-1]),flush=True)
            del wrapped,objective,optimizer,model,atlas,params,ema,paired,extractor,terms,loss,lr,hr,base
            gc.collect();torch.cuda.empty_cache();context.barrier()
        context.main_call(lambda:atomic_save(dict(status='passed',world_size=4,
            real_mri_quality_claim=False,arms=records),output,json_file=True))
    finally:
        context.cleanup()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configs-dir',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.configs_dir,a.output)
