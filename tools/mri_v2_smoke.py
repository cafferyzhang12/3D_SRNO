"""Bounded four-ablation integration diagnostic; no MRI files or TEST are read.

The phantom deliberately makes missing contrast recoverable from LR. Passing
this 20-step check demonstrates optimization/gradient plumbing, not clinical
generalization or an achievable whole-VAL SSIM ceiling.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import torch
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vemgsbm.hybrid import upsample_xy
from vemgsbm.mri_metrics import mean_xy_ssim
from vemgsbm.mri_v2 import make_model,V2Objective
from vemgsbm.mri_v2_diagnostics import local_variance_ratio
from vemgsbm.jetatlas import hf_patches,JetAtlasRegularizer


CONFIGS=('mri_v2_exp1_backbone_ssim.yaml','mri_v2_exp2_freq_gsbm_no_atlas.yaml',
         'mri_v2_exp3_freq_gsbm_jetatlas_train.yaml','mri_v2_exp4_freq_gsbm_jetatlas_full.yaml')


def phantom(device,case=0):
    yy,xx=torch.meshgrid(torch.linspace(0,1,8,device=device),torch.linspace(0,1,8,device=device),indexing='ij')
    lr=(.5+.12*torch.sin(2*torch.pi*xx+.07*case)+.09*torch.cos(2*torch.pi*yy-.05*case))[None,None,None]
    lr=lr.repeat(1,1,3,1,1)
    # Low-amplitude independent Z texture avoids rank-deficient chart banks.
    gen=torch.Generator(device=device).manual_seed(31+case)
    lr=lr+.0001*torch.randn(lr.shape,device=device,generator=gen)
    return lr,.5+1.5*(upsample_xy(lr)-.5)


def build_atlas(model,device):
    bank,ids,sources=[],[],[]
    with torch.no_grad():
        for index in range(5):
            lr,hr=phantom(device,index)
            truth=model.normalize_haar(model.codec.encode(hr-model.prepare_base(lr)))
            rows=hf_patches(truth)
            sites=torch.linspace(0,len(rows)-1,64,device=device).long()
            bank.append(rows[sites]);ids.extend([f'synthetic_train_{index}']*64)
            sources.append(lr.permute(0,2,3,4,1).reshape(-1,1)[sites])
        return JetAtlasRegularizer.fit_from_bank(torch.cat(bank),ids,
            dict(folds=5,projection_dim=4,intrinsic=2,centers=4,neighbors=48,
                 normal_rank=2,query_sites=16,query_chunk=256,top_m=3,
                 router_hidden=16,lr_dim=1,seed=42,normal_weight=1.,tangent_weight=.05,
                 neighbor_weight=.1,potential_weight=.02,router_weight=.05),
            lr_features=torch.cat(sources),split='train').to(device)


@torch.no_grad()
def metrics(model,lr,hr,steps,atlas=None,sampler='ode'):
    prediction=model.restore(lr,model.prepare_base(lr),steps=steps,atlas=atlas,sampler_type=sampler,
        generator=torch.Generator(device=lr.device).manual_seed(42))
    mse=(prediction-hr).double().square().mean()
    return dict(psnr=float(-10*torch.log10(mse)),ssim=float(mean_xy_ssim(prediction,hr)),
                local_variance_ratio=local_variance_ratio(prediction,hr),
                finite=bool(torch.isfinite(prediction).all()),max_abs=float(prediction.abs().max()))


def gradient_probe(terms,model,atlas=None):
    params=tuple(p for p in model.parameters() if p.requires_grad)
    result={}
    for name in ('start_ssim','start_local_std','jet_potential','jet_asym','router_kl','noise_loss'):
        if name not in terms or not terms[name].requires_grad:
            continue
        grads=torch.autograd.grad(terms[name],params,retain_graph=True,allow_unused=True)
        norm=sum(float(g.detach().double().square().sum()) for g in grads if g is not None)**.5
        result[name]=dict(raw_model_gradient_norm=norm,finite=bool(torch.isfinite(terms[name])))
        if name=='router_kl' and atlas is not None:
            router_params=tuple(p for p in atlas.parameters() if p.requires_grad)
            router_grads=torch.autograd.grad(terms[name],router_params,retain_graph=True,allow_unused=True)
            result[name]['raw_router_gradient_norm']=sum(float(g.detach().double().square().sum())
                for g in router_grads if g is not None)**.5
    return result


def run(args):
    torch.set_num_threads(args.threads)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable')
    root=Path(__file__).resolve().parents[1]
    extractors=None
    error='No verified pretrained perceptual weights supplied'
    if args.weights_dir:
        try:
            from vemgsbm.mri_perceptual import load_extractors
            extractors=load_extractors(args.weights_dir,device)
            error=None
        except Exception as exc:
            if args.require_perceptual:
                raise
            error=f'{type(exc).__name__}: {exc}'
    elif args.require_perceptual:
        raise ValueError('--require-perceptual needs --weights-dir')
    report=dict(device=str(device),torch=torch.__version__,synthetic=True,
        test_accessed=False,steps=args.steps,
        memory_note='CUDA allocator peaks measured on CUDA; CPU has no GPU allocation and reports null',
        phantom='TRAIN synthetic contrast-recovery phantom, LR 3x8x8 to HR 3x32x32',
        overrides=dict(widths=[4,8,12],blocks=[1,1,1],heads=2,refine_width=4,
            gradient_checkpointing=False,use_band_whitening=False,learning_rate=.002),variants={})
    for index,filename in enumerate(CONFIGS):
        torch.manual_seed(31)
        started=time.perf_counter()
        cfg=yaml.safe_load((root/'configs'/filename).read_text(encoding='utf-8'))
        cfg=copy.deepcopy(cfg)
        cfg['model'].update(widths=[4,8,12],blocks=[1,1,1],heads=2,refine_width=4,gradient_checkpointing=False)
        cfg['use_band_whitening']=False
        cfg['train']['learning_rate']=.002
        model=make_model(cfg).to(device)
        lr,hr=phantom(device)
        atlas=build_atlas(model,device) if cfg.get('atlas_weight',0) else None
        objective=V2Objective(model,cfg,atlas=atlas)
        parameters=list(model.parameters())+([] if atlas is None else list(atlas.parameters()))
        optimizer=torch.optim.Adam(parameters,lr=.002)
        if device.type=='cuda':
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
        curve=[]
        diagnostics={}
        for step in range(args.steps+1):
            row=metrics(model,lr,hr,1)
            row['step']=step;curve.append(row)
            if step==args.steps:
                break
            terms=objective(lr,hr,['synthetic_train_0'])
            if step==0:
                diagnostics=gradient_probe(terms,model,atlas)
            optimizer.zero_grad(set_to_none=True)
            terms['loss'].backward()
            if not all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None):
                raise FloatingPointError(f'{filename}: nonfinite gradient')
            optimizer.step()
            row['training_loss']=float(terms['loss'].detach())
        online=atlas if cfg.get('online_atlas') else None
        stability={mode:{str(n):metrics(model,lr,hr,n,online,mode) for n in (1,4,8,16)}
                   for mode in ('ode','sde')}
        perceptual=dict(status='unavailable',reason=error,lpips=None,fid=None)
        if extractors is not None:
            from mri_ceiling_audit import perceptual_update
            from vemgsbm.mri_perceptual import FeatureMoments,frechet_distance
            with torch.no_grad():
                prediction=model.restore(lr,model.prepare_base(lr),8,atlas=online).clamp(0,1)
                moments=(FeatureMoments(),FeatureMoments())
                total,count=perceptual_update(prediction,hr,extractors,moments,batch_size=3)
                perceptual=dict(status='measured_pretrained',lpips=total/count,
                                fid=frechet_distance(*moments),slices=count,dimension=2048)
        record=dict(curve=curve,step_stability=stability,gradient_probe=diagnostics,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,
            perceptual=perceptual,atlas=None if atlas is None else atlas.diagnostics(),
            seconds=time.perf_counter()-started,
            overfit_passed=curve[-1]['ssim']>.98,
            inference_stable=all(row['finite'] for mode in stability.values() for row in mode.values()))
        report['variants'][f'exp{index+1}']=record
        print(json.dumps(dict(variant=f'exp{index+1}',initial=curve[0],final=curve[-1],
            overfit_passed=record['overfit_passed'],gradient_probe=diagnostics,
            peak_cuda_allocated_bytes=record['peak_cuda_allocated_bytes'],
            ode=stability['ode'],perceptual=perceptual)),flush=True)
        # Preserve evidence even if a later ablation fails.
        destination=Path(args.output)
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        del objective,optimizer,atlas,model
    if not all(v['overfit_passed'] and v['inference_stable'] for v in report['variants'].values()):
        raise RuntimeError('Four-arm smoke failed overfit SSIM>.98 or finite inference gate; see JSON')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--weights-dir')
    parser.add_argument('--require-perceptual',action='store_true')
    parser.add_argument('--output',default='reports/mri/v2_smoke.json')
    run(parser.parse_args())
