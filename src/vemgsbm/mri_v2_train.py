"""Four-metric, TRAIN/VAL-only training for the opt-in MRI v2 ablations."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .datasets import training_data
from .distributed import initialize_distributed
from .mri_v2 import V2Objective, make_model, soft_pareto, inference_predictor
from .runtime import atomic_save, read_config, project_path, load_checkpoint, setup
from .training_state import ModelEMA, warmup_cosine


def geometry_contract(cfg, digest):
    return dict(manifest=digest, base=cfg['base_interpolation'],
        whitening=cfg.get('use_band_whitening',True),patch=cfg['data']['patch_size'],
        atlas=cfg.get('jetatlas'),atlas_weight=cfg.get('atlas_weight',0),seed=cfg['train']['seed'])


def prepare_train_geometry(model, dataset, cfg, device, digest):
    """Deterministic case-balanced TRAIN bank. No VAL tensors or cached old bases."""
    from .jetatlas import hf_patches, JetAtlasRegularizer
    if dataset.split != 'train':
        raise ValueError('Only TRAIN may fit whitening/PCA/router/atlas')
    ds = copy.copy(dataset)
    ds.samples_per_epoch = None
    ds.samples_per_volume = cfg.get('jetatlas',{}).get('patches_per_case',1)
    ds.set_epoch(0)
    count, total, squared = 0, torch.zeros(16,dtype=torch.float64), torch.zeros(16,dtype=torch.float64)
    # Store selected coefficient neighborhoods before whitening; fold PCA is fitted later.
    patches, lows, cases = [], [], []
    settings = cfg.get('jetatlas',{})
    sites = settings.get('sites_per_case',64)
    gen = torch.Generator().manual_seed(cfg['train']['seed'])
    model.eval()
    with torch.no_grad():
        for idx in range(len(ds)):
            item = ds[idx]
            lr, hr = item['lr'][None].to(device), item['hr'][None].to(device)
            coeff = model.codec.encode(hr-model.prepare_base(lr))
            flat = coeff.movedim(1,0).reshape(16,-1).double().cpu()
            count += flat.shape[1]
            total += flat.sum(1)
            squared += flat.square().sum(1)
            if cfg.get('atlas_weight',0):
                rows = hf_patches(coeff).cpu()
                selected = torch.randperm(len(rows), generator=gen)[:sites]
                patches.append(rows[selected])
                # Router uses raw LR+state, so its conditioning does not become stale as the encoder learns.
                low_rows = lr.movedim(1,-1).reshape(-1,1).cpu()
                lows.append(low_rows[selected])
                cases.extend([item['id']]*len(selected))
            if idx % 100 == 0:
                print(json.dumps({'TRAIN_bank_patches':idx+1,'total':len(ds)}),flush=True)
    mean = (total/count).float()
    std = (squared/count-(total/count).square()).clamp_min(1.e-8).sqrt().float()
    stats = {'mean':mean,'std':std,'identity':{'split':'train','manifest':digest,
        'base':cfg['base_interpolation'],'patch_size':cfg['data']['patch_size'],
        'patches_per_case':ds.samples_per_volume,'seed':cfg['train']['seed']},'coefficient_count':count}
    model.set_band_statistics(stats)
    atlas = None
    if cfg.get('atlas_weight',0):
        rows = torch.cat(patches)
        # hf_patches flattens channel-major 3x3 neighborhoods.
        if cfg.get('use_band_whitening',True):
            rows = (rows-mean[1:].repeat_interleave(9))/(std[1:].repeat_interleave(9)+1.e-6)
        atlas = JetAtlasRegularizer.fit_from_bank(rows, cases, settings,
            lr_features=torch.cat(lows),split='train')
    return {'statistics':stats, 'jetatlas':None if atlas is None else atlas.payload(),
        'manifest':digest,'contract':geometry_contract(cfg,digest)}


@torch.no_grad()
def validate(model, cases, cfg, context, directory, atlas=None, extractors=None, steps=None):
    """Unchanged Hann inference, pooled SSE, equal slices, exact perceptual models."""
    from .mri_infer import _infer_case
    from .mri_data import read_plane
    from .mri_perceptual import FeatureMoments, consume_case, frechet_distance, load_extractors
    from .mri_v2_diagnostics import local_variance_ratio, textured_region_ssim
    paired, features = extractors or load_extractors(project_path(cfg['perceptual']['weights_dir']),context.device)
    predicted, reference = FeatureMoments(), FeatureMoments()
    totals = np.zeros(9,dtype=np.float64)
    records = []
    error = None
    try:
        for case in cases[context.rank::context.world_size]:
            with tempfile.TemporaryDirectory(dir=directory) as temp:
                folder = Path(temp)/'case'
                shape, row = _infer_case(case,folder,model,None,cfg,context.device,
                    cfg['train']['validation_seed'],steps or cfg['sampling']['steps'],1,
                    volume_predictor=inference_predictor(model,cfg,atlas,steps))
                volume = np.load(folder/'mean.npy',mmap_mode='r')
                perceptual = consume_case(volume,case,paired,features,predicted,reference,
                    context.device,cfg['perceptual'].get('batch_size',8))
                # Additive local-variance totals (no average of case ratios).
                var_p, var_y, texture_sum, texture_count = 0.,0.,0.,0
                from .mri_v2_losses import local_moments
                for z,path in enumerate(case.hq):
                    y = torch.from_numpy(read_plane(path,case.normalization_scale,
                        case.normalization_offset,case.channel_policy,case.bits))[None,None,None].to(context.device)
                    p = torch.from_numpy(np.array(volume[z],copy=True))[None,None,None].to(context.device)
                    # Independently additive std-square energies for pooled variance ratio.
                    _, vp = local_moments(p)
                    _, vy = local_moments(y)
                    var_p += float(vp.double().sum())
                    var_y += float(vy.double().sum())
                    value = textured_region_ssim(p,y)
                    if value is not None and math.isfinite(float(value)):
                        texture_sum += float(value); texture_count += 1
                totals += [row['squared_error_sum'],row['pixel_count'],
                    row['mean_slice_ssim']*row['ssim_slice_count'],row['ssim_slice_count'],
                    perceptual['lpips_sum'],var_p,var_y,texture_sum,texture_count]
                records.append({'case_id':case.id,**{k:v for k,v in row.items() if k!='slices'},**perceptual})
                del volume
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    context.check_errors(error)
    result = context.sum(totals.tolist())
    predicted.reduce(context); reference.reduce(context)
    # FID eigensolve once on rank zero, then broadcast identical model-selection metrics.
    fid = context.main_call(lambda:frechet_distance(predicted,reference),broadcast_result=True)
    mse = result[0]/result[1]
    metrics = dict(psnr=-10*math.log10(max(mse,1.e-30)),ssim=result[2]/result[3],
        lpips=result[4]/result[3],fid=fid,local_variance_ratio=result[5]/max(result[6],1.e-30),
        textured_region_ssim=result[7]/max(result[8],1),pixel_count=int(result[1]),slice_count=int(result[3]))
    if context.is_distributed:
        gathered = [None]*context.world_size
        torch.distributed.all_gather_object(gathered,records)
        records = [r for shard in gathered for r in shard]
    if sorted(r['case_id'] for r in records) != sorted(c.id for c in cases):
        raise RuntimeError('Incomplete or duplicated whole-case VAL coverage')
    return metrics, records


def run(config, inventory=None, device='auto', resume=None, evaluate=False):
    from .jetatlas import JetAtlasRegularizer
    from .mri_perceptual import load_extractors
    cfg = read_config(config)
    context = initialize_distributed(device)
    run_dir = project_path(cfg['run_dir'])
    run_dir.mkdir(parents=True,exist_ok=True)
    lock = None
    try:
        def acquire():
            nonlocal lock
            if os.name == 'posix':
                import fcntl
                lock = open(run_dir/'.v2.lock','a')
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (run_dir/'last.pt').exists() and not resume:
                raise FileExistsError('Run already exists; use --resume')
        context.main_call(acquire)
        setup(cfg['train']['seed'],str(context.device),cfg['train'].get('threads',4))
        train, val, digest, identity = training_data(cfg,inventory)
        if any(row['split'] == 'test' for row in identity):
            raise ValueError('TEST may not enter v2 development identity')
        model = make_model(cfg).to(context.device)
        old = load_checkpoint(resume) if resume else None
        if old and (old['manifest_sha256'] != digest or old['config'] != cfg or old['world_size'] != context.world_size):
            raise ValueError('Resume identity/config/WORLD_SIZE differs')
        bank_path = run_dir/'train_geometry.pt'
        def prepare():
            if bank_path.exists():
                bank = load_checkpoint(bank_path)
                if bank.get('contract') != geometry_contract(cfg,digest):
                    raise ValueError('TRAIN geometry identity/base/whitening/atlas settings differ')
            else:
                bank = prepare_train_geometry(model,train,cfg,context.device,digest)
                atomic_save(bank,bank_path)
            return bank
        # Payloads are read per rank from shared storage rather than NCCL object-broadcast.
        context.main_call(prepare); context.barrier()
        bank = load_checkpoint(bank_path)
        model.set_band_statistics(bank['statistics'])
        atlas = JetAtlasRegularizer.from_payload(bank['jetatlas']).to(context.device) if bank['jetatlas'] else None
        if old:
            model.load_state_dict(old['model'])
            if atlas is not None: atlas.load_state_dict(old['jetatlas_state'])
        paired, features = load_extractors(project_path(cfg['perceptual']['weights_dir']),context.device)
        objective = V2Objective(model,cfg,atlas,paired if cfg['loss'].get('lpips',0) else None)
        sampler = DistributedSampler(train,context.world_size,context.rank,seed=cfg['train']['seed']) if context.is_distributed else None
        loader = DataLoader(train,batch_size=cfg['train']['batch_size'],sampler=sampler,shuffle=sampler is None,
            num_workers=cfg['train'].get('workers',4),pin_memory=context.device.type=='cuda')
        if len(train) % context.world_size:
            raise ValueError('TRAIN patch budget must divide WORLD_SIZE')
        optimizer = torch.optim.AdamW([p for p in objective.parameters() if p.requires_grad],
            lr=cfg['train']['learning_rate'],weight_decay=cfg['train'].get('weight_decay',1.e-4))
        accumulation = cfg['train'].get('accumulation_steps',1)
        scheduler = warmup_cosine(optimizer,cfg['train'],math.ceil(len(loader)/accumulation))
        ema = ModelEMA(model,cfg['train'].get('ema_decay',.999))
        start, best, history = 0,None,[]
        if old:
            optimizer.load_state_dict(old['optimizer']);scheduler.load_state_dict(old['scheduler'])
            ema.model.load_state_dict(old['ema_model']);ema.updates=old['ema_updates']
            start,best,history = old['epoch'],old['best'],old['history']
            best = tuple(best) if best else None
        if evaluate:
            metrics, records = validate(ema.model,val.all_cases,cfg,context,run_dir,atlas,(paired,features))
            context.main_call(lambda:atomic_save({'metrics':metrics,'cases':records},run_dir/'final_val.json',json_file=True))
            return metrics
        wrapped = DDP(objective,device_ids=[context.local_rank] if context.device.type=='cuda' else None,
                      find_unused_parameters=True) if context.is_distributed else objective
        bf16 = cfg['train'].get('amp_dtype')=='bfloat16' and context.device.type=='cuda'
        for epoch in range(start,cfg['train']['epochs']):
            train.set_epoch(epoch)
            if sampler: sampler.set_epoch(epoch)
            objective.train(); paired.eval()
            optimizer.zero_grad(set_to_none=True)
            sums, batches = {},0
            for index,batch in enumerate(loader):
                with torch.autocast(context.device.type,dtype=torch.bfloat16,enabled=bf16):
                    terms = wrapped(batch['lr'].to(context.device),batch['hr'].to(context.device),batch['id'])
                    # Correct the final partial accumulation window as well.
                    window = min(accumulation,len(loader)-(index//accumulation)*accumulation)
                    loss = terms['loss']/window
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite TRAIN loss')
                loss.backward()
                if (index+1)%accumulation==0 or index+1==len(loader):
                    norm = torch.nn.utils.clip_grad_norm_(objective.parameters(),cfg['train']['clip_grad_norm'],error_if_nonfinite=True)
                    optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);ema.update(model)
                for k,v in terms.items(): sums[k]=sums.get(k,0.)+float(v.detach())
                batches+=1
                if (index+1)%cfg['train'].get('report_every_updates',32)==0 and context.is_main:
                    print(json.dumps({'epoch':epoch+1,'batch':index+1,'loss':float(loss)*window,
                        'gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30 if bf16 else None}),flush=True)
            ema.model.eval()
            metrics, records = validate(ema.model,val.all_cases,cfg,context,run_dir,atlas,(paired,features))
            eligible, score = soft_pareto(metrics,cfg['train']['baseline_psnr'],cfg['train'].get('psnr_tolerance',.5))
            improved = eligible and (best is None or score>best)
            if improved:best=score
            keys=sorted(sums);reduced=context.sum([*[sums[k] for k in keys],batches])
            row = dict(epoch=epoch+1,**metrics,train={k:v/reduced[-1] for k,v in zip(keys,reduced)},
                selection_eligible=eligible,learning_rate=optimizer.param_groups[0]['lr'])
            history.append(row)
            def save():
                checkpoint=dict(format_version=6,algorithm='frequency_split_conditional_bridge',config=cfg,
                    epoch=epoch+1,model=model.state_dict(),ema_model=ema.model.state_dict(),ema_updates=ema.updates,
                    evaluation_weights='ema',jetatlas=bank['jetatlas'],
                    jetatlas_state=None if atlas is None else atlas.state_dict(),band_statistics=bank['statistics'],
                    optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),best=best,history=history,
                    data_identity=identity,manifest_sha256=digest,world_size=context.world_size)
                atomic_save(checkpoint,run_dir/'last.pt')
                if improved:atomic_save(checkpoint,run_dir/'best.pt')
                atomic_save({'history':history,'selection':'SSIM soft Pareto; VAL only'},run_dir/'history.json',json_file=True)
                atomic_save({'metrics':metrics,'cases':records},run_dir/f'val_epoch_{epoch+1:04}.json',json_file=True)
                print(json.dumps(row),flush=True)
            context.main_call(save)
        return history[-1]
    finally:
        if lock:lock.close()
        context.cleanup()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--case-inventory');p.add_argument('--device',default='auto')
    p.add_argument('--resume');p.add_argument('--evaluate',action='store_true')
    a=p.parse_args();run(a.config,a.case_inventory,a.device,a.resume,a.evaluate)


if __name__=='__main__':main()
