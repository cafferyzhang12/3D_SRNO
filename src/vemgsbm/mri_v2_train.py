"""Four-metric, TRAIN/VAL-only training for the opt-in MRI v2 ablations."""
import argparse
from contextlib import ExitStack, nullcontext
import copy
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .datasets import training_data
from .distributed import initialize_distributed
from .mri_v2 import V2Objective, make_model, soft_pareto, inference_predictor
from .runtime import atomic_save, fingerprint, read_config, project_path, load_checkpoint, setup
from .training_state import ModelEMA, warmup_cosine


class EpochPatchSampler(DistributedSampler):
    """Send epoch identity to persistent workers, and resume without decoding skipped batches."""

    offset = 0

    def set_progress(self, epoch, next_batch, batch_size):
        self.set_epoch(epoch)
        self.offset = min(self.num_samples, next_batch * batch_size)

    def __iter__(self):
        indices = islice(super().__iter__(), self.offset, None)
        return ((self.epoch, index) for index in indices)

    def __len__(self):
        return self.num_samples - self.offset


def _capture_rng(device):
    algorithm, keys, position, has_gauss, cached_gauss = np.random.get_state()
    return dict(python=random.getstate(), numpy=(algorithm, keys.tolist(), position, has_gauss, cached_gauss),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def _restore_rng(state, device):
    random.setstate(state["python"])
    algorithm, keys, position, has_gauss, cached_gauss = state["numpy"]
    np.random.set_state((algorithm, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gauss))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def _should_validate(epoch, settings):
    completed = epoch + 1
    return (completed == settings["epochs"]
            or completed % settings.get("val_every_epochs", 1) == 0
            or completed > settings["epochs"] - settings.get("val_final_epochs", 10))


def _optimizer(parameters, settings, device):
    kwargs = dict(lr=settings["learning_rate"], weight_decay=settings.get("weight_decay", 1.e-4))
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(parameters, **kwargs, fused=True)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(parameters, **kwargs)


def _cuda_convolution_layout(model):
    torch.nn.utils.convert_conv2d_weight_memory_format(model, torch.channels_last)
    torch.nn.utils.convert_conv3d_weight_memory_format(model, torch.channels_last_3d)
    # Bound cuDNN v8 autotuning probes on the full HR patch shape.
    torch.backends.cudnn.benchmark_limit = 4


def _compile_backbone(model, settings, context, stack):
    # DDP owns the eager objective and original parameters. Compile stable tensor
    # subgraphs without specializing on case IDs, fold sizes, or logging.
    if context.device.type != "cuda" or not settings.get("compile", True):
        return
    try:
        from torch._dynamo import config as dynamo_config
        from torch._inductor import config as inductor_config

        stack.enter_context(dynamo_config.patch(suppress_errors=True))
        threads = int(os.environ.get("TORCHINDUCTOR_COMPILE_THREADS", "1"))
        stack.enter_context(inductor_config.patch(compile_threads=threads))
        modules = (model, model.lr_encoder, model.refinement)
        compiled = [torch.compile(module.forward) for module in modules]
    except (AttributeError, ImportError, RuntimeError, ValueError) as exc:
        if context.is_main:
            print(json.dumps({"compile_fallback": f"{type(exc).__name__}: {exc}"}), flush=True)
        return
    for module, forward in zip(modules, compiled, strict=True):
        module.forward = forward


def geometry_contract(cfg, digest):
    return dict(manifest=digest, base=cfg['base_interpolation'],
        whitening=cfg.get('use_band_whitening',True),patch=cfg['data']['patch_size'],
        atlas=cfg.get('jetatlas'),atlas_weight=cfg.get('atlas_weight',0),seed=cfg['train']['seed'])


def prepare_train_geometry(model, dataset, cfg, device, digest):
    """Deterministic case-balanced TRAIN bank. No VAL tensors or cached old bases."""
    from .jetatlas import hf_patches, lr_router_rows, JetAtlasRegularizer
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
                low_rows = lr_router_rows(lr, int(settings.get('lr_dim',9)), grid_shape=coeff.shape[2:]).cpu()
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
    from .mri_perceptual import FeatureMoments, consume_case, frechet_distance, load_extractors
    from .mri_v2_diagnostics import textured_region_ssim
    from .mri_v2_losses import local_moments
    paired, features = extractors or load_extractors(project_path(cfg['perceptual']['weights_dir']),context.device)
    predicted, reference = FeatureMoments(), FeatureMoments()
    totals = np.zeros(9,dtype=np.float64)
    records = []
    error = None
    try:
        for case in cases[context.rank::context.world_size]:
            with tempfile.TemporaryDirectory(dir=directory) as temp:
                folder = Path(temp)/'case'
                _, row = _infer_case(case,folder,model,None,cfg,context.device,
                    cfg['train']['validation_seed'],steps or cfg['sampling']['steps'],1,
                    volume_predictor=inference_predictor(model,cfg,atlas,steps))
                diagnostics = np.zeros(4, dtype=np.float64)

                def observe(prediction, truth):
                    p = torch.from_numpy(np.array(prediction, copy=True))[None, None].to(context.device)
                    y = torch.from_numpy(truth)[None, None].to(context.device)
                    _, vp = local_moments(p)
                    _, vy = local_moments(y)
                    diagnostics[0] += float(vp.double().sum())
                    diagnostics[1] += float(vy.double().sum())
                    for z in range(len(truth)):
                        value = textured_region_ssim(p[:, :, z:z+1], y[:, :, z:z+1])
                        if value is not None and math.isfinite(value):
                            diagnostics[2] += value
                            diagnostics[3] += 1

                volume = np.load(folder/'mean.npy', mmap_mode='r', allow_pickle=False)
                try:
                    perceptual = consume_case(volume,case,paired,features,predicted,reference,
                        context.device,cfg['perceptual'].get('batch_size',8),observe=observe)
                finally:
                    volume._mmap.close()
                totals += [row['squared_error_sum'],row['pixel_count'],
                    row['mean_slice_ssim']*row['ssim_slice_count'],row['ssim_slice_count'],
                    perceptual['lpips_sum'],*diagnostics]
                records.append({'case_id':case.id,**{k:v for k,v in row.items() if k!='slices'},**perceptual})
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
    settings = cfg['train']
    if evaluate and resume is None:
        raise ValueError('--evaluate requires --resume with a frozen checkpoint')
    for key, default, minimum in (
            ('workers', 6, 0), ('val_every_epochs', 1, 1), ('val_final_epochs', 10, 0),
            ('save_every_updates', 0, 0), ('accumulation_steps', 1, 1), ('report_every_updates', 32, 1)):
        value = settings.get(key, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    context = initialize_distributed(device)
    run_dir = project_path(cfg['run_dir'])
    run_dir.mkdir(parents=True,exist_ok=True)
    lock = None
    stack = ExitStack()
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
        setup(settings['seed'],str(context.device),settings.get('threads',4))
        train, val, digest, identity = training_data(cfg,inventory,return_base=False,fast_trusted=True)
        if any(row['split'] == 'test' for row in identity):
            raise ValueError('TEST may not enter v2 development identity')
        model = make_model(cfg).to(context.device)
        cuda = context.device.type == 'cuda'
        if cuda:
            _cuda_convolution_layout(model)
        old = load_checkpoint(resume) if resume else None
        if old and (old['manifest_sha256'] != digest or old['config'] != cfg or old['world_size'] != context.world_size):
            raise ValueError('Resume identity/config/WORLD_SIZE differs')
        if evaluate:
            model.set_band_statistics(old['band_statistics'])
            model.load_state_dict(old['ema_model'])
            atlas = JetAtlasRegularizer.from_payload(old['jetatlas']).to(context.device) if old['jetatlas'] else None
            if atlas is not None:
                atlas.load_state_dict(old['jetatlas_state'])
                atlas.eval()
            metrics, records = validate(model.eval(),val.all_cases,cfg,context,run_dir,atlas)
            context.main_call(lambda:atomic_save({'metrics':metrics,'cases':records,'checkpoint_sha256':fingerprint(resume)},
                run_dir/'final_val.json',json_file=True))
            return metrics
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
        if len(train) % context.world_size:
            raise ValueError('TRAIN patch budget must divide WORLD_SIZE')
        sampler = EpochPatchSampler(train,context.world_size,context.rank,seed=settings['seed'])
        workers, batch_size = settings.get('workers', 6), settings['batch_size']
        loader = DataLoader(train,batch_size=batch_size,sampler=sampler,num_workers=workers,
            pin_memory=True,persistent_workers=workers > 0,
            generator=torch.Generator().manual_seed(settings['seed']+context.rank),
            **({'prefetch_factor':4} if workers else {}))
        epoch_batches = len(loader)
        parameters = [p for p in objective.parameters() if p.requires_grad]
        optimizer = _optimizer(parameters,settings,context.device)
        accumulation = settings.get('accumulation_steps',1)
        scheduler = warmup_cosine(optimizer,settings,math.ceil(epoch_batches/accumulation))
        ema = ModelEMA(model,settings.get('ema_decay',.999))
        start, best, history = 0, None, []
        next_batch, progress = 0, None
        if old:
            optimizer.load_state_dict(old['optimizer']);scheduler.load_state_dict(old['scheduler'])
            ema.model.load_state_dict(old['ema_model']);ema.updates=old['ema_updates']
            start,best,history = old['epoch'],old['best'],old['history']
            best = tuple(best) if best else None
            next_batch = old.get('next_batch', 0)
            rank_progress = old.get('rank_progress')
            if not 0 <= next_batch <= epoch_batches or (next_batch != epoch_batches and next_batch % accumulation):
                raise ValueError('Checkpoint batch cursor must land on an optimizer update boundary')
            if rank_progress:
                progress = rank_progress[context.rank]
            if next_batch and (progress is None or progress['batches'] != next_batch):
                raise ValueError('Incomplete rank progress for a mid-epoch checkpoint')
        _compile_backbone(model,settings,context,stack)
        wrapped = DDP(objective,device_ids=[context.local_rank] if cuda else None,
                      broadcast_buffers=False,gradient_as_bucket_view=True,
                      find_unused_parameters=objective.has_unused_parameters) if context.is_distributed else objective
        bf16 = settings.get('amp_dtype')=='bfloat16' and cuda
        if progress is not None:
            _restore_rng(progress['rng'],context.device)

        def save_checkpoint(epoch, cursor, names, sums, batches, improved=False):
            local = dict(names=names, sums=None if sums is None else sums.detach().cpu(),
                         batches=batches, rng=_capture_rng(context.device))
            all_ranks = [local]
            if context.is_distributed:
                all_ranks = [None] * context.world_size
                torch.distributed.all_gather_object(all_ranks,local)

            def save():
                checkpoint=dict(format_version=6,algorithm='frequency_split_conditional_bridge',config=cfg,
                    epoch=epoch,next_batch=cursor,rank_progress=all_ranks,
                    model=model.state_dict(),ema_model=ema.model.state_dict(),ema_updates=ema.updates,
                    evaluation_weights='ema',jetatlas=bank['jetatlas'],
                    jetatlas_state=None if atlas is None else atlas.state_dict(),band_statistics=bank['statistics'],
                    optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),best=best,history=history,
                    data_identity=identity,manifest_sha256=digest,world_size=context.world_size)
                if improved:
                    atomic_save(checkpoint,run_dir/'best.pt')
                atomic_save(checkpoint,run_dir/'last.pt')
            context.main_call(save)

        for epoch in range(start,settings['epochs']):
            sampler.set_progress(epoch,next_batch,batch_size)
            objective.train()
            paired.eval()
            optimizer.zero_grad(set_to_none=True)
            names, sums, batches = [], None, 0
            if next_batch:
                names, sums, batches = progress['names'],progress['sums'].to(context.device),progress['batches']
            report_start, report_batches = time.perf_counter(), 0
            for index,batch in enumerate(loader,start=next_batch):
                final = (index+1) % accumulation == 0 or index+1 == epoch_batches
                sync = wrapped.no_sync() if context.is_distributed and not final else nullcontext()
                with sync:
                    with torch.autocast(context.device.type,dtype=torch.bfloat16,enabled=bf16):
                        terms = wrapped(batch['lr'].to(context.device,non_blocking=True),
                            batch['hr'].to(context.device,non_blocking=True),batch['id'])
                        # Correct the final partial accumulation window as well.
                        window = min(accumulation,epoch_batches-(index//accumulation)*accumulation)
                        loss = terms['loss']/window
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite TRAIN loss')
                    loss.backward()
                if final:
                    torch.nn.utils.clip_grad_norm_(parameters,settings['clip_grad_norm'],error_if_nonfinite=True)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(model)
                measured = torch.stack([value.detach().reshape(()) for value in terms.values()]).double()
                if sums is None:
                    names, sums = list(terms), measured
                else:
                    sums.add_(measured)
                batches += 1
                report_batches += 1
                if (index==next_batch or (index+1)%settings.get('report_every_updates',32)==0) and context.is_main:
                    loss_value = float(terms['loss'].detach())
                    report_end = time.perf_counter()
                    print(json.dumps({'epoch':epoch+1,'batch':index+1,'update':ema.updates,
                        'loss':loss_value,'seconds_per_batch':(report_end-report_start)/report_batches,
                        'gpu_peak_gib':torch.cuda.max_memory_allocated(context.device)/2**30 if cuda else None}),flush=True)
                    report_start, report_batches = report_end, 0
                interval = settings.get('save_every_updates',0)
                if final and interval and ema.updates % interval == 0:
                    save_checkpoint(epoch,index+1,names,sums,batches)
            next_batch, progress = 0, None
            metrics, records, eligible, improved = {}, None, None, False
            if _should_validate(epoch,settings):
                metrics, records = validate(ema.model,val.all_cases,cfg,context,run_dir,atlas,(paired,features))
                eligible, score = soft_pareto(metrics,settings['baseline_psnr'],settings.get('psnr_tolerance',.5))
                improved = eligible and (best is None or score>best)
                if improved:
                    best = score
            reduced = context.sum([*sums.tolist(),batches])
            row = dict(epoch=epoch+1,**metrics,train={k:v/reduced[-1] for k,v in zip(names,reduced)},
                       selection_eligible=eligible,learning_rate=optimizer.param_groups[0]['lr'])
            history.append(row)
            save_checkpoint(epoch+1,0,[],None,0,improved)

            def save_history():
                atomic_save({'history':history,'selection':'SSIM soft Pareto; VAL only'},run_dir/'history.json',json_file=True)
                if records is not None:
                    atomic_save({'metrics':metrics,'cases':records},run_dir/f'val_epoch_{epoch+1:04}.json',json_file=True)
                print(json.dumps(row),flush=True)
            context.main_call(save_history)
        return history[-1]
    finally:
        stack.close()
        if lock:lock.close()
        context.cleanup()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--case-inventory');p.add_argument('--device',default='auto')
    p.add_argument('--resume');p.add_argument('--evaluate',action='store_true')
    a=p.parse_args();run(a.config,a.case_inventory,a.device,a.resume,a.evaluate)


if __name__=='__main__':main()
