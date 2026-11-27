"""Paired whole-VAL online/offline atlas evaluation of one frozen exp3 weight."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from .distributed import initialize_distributed
from .jetatlas import JetAtlasRegularizer
from .mri_inventory import load_inventory
from .mri_perceptual import load_extractors
from .mri_v2_train import validate
from .runtime import atomic_save, build_models, fingerprint, load_checkpoint, project_path, setup


def comparison_configs(checkpoint):
    cfg=checkpoint['config']
    if (checkpoint.get('format_version')!=6 or cfg.get('online_atlas',False)
        or not cfg.get('frequency_split') or not cfg.get('atlas_weight',0)
        or checkpoint.get('jetatlas') is None or checkpoint.get('jetatlas_state') is None):
        raise ValueError('Paired evaluation requires an offline-selected exp3 checkpoint with trained router/atlas')
    if 'test' in cfg['data']:
        raise ValueError('Paired development evaluation accepts TRAIN/VAL contracts only')
    offline=copy.deepcopy(cfg)
    online=copy.deepcopy(cfg)
    online['online_atlas']=True
    return {'offline':offline,'online':online}


def verify_identity(checkpoint,identity):
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if any(row['split']=='test' for row in identity):
        raise ValueError('TEST is forbidden in paired development evaluation')
    if digest!=checkpoint['manifest_sha256'] or identity!=checkpoint['data_identity']:
        raise ValueError('Paired VAL inventory differs from training identity')


def run(checkpoint,inventory,output,device='auto'):
    context=initialize_distributed(device)
    output=project_path(output)
    source=project_path(checkpoint)
    try:
        source_hash=context.main_call(lambda:fingerprint(source),broadcast_result=True)
        def freeze():
            output.mkdir(parents=True,exist_ok=False)
            frozen=output/'checkpoint.pt'
            shutil.copyfile(source,frozen)
            if fingerprint(frozen)!=source_hash:
                raise RuntimeError('Checkpoint changed while freezing; no comparison performed')
            return str(frozen)
        frozen=Path(context.main_call(freeze,broadcast_result=True))
        ck=load_checkpoint(frozen)
        variants=comparison_configs(ck)
        cfg=variants['offline']
        cases,identity=load_inventory(inventory,cfg['data'],('val',))
        verify_identity(ck,identity)
        model,_=build_models(ck,context.device)
        atlas=JetAtlasRegularizer.from_payload(ck['jetatlas']).to(context.device)
        atlas.load_state_dict(ck['jetatlas_state'])
        atlas.eval().requires_grad_(False)
        extractors=load_extractors(project_path(cfg['perceptual']['weights_dir']),context.device)
        manifest=dict(checkpoint_sha256=source_hash,source_checkpoint=str(source),
            frozen_checkpoint=str(frozen),epoch=ck['epoch'],weights=ck.get('evaluation_weights','model'),
            selection='offline_VAL_best' if source.name=='best.pt' else 'explicit_checkpoint_or_last_fallback',
            online_metrics_reselect_weights=False,
            inventory_sha256=fingerprint(project_path(inventory)),test_used=False,
            sampling=cfg['sampling'],validation_seed=cfg['train']['validation_seed'],world_size=context.world_size)
        context.main_call(lambda:atomic_save(manifest,output/'evaluation_identity.json',json_file=True))
        results={}
        for name,candidate in variants.items():
            setup(candidate['train']['seed'],str(context.device),candidate['train'].get('threads',4))
            folder=output/name
            context.main_call(lambda:folder.mkdir())
            metrics,records=validate(model,cases['val'],candidate,context,folder,atlas,extractors)
            results[name]=metrics
            context.main_call(lambda:atomic_save(dict(checkpoint_sha256=source_hash,
                online_atlas=candidate['online_atlas'],metrics=metrics,cases=records),folder/'val.json',json_file=True))
            if context.is_main:print(json.dumps(dict(variant=name,checkpoint_sha256=source_hash,**metrics)),flush=True)
            if context.main_call(lambda:fingerprint(frozen),broadcast_result=True)!=source_hash:
                raise RuntimeError('Frozen weights changed between evaluations')
        delta={key:results['online'][key]-results['offline'][key] for key in
            ('psnr','ssim','lpips','fid','local_variance_ratio','textured_region_ssim')}
        report=dict(identity=manifest,results=results,online_minus_offline=delta)
        context.main_call(lambda:atomic_save(report,output/'comparison.json',json_file=True))
        return report
    finally:
        context.cleanup()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--case-inventory',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='auto')
    a=p.parse_args();run(a.checkpoint,a.case_inventory,a.output,a.device)
