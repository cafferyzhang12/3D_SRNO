"""Strict PSNR constrained whole-VAL selection and fresh EMA warm starts."""
import copy
import json
import pytest
import torch
import yaml
from vemgsbm import train
from vemgsbm.runtime import load_checkpoint
from test_x1_bridge import x1_config


def metrics(psnr,ssim):
    return dict(val_pooled_psnr_db=psnr,val_mean_slice_ssim=ssim)


def test_guard_rejects_any_psnr_regression_despite_higher_ssim():
    baseline=metrics(32.5,.905)
    score,eligible,improved=train.guarded_selection(metrics(32.499999,.91),baseline,1-.905)
    assert not eligible and not improved
    assert train.guarded_selection(metrics(32.5,.906),baseline,1-.905)[1:] == (True,True)
    assert train.guarded_selection(metrics(33.,.904),baseline,1-.905)[1:] == (True,False)
    assert train.guarded_selection(baseline,baseline,1-.905)[1:] == (True,False)
    with pytest.raises(ValueError): train.guarded_selection(metrics(float('nan'),.91),baseline,.1)


def test_guarded_warm_start_rechecks_baseline_and_preserves_fallback(tmp_path,monkeypatch):
    cfg=x1_config(tmp_path)
    cfg.update(sigma_0=0.,sigma_bridge=0.,infer_steps=4)
    cfg['path']['sigma']=0.
    cfg['sampling'].update(steps=4,sigma=0.,solver='euler')
    cfg['train'].update(objective='sr_reconstruction',rollout_steps=4)
    path=tmp_path/'train.yaml';path.write_text(yaml.safe_dump(cfg))
    source=train.run_training(path,device_name='cpu')/'best.pt'
    saved=load_checkpoint(source)
    original_validation=train._whole_case_validation
    calls=[]; baseline={}
    def validation(*args,**kwargs):
        row=original_validation(*args,**kwargs)
        epoch=args[5];calls.append(epoch)
        if epoch==0:baseline.update(row)
        else:row.update(val_pooled_psnr_db=baseline['val_pooled_psnr_db']-1,
                        val_mean_slice_ssim=baseline['val_mean_slice_ssim']+.01)
        return row
    monkeypatch.setattr(train,'_whole_case_validation',validation)
    cfg['run_dir']=str(tmp_path/'fine')
    cfg['train'].update(warm_start_checkpoint=str(source),selection_metric='ssim_psnr_guard',ssim_weight=.3)
    path.write_text(yaml.safe_dump(cfg))
    run=train.run_training(path,device_name='cpu')
    best,last=load_checkpoint(run/'best.pt'),load_checkpoint(run/'last.pt')
    assert calls==[0,1]
    assert best['epoch']==0 and best['selection_origin']=='warm_start_fallback'
    assert last['epoch']==1 and last['ema_updates']==2 and last['scheduler']['last_epoch']==2
    assert last['baseline_validation']==best['baseline_validation']==baseline
    assert last['epoch_history'][0]['selection_eligible'] is False
    for key,value in saved['ema_model'].items():
        torch.testing.assert_close(best['ema_model'][key],value,rtol=0,atol=0)
        torch.testing.assert_close(best['model'][key],value,rtol=0,atol=0)
    cfg['train']['bridge_epochs']=2
    path.write_text(yaml.safe_dump(cfg))
    train.run_training(path,device_name='cpu',resume=run/'last.pt')
    assert calls==[0,1,2]
    resumed=load_checkpoint(run/'last.pt')
    assert resumed['ema_updates']==4 and resumed['baseline_validation']==baseline
    assert load_checkpoint(run/'best.pt')['epoch']==0

