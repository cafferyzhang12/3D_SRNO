"""Opt-in MRI v2 contracts; historical metrics and tiling remain authoritative."""
import copy
import math

import torch
from torch import nn


def is_v2(cfg):
    return cfg.get('model', {}).get('architecture') == 'HybridWaveletNAFNet25D'


def validate_config(cfg):
    if cfg['mode'] != 'haar_bridge' or cfg.get('predict_target') != 'x1':
        raise ValueError('Hybrid v2 requires the native-LR Haar endpoint API')
    if set(cfg['data']) & {'test'}:
        raise ValueError('v2 development configurations must contain TRAIN/VAL only')
    if cfg['data']['scale'] != 4 or cfg['data']['degradation'] != 'paired_xy':
        raise ValueError('v2 requires paired XY x4 data')
    z, h, w = cfg['data']['patch_size']
    if z < 1 or min(h,w) < 12 or h % 4 or w % 4:
        raise ValueError('Expected positive depth and XY patches divisible by four')
    train = cfg['train']
    if min(train['epochs'], train['batch_size'], train['samples_per_epoch']) < 1:
        raise ValueError('Positive training budget required')
    if train['selection_metric'] != 'ssim_soft_pareto':
        raise ValueError('v2 uses whole-VAL SSIM soft Pareto selection')
    if cfg['sampling']['solver'] not in {'euler','heun'}:
        raise ValueError('Expected Euler or Heun solver')
    if cfg['sampling']['sampler_type'] not in {'ode','sde'}:
        raise ValueError('Expected ODE or SDE')
    if cfg['sampling']['steps'] < 1 or not 0 <= cfg['sampling'].get('temperature',.1) <= .3:
        raise ValueError('Invalid inference steps or SDE temperature')
    if cfg.get('atlas_weight',0) and not cfg.get('jetatlas'):
        raise ValueError('Atlas weight requires explicit TRAIN atlas settings')
    if cfg.get('online_atlas',False) and not cfg.get('atlas_weight',0):
        raise ValueError('Online geometry requires a trained atlas and router')
    if cfg.get('sigma_0',0) != 0:
        raise ValueError('v2 starts LL deterministically and HF at zero by default')
    return cfg


def make_model(cfg):
    from .hybrid import HybridWaveletNAFNet25D
    model = dict(cfg['model'])
    model.pop('architecture')
    result = HybridWaveletNAFNet25D(**model, sigma_bridge=cfg.get('sigma_bridge',.08),
        frequency_split=cfg.get('frequency_split',True),
        use_band_whitening=cfg.get('use_band_whitening',True),
        sampler_type=cfg['sampling']['sampler_type'],solver=cfg['sampling']['solver'],
        temperature=cfg['sampling'].get('temperature',.1),
        base_interpolation=cfg.get('base_interpolation','bicubic'))
    result.online_geometry_start = cfg.get('sampling',{}).get('online_geometry_start',1.)
    result.eb_tau = cfg.get('sampling',{}).get('eb_tau',.1)
    return result


def soft_pareto(metrics, baseline_psnr, tolerance=.5):
    """SSIM leads; LPIPS/FID/PSNR break ties, with a broad divergence floor."""
    names = ('psnr','ssim','lpips','fid')
    if any(not math.isfinite(metrics[k]) for k in names):
        raise ValueError('Finite four-metric VAL results required')
    eligible = metrics['psnr'] >= baseline_psnr-tolerance
    return eligible, (metrics['ssim'], -metrics['lpips'], -metrics['fid'], metrics['psnr'])


class V2Objective(nn.Module):
    def __init__(self, model, cfg, atlas=None, lpips_model=None):
        super().__init__()
        self.model, self.cfg, self.atlas = model, cfg, atlas
        self.lpips_model = lpips_model

    def forward(self, lr, hr, case_ids=None):
        from .frequency_bridge import sample_frequency_bridge, integrate_frequency_bridge
        from .mri_v2_losses import reconstruction_terms
        m, cfg = self.model, self.cfg
        base = m.prepare_base(lr)
        condition = m.lr_encoder(lr)
        target = m.normalize_haar(m.codec.encode(hr-base)).detach()
        zero = torch.zeros_like(target)
        terms = {}

        def supervise(clean, reference, image_base, cond, prefix):
            image = m.decode_prediction(clean, image_base, cond)
            values = reconstruction_terms(image, reference, cfg.get('loss'), lpips_model=self.lpips_model)
            terms.update({prefix+k:v for k,v in values.items() if k != 'loss'})
            return values['loss']

        # Every item receives structural/variance supervision from deployment X0.
        start, _ = m(zero, condition, 0.)
        loss = supervise(start, hr, base, condition, 'start_')
        if cfg.get('frequency_split',False):
            state, t, noise = sample_frequency_bridge(target, sigma_max=m.sigma_bridge)
            clean, epsilon = m(state, condition, t)
            bridge_rec = supervise(clean, hr, base, condition, 'bridge_')
            tau = t.reshape(-1,1,1,1,1)
            # Noise is unidentifiable at either boundary; never supervise it there.
            interior = ((tau > 0) & (tau < 1)).to(clean)
            noise_loss = ((epsilon[:,1:]-noise[:,1:]).square()*interior).sum() / (
                interior.sum()*epsilon[:,1:].numel()/len(epsilon)).clamp_min(1)
            loss = loss + cfg['train'].get('bridge_weight',1.)*bridge_rec + cfg['train'].get('noise_weight',.05)*noise_loss
            terms['noise_loss'] = noise_loss
            n = max(1, round(len(lr)*cfg['train'].get('rollout_fraction',.25)))
            sub = [c[:n] for c in condition]
            # Random short grids, plus continuous teacher states, train beyond a fixed N=4.
            steps = int(torch.randint(2,5,()).item())
            with torch.no_grad():
                prefix, last_t = integrate_frequency_bridge(
                    lambda x,t:m(x,[c.detach() for c in sub],t), zero[:n], steps,
                    sigma_max=m.sigma_bridge, solver='heun',return_prefix=True,
                    sampler_type='sde' if self.atlas is not None else 'ode',
                    temperature=cfg['sampling'].get('temperature',.1),
                    curvature=(lambda x,t:self.atlas.curvature_drift(x,lr[:n])) if self.atlas is not None else None)
            rollout, _ = m(prefix, sub, last_t)
            loss = loss + cfg['train'].get('rollout_weight',.25)*supervise(
                rollout, hr[:n], base[:n], sub, 'rollout_')
        else:
            state, clean = start, start
        if cfg.get('atlas_weight',0):
            if self.atlas is None or case_ids is None:
                raise ValueError('TRAIN geometry requires cross-fitted atlas and case IDs')
            # Evaluate the potential on a generated intermediate state so it trains
            # the main network, rather than on a fixed teacher bridge sample.
            generated = (.5*clean if not cfg.get('frequency_split',False) else
                         tau*clean+(state-tau*target))
            geometry = self.atlas(clean, target, case_ids, lr_features=lr, state=generated, reference=hr)
            terms.update({k:v for k,v in geometry.items() if k != 'loss'})
            loss = loss + cfg['atlas_weight']*geometry['loss']
        terms['loss'] = loss
        return terms


def inference_predictor(model, cfg, atlas=None, steps=None):
    def predict(lr, generator):
        return model.restore(lr, model.prepare_base(lr), steps or cfg['sampling']['steps'],
            generator=generator, atlas=atlas if cfg.get('online_atlas',False) else None)
    return predict
