"""Opt-in MRI v2 contracts; historical metrics and tiling remain authoritative."""
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
    sampling = cfg['sampling']
    if sampling.get('solver', 'heun') not in {'euler','heun'}:
        raise ValueError('Expected Euler or Heun solver')
    if sampling.get('sampler_type', 'ode') not in {'ode','sde'}:
        raise ValueError('Expected ODE or SDE')
    if type(sampling['steps']) is not int or sampling['steps'] < 1 or not 0 <= sampling.get('temperature', 0.) <= .3:
        raise ValueError('Invalid inference steps or SDE temperature')
    for key in ('bridge_weight', 'noise_weight', 'rollout_weight'):
        weight = train.get(key, 0.)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f'Invalid {key}')
    if train.get('rollout_weight', 0.):
        steps = train.get('rollout_steps', 3)
        if not 0 < train.get('rollout_fraction', .25) <= 1 or type(steps) is not int or steps < 2:
            raise ValueError('A rollout requires a fraction in (0, 1] and at least two steps')
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
    sampling = cfg['sampling']
    result = HybridWaveletNAFNet25D(**model, sigma_bridge=cfg.get('sigma_bridge', 0.),
        frequency_split=cfg.get('frequency_split',True),
        use_band_whitening=cfg.get('use_band_whitening',True),
        sampler_type=sampling.get('sampler_type', 'ode'), solver=sampling.get('solver', 'heun'),
        temperature=sampling.get('temperature', 0.),
        base_interpolation=cfg.get('base_interpolation','bicubic'))
    result.online_geometry_start = sampling.get('online_geometry_start', 1.)
    result.eb_tau = sampling.get('eb_tau', .1)
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
        from .mri_v2_losses import ReconstructionLoss

        self.model, self.cfg, self.atlas = model, cfg, atlas
        self.reconstruction = ReconstructionLoss(cfg.get('loss'), lpips_model)
        train = cfg['train']
        self.bridge_weight = train.get('bridge_weight', 0.) if model.frequency_split else 0.
        self.noise_weight = train.get('noise_weight', 0.) if model.sigma_bridge else 0.
        self.rollout_weight = train.get('rollout_weight', 0.) if model.frequency_split else 0.
        self.atlas_weight = cfg.get('atlas_weight', 0.)
        self.model.epsilon.requires_grad_(bool(self.noise_weight))
        self.has_unused_parameters = atlas is not None and not self.atlas_weight and any(
            parameter.requires_grad for parameter in atlas.parameters())

    def _bridge_supervision(self, clean, target_coefficients, prefix, terms):
        model = self.model
        epsilon = self.reconstruction.epsilon
        with torch.autocast(clean.device.type, enabled=False):
            bands = model.denormalize_haar(clean.float())-target_coefficients
            haar = bands.abs().mean()
            pixels = model.codec.decode(bands)
            pixel = (pixels.square()+epsilon**2).sqrt().mean()-epsilon
        terms[f'{prefix}_haar'] = haar
        terms[f'{prefix}_pixel'] = pixel
        return haar+pixel

    def forward(self, lr, hr, case_ids=None):
        from .frequency_bridge import sample_frequency_bridge, integrate_frequency_bridge

        m = self.model
        base = m.prepare_base(lr)
        condition = m.lr_encoder(lr)
        with torch.no_grad(), torch.autocast(lr.device.type, enabled=False):
            target_coefficients = m.codec.encode(hr-base)
            target = m.normalize_haar(target_coefficients)
        zero = torch.zeros_like(target)

        # Every item receives structural/variance supervision from deployment X0.
        start, _ = m(zero, condition, 0., predict_noise=False, trusted_time=True)
        values = self.reconstruction(m.decode_prediction(start, base, condition), hr)
        loss = values['loss']
        terms = {f'start_{key}': value for key, value in values.items() if key != 'loss'}
        clean = start
        generated = .5*start if self.atlas_weight else None
        if self.bridge_weight or self.noise_weight:
            state, t, noise = sample_frequency_bridge(target, sigma_max=m.sigma_bridge)
            clean, epsilon = m(state, condition, t, predict_noise=bool(self.noise_weight), trusted_time=True)
            if self.bridge_weight:
                loss = loss + self.bridge_weight*self._bridge_supervision(clean, target_coefficients, 'bridge', terms)
            tau = t[:, None, None, None, None]
            if self.noise_weight:
                # Epsilon is defined only in the interior of a noisy bridge.
                interior = (tau > 0) & (tau < 1)
                hf_error = (epsilon[:, 1:].float()-noise[:, 1:]).square()*interior
                noise_loss = hf_error.sum()/(interior.sum()*hf_error[0].numel()).clamp_min(1)
                terms['noise_loss'] = noise_loss
                loss = loss + self.noise_weight*noise_loss
            if self.atlas_weight:
                # The generated state carries endpoint gradients into the potential.
                generated = tau*clean
                if m.sigma_bridge:
                    generated = generated+(state-tau*target)

        if self.rollout_weight:
            train = self.cfg['train']
            n = max(1, round(len(lr)*train.get('rollout_fraction', .25)))
            sub = [c[:n] for c in condition]
            with torch.no_grad():
                prefix, last_t = integrate_frequency_bridge(
                    lambda x, t: m(x, sub, t, predict_noise=False, trusted_time=True), zero[:n],
                    train.get('rollout_steps', 3), solver=self.model.solver, return_prefix=True)
            rollout, _ = m(prefix, sub, last_t, predict_noise=False, trusted_time=True)
            loss = loss + self.rollout_weight*self._bridge_supervision(rollout, target_coefficients[:n], 'rollout', terms)

        if self.atlas_weight:
            if self.atlas is None or case_ids is None:
                raise ValueError('TRAIN geometry requires cross-fitted atlas and case IDs')
            geometry = self.atlas(clean, target, case_ids, lr_features=lr, state=generated, reference=hr)
            terms.update({k:v for k,v in geometry.items() if k != 'loss'})
            loss = loss + self.atlas_weight*geometry['loss']
        terms['loss'] = loss
        return terms


def inference_predictor(model, cfg, atlas=None, steps=None):
    def predict(lr, generator):
        return model.restore(lr, steps=steps or cfg['sampling']['steps'],
            generator=generator, atlas=atlas if cfg.get('online_atlas',False) else None)
    return predict
