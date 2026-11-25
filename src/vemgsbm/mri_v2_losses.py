"""Full-batch MRI reconstruction losses in fixed physical intensity units.

The single-scale SSIM calls the existing evaluation function unchanged. Slices
are checkpointed in chunks to bound its float64 intermediate storage.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .mri_metrics import mean_xy_ssim


DEFAULT_WEIGHTS = dict(charbonnier=1., ssim=.3, msssim=.1,
                       fft_log=.01, fft_complex=.05, local_std=.3, lpips=0.)


def _validate_pair(prediction, reference):
    if (prediction.ndim != 5 or prediction.shape[1] != 1
            or prediction.shape != reference.shape or min(prediction.shape[-2:]) < 3):
        raise ValueError("MRI reconstruction expects aligned B1ZYX images, XY >= 3")


def image_planes(value):
    return value.reshape(-1, 1, *value.shape[-2:])


def local_moments(value, window=7):
    """Valid XY windows, sample variance, no mixing across batch or slices."""
    x = image_planes(value).float() if value.dtype != torch.float64 else image_planes(value)
    window = min(window, *x.shape[-2:])
    window -= 1-window % 2
    if window < 3:
        raise ValueError("Local statistics need XY >= 3")
    mean = F.avg_pool2d(x, window, stride=1)
    variance = (F.avg_pool2d(x.square(), window, stride=1)-mean.square()) * window**2/(window**2-1)
    return mean, variance.clamp_min(0)


def full_batch_ssim(prediction, reference, chunk_slices=4, memory_efficient=True):
    """Exact established mean_xy_ssim, every B*Z slice receives gradients."""
    _validate_pair(prediction, reference)
    if type(chunk_slices) is not int or chunk_slices < 1:
        raise ValueError("SSIM chunk_slices must be a positive integer")
    x, y = image_planes(prediction), image_planes(reference)
    total = prediction.new_zeros((), dtype=torch.float64)
    for start in range(0, len(x), chunk_slices):
        a, b = x[start:start+chunk_slices, None], y[start:start+chunk_slices, None]
        if memory_efficient and torch.is_grad_enabled() and a.requires_grad:
            score = checkpoint(mean_xy_ssim, a, b, use_reentrant=False)
        else:
            score = mean_xy_ssim(a, b)
        total = total + score*len(a)
    return total/len(x)


def _ssim_components(x, y):
    window = min(7, *x.shape[-2:])
    window -= 1-window % 2
    ux = F.avg_pool2d(x, window, 1)
    uy = F.avg_pool2d(y, window, 1)
    factor = window**2/(window**2-1)
    vx = (factor*(F.avg_pool2d(x.square(), window, 1)-ux.square())).clamp_min(0)
    vy = (factor*(F.avg_pool2d(y.square(), window, 1)-uy.square())).clamp_min(0)
    cxy = factor*(F.avg_pool2d(x*y, window, 1)-ux*uy)
    luminance = (2*ux*uy+.01**2)/(ux.square()+uy.square()+.01**2)
    contrast_structure = (2*cxy+.03**2)/(vx+vy+.03**2)
    return luminance.mean((1, 2, 3)), contrast_structure.mean((1, 2, 3))


def multiscale_ssim(prediction, reference, levels=4):
    """Uniform-window MS-SSIM with standard product across XY pyramid levels.

    Sample covariance and unit range match single-scale MRI SSIM; this auxiliary
    loss is explicitly not substituted for the reported single-scale metric.
    Negative structure factors are floored before fractional powers.
    """
    _validate_pair(prediction, reference)
    if type(levels) is not int or not 1 <= levels <= 5:
        raise ValueError("MS-SSIM levels must lie in [1,5]")
    x, y = image_planes(prediction).float(), image_planes(reference).float()
    structures, last_luminance = [], None
    for level in range(levels):
        last_luminance, cs = _ssim_components(x, y)
        structures.append(cs)
        if level+1 == levels or min(x.shape[-2:])//2 < 3:
            break
        x, y = F.avg_pool2d(x, 2, 2), F.avg_pool2d(y, 2, 2)
    weights = x.new_tensor([.0448, .2856, .3001, .2363, .1333])[:len(structures)]
    weights = weights/weights.sum()
    factors = torch.stack(structures)
    factors = torch.cat((factors[:-1], (factors[-1]*last_luminance)[None]), dim=0)
    return torch.exp((weights[:, None]*factors.clamp_min(1.e-6).log()).sum(0)).mean()


class ReconstructionLoss(nn.Module):
    """Configurable physical-space loss; no HR-dependent rescaling or sampling.

    ``config`` is a weight mapping, or contains a ``weights`` mapping. Optional
    LPIPS models must be supplied explicitly and use frozen pretrained weights.
    """
    def __init__(self, config=None, lpips_model=None):
        super().__init__()
        config = dict(config or {})
        supplied = config.get("weights", config)
        settings = {key: value for key, value in supplied.items() if key in DEFAULT_WEIGHTS}
        self.weights = {**DEFAULT_WEIGHTS, **settings}
        if any(not math.isfinite(float(v)) or float(v) < 0 for v in self.weights.values()):
            raise ValueError("Reconstruction weights must be finite and nonnegative")
        self.epsilon = float(config.get("charbonnier_epsilon", 1.e-3))
        self.chunk_slices = int(config.get("ssim_chunk_slices", 4))
        if not math.isfinite(self.epsilon) or self.epsilon <= 0 or self.chunk_slices < 1:
            raise ValueError("Invalid loss epsilon or SSIM chunk size")
        self.lpips_model = lpips_model
        if self.weights["lpips"] and lpips_model is None:
            raise ValueError("Nonzero LPIPS loss requires an explicitly loaded pretrained model")
        if lpips_model is not None:
            lpips_model.eval().requires_grad_(False)

    def forward(self, prediction, reference):
        _validate_pair(prediction, reference)
        # Disable AMP for statistics/FFT: bf16 cancellation erases small MRI
        # variances, and CUDA FFT does not support arbitrary half-size grids.
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            p, r = prediction.float(), reference.float()
            zero = p.sum()*0
            terms = {}
            terms["charbonnier"] = ((p-r).square()+self.epsilon**2).sqrt().mean()-self.epsilon
            terms["ssim"] = 1-full_batch_ssim(p, r, self.chunk_slices) if self.weights["ssim"] else zero
            terms["msssim"] = 1-multiscale_ssim(p, r) if self.weights["msssim"] else zero
            if self.weights["fft_log"] or self.weights["fft_complex"]:
                a, b = torch.fft.rfft2(p, norm="ortho"), torch.fft.rfft2(r, norm="ortho")
                terms["fft_log"] = (torch.log1p(a.abs())-torch.log1p(b.abs())).abs().mean()
                terms["fft_complex"] = (a-b).abs().mean()
            else:
                terms.update(fft_log=zero, fft_complex=zero)
            if self.weights["local_std"]:
                _, vp = local_moments(p)
                _, vr = local_moments(r)
                # Epsilon makes flat-patch derivatives finite, including black MRI background.
                terms["local_std"] = ((vp+1.e-8).sqrt()-(vr+1.e-8).sqrt()).abs().mean()
            else:
                terms["local_std"] = zero
            if self.weights["lpips"]:
                xp, xr = image_planes(p).repeat(1, 3, 1, 1), image_planes(r).repeat(1, 3, 1, 1)
                terms["lpips"] = self.lpips_model(2*xp-1, 2*xr-1).mean()
            else:
                terms["lpips"] = zero
            terms["loss"] = sum(self.weights[key]*value for key, value in terms.items())
            return terms


def gradient_diagnostics(terms, parameters, weights=None):
    """Real per-component backprop norms and relative weighted norm shares."""
    parameters = tuple(p for p in parameters if p.requires_grad)
    weights = DEFAULT_WEIGHTS if weights is None else weights
    norms = {}
    for name, term in terms.items():
        if name == "loss" or not torch.is_tensor(term) or not term.requires_grad:
            continue
        gradients = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
        norm = sum(float(g.detach().double().square().sum()) for g in gradients if g is not None)**.5
        norms[name] = dict(raw_norm=norm, weighted_norm=abs(float(weights.get(name, 1.)))*norm)
    total = sum(row["weighted_norm"] for row in norms.values())
    return {name: {**row, "norm_share": row["weighted_norm"]/max(total, 1.e-30)} for name, row in norms.items()}


def reconstruction_terms(prediction, reference, config=None, lpips_model=None):
    """Functional adapter for objectives; reuses the supplied frozen LPIPS model."""
    return ReconstructionLoss(config, lpips_model)(prediction, reference)
