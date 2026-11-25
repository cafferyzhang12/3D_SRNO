"""Additional MRI diagnostics; established PSNR/SSIM/perceptual protocols stay intact."""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F

from .mri_v2_losses import image_planes, local_moments


def upsample_xy(value, scale=4, mode="bicubic"):
    """Differentiable XY interpolation with align_corners=False coordinates.

    Lanczos-3 uses separable sinc kernels and replicated edge samples. No Z
    interpolation is performed. Boundary behavior is fixed across train/audit.
    """
    if value.ndim != 5 or type(scale) is not int or scale < 1 or mode not in {"bilinear", "bicubic", "lanczos"}:
        raise ValueError("Expected B C Z Y X, integer scale and bilinear/bicubic/lanczos")
    b, c, z, h, w = value.shape
    planes = value.permute(0, 2, 1, 3, 4).reshape(b*z, c, h, w)
    if mode != "lanczos":
        result = F.interpolate(planes, scale_factor=scale, mode=mode, align_corners=False)
    else:
        result = planes
        for axis in (-1, -2):
            n = result.shape[axis]
            position = (torch.arange(n*scale, device=value.device, dtype=torch.float32)+.5)/scale-.5
            indices = position.floor().long()[:, None]+torch.arange(-2, 4, device=value.device)
            distance = position[:, None]-indices
            weights = torch.sinc(distance)*torch.sinc(distance/3)
            weights = weights/weights.sum(-1, keepdim=True)
            indices = indices.clamp(0, n-1)
            result = result.movedim(axis, -1)
            result = (result[..., indices]*weights.to(result.dtype)).sum(-1)
            result = result.movedim(-1, axis)
    return result.reshape(b, z, c, h*scale, w*scale).permute(0, 2, 1, 3, 4)


@torch.no_grad()
def local_variance_ratio(prediction, reference):
    """Pooled valid 7x7 sample variances; excludes no tissue or background."""
    _, vp = local_moments(prediction)
    _, vr = local_moments(reference)
    denom = float(vr.double().sum())
    return float(vp.double().sum())/denom if denom > 0 else None


def ssim_map(prediction, reference):
    """Diagnostic map with the same uniform/sample-covariance arithmetic."""
    x, y = image_planes(prediction).double(), image_planes(reference).double()
    k = min(7, *x.shape[-2:]); k -= 1-k % 2
    if k < 3:
        raise ValueError("SSIM map requires XY>=3")
    ux, uy, exx, eyy, exy = F.avg_pool2d(torch.cat((x, y, x*x, y*y, x*y), 1), k, 1).unbind(1)
    factor = k*k/(k*k-1)
    vx, vy, cxy = factor*(exx-ux*ux), factor*(eyy-uy*uy), factor*(exy-ux*uy)
    return (2*ux*uy+.01**2)/(ux*ux+uy*uy+.01**2)*(2*cxy+.03**2)/(vx+vy+.03**2)


@torch.no_grad()
def textured_region_ssim(prediction, reference):
    """Diagnostic only: foreground variance>=1e-4, exclude foreground edge p90."""
    y = image_planes(reference).float()
    mean, variance = local_moments(reference)
    k = y.shape[-1]-mean.shape[-1]+1
    gradient = F.pad((y[..., 1:]-y[..., :-1]).abs(), (0, 1, 0, 0))+F.pad((y[..., 1:, :]-y[..., :-1, :]).abs(), (0, 0, 0, 1))
    gradient = F.avg_pool2d(gradient, k, 1)
    mask = (mean > 2/255) & (variance >= 1.e-4)
    for index in range(len(mask)):
        fg = mean[index] > 2/255
        if fg.any():
            threshold = torch.quantile(gradient[index][fg], .9)
            mask[index] &= (gradient[index] < threshold) | (gradient[index] == 0)
    values = ssim_map(prediction, reference)
    return float(values[mask[:, 0]].mean()) if mask.any() else None


@torch.no_grad()
def noise_ceiling_estimate(reference):
    """Noise-floor diagnostic under additive independent stationary-noise assumptions.

    A single magnitude MRI acquisition cannot identify irreducible noise or a
    physical SSIM upper bound. Texture, reconstruction correlations and Rician
    bias confound this robust Laplacian estimate. Report it as a model estimate.
    """
    y = image_planes(reference).float()
    mean, variance = local_moments(reference)
    k = y.shape[-1]-mean.shape[-1]+1
    radius = k//2
    kernel = y.new_tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
    laplace = F.conv2d(F.pad(y, (1, 1, 1, 1), mode="replicate"), kernel)
    laplace = laplace[..., radius:y.shape[-2]-radius, radius:y.shape[-1]-radius]
    background = mean <= 2/255
    foreground = ~background
    homogeneous = torch.zeros_like(foreground)
    if foreground.any():
        homogeneous = foreground & (variance <= torch.quantile(variance[foreground], .25))
    def estimate(mask):
        if not mask.any():
            return None
        values = laplace[mask]
        return float((values-values.median()).abs().median()/(.67448975*math.sqrt(20)))
    sigma_background, sigma_homogeneous = estimate(background), estimate(homogeneous)
    sigma = sigma_homogeneous if sigma_homogeneous is not None else sigma_background
    estimate_ssim = None
    if sigma is not None:
        signal_variance = (variance-sigma**2).clamp_min(0)
        estimate_ssim = float(((2*signal_variance+.03**2)/(2*signal_variance+sigma**2+.03**2)).mean())
    return dict(background_sigma=sigma_background, homogeneous_sigma=sigma_homogeneous,
                background_windows=int(background.sum()), homogeneous_windows=int(homogeneous.sum()),
                additive_noise_oracle_ssim_estimate=estimate_ssim,
                physical_upper_bound_identified=False,
                assumptions="Single-image iid additive noise approximation; not a physical SSIM bound")
