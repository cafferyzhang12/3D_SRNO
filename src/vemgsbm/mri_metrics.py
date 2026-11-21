"""Slice-wise MRI structural similarity for validation."""

import torch
from torch.nn.functional import avg_pool2d


def mean_xy_ssim(prediction, reference):
    """Mean 2D SSIM over batch and Z in the normalized [0, 1] image range.

    The uniform window, covariance normalization and border exclusion match
    the full-case test convention in mri_infer.case_metrics.
    """
    if prediction.ndim != 5 or prediction.shape[1] != 1 or prediction.shape != reference.shape:
        raise ValueError("MRI SSIM inputs must have the same B1ZYX shape")
    height, width = prediction.shape[-2:]
    window = min(7, height, width)
    window -= 1 - window % 2
    if window < 3:
        raise ValueError("MRI SSIM needs at least three pixels on each in-plane axis")

    x = prediction.reshape(-1, 1, height, width).to(torch.float64)
    y = reference.reshape_as(x).to(torch.float64)
    ux, uy, exx, eyy, exy = avg_pool2d(
        torch.cat((x, y, x.square(), y.square(), x * y), dim=1), window, stride=1
    ).unbind(dim=1)
    correction = window**2 / (window**2 - 1)
    vx, vy = correction * (exx - ux.square()), correction * (eyy - uy.square())
    covariance = correction * (exy - ux * uy)
    luminance = (2 * ux * uy + .01**2) / (ux.square() + uy.square() + .01**2)
    structure = (2 * covariance + .03**2) / (vx + vy + .03**2)
    return (luminance * structure).mean()
