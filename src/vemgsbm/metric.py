"""Train-only, data-induced geometry in compact 3D patch representations.

This is a feature-space adaptation of the local second-moment/RBF metric in
Metric Flow Matching (K. Kapusniak et al., NeurIPS 2024), not an invertible
change of image coordinates. ``diagonal`` is the inverse unnormalised local
second moment plus a positive floor, as in LAND. A separate stable RBF
negative-log-support potential can be used as a GSBM state cost.

The representation is learned without anatomical labels; neither the feature
metric nor its state potential supplies a proof of anatomical/topological
preservation. The caller must supply features from training HR patches only.

Formula attribution: https://github.com/kkapusniak/metric-flow-matching
Copyright (c) 2024 Kacper Kapusniak, MIT License. See docs/THIRD_PARTY.md.
The code below is a standalone PyTorch adaptation; no Lightning/sklearn needed.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _volume(x: Tensor) -> None:
    if x.ndim != 5 or x.shape[1] != 1 or min(x.shape[2:]) < 1:
        raise ValueError("Expected floating-point images shaped [B,1,Z,Y,X]")
    if not x.is_floating_point():
        raise ValueError("Images must be floating point")


class Representation(nn.Module):
    """Small 3D denoising autoencoder with a spatially pooled feature head.

    The pooling grid retains coarse spatial layout within a patch. It is not a
    segmentation model and assumes neither a named organ nor a fixed manifold.
    Two XY stride-2 layers preserve shallow, anisotropic volume support.
    """

    def __init__(self, width: int = 8, feature_dim: int = 32, packed_xy: bool = False):
        super().__init__()
        if width < 2 or feature_dim < 2:
            raise ValueError("width and feature_dim must both be >= 2")
        self.width, self.feature_dim = int(width), int(feature_dim)
        self.packed_xy = packed_xy
        self.codec = None
        channels = 1
        if packed_xy:
            from .momentum.model import HaarXY
            self.codec = HaarXY()
            channels = self.codec.channels
        self.encoder = nn.Sequential(
            nn.Conv3d(channels, width, 3, padding=1), nn.SiLU(),
            nn.Conv3d(width, 2 * width, 3, stride=(1, 2, 2), padding=1), nn.SiLU(),
            nn.Conv3d(2 * width, 2 * width, 3, stride=(1, 2, 2), padding=1), nn.SiLU(),
            nn.Conv3d(2 * width, 2 * width, 3, padding=1), nn.SiLU(),
        )
        self.feature_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((2, 2, 2)), nn.Flatten(),
            nn.Linear(16 * width, feature_dim),
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(2 * width, 2 * width, 3, padding=1), nn.SiLU(),
            nn.Conv3d(2 * width, width, 3, padding=1), nn.SiLU(),
            nn.Conv3d(width, channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        _volume(x)
        return self.feature_head(self.encoder(self.codec.encode(x) if self.codec is not None else x))

    def reconstruct(self, x: Tensor) -> Tensor:
        _volume(x)
        packed = self.codec.encode(x) if self.codec is not None else x
        hidden = self.encoder(packed)
        hidden = F.interpolate(hidden, size=packed.shape[-3:], mode="trilinear", align_corners=False)
        decoded = self.decoder(hidden)
        return self.codec.decode(decoded) if self.codec is not None else decoded

    def export_config(self) -> dict:
        config = {"width": self.width, "feature_dim": self.feature_dim}
        if self.packed_xy:
            config["packed_xy"] = True
        return config


def _view(hr: Tensor) -> Tensor:
    shape = (hr.shape[0], 1, 1, 1, 1)
    contrast = 0.9 + 0.2 * torch.rand(shape, device=hr.device, dtype=hr.dtype)
    offset = 0.02 * torch.randn(shape, device=hr.device, dtype=hr.dtype)
    return (hr * contrast + offset + 0.02 * torch.randn_like(hr)).clamp(0, 1)


def representation_loss(model: Representation, hr: Tensor) -> dict[str, Tensor]:
    """Denoising + same-region invariance + anti-collapse feature regularisers.

    Reconstruction targets are original intensities. No spatial displacement is
    classified as a negative, and no skeleton/tree/anatomical labels are used.
    Variance/covariance diagnostics are meaningful with at least two items;
    single-item microbatches still receive denoising and invariance gradients.
    """
    _volume(hr)
    first, second = _view(hr), _view(hr)
    # Sparse blocks force recovery from local context without changing the target.
    grid = tuple(max(1, n // 8) for n in hr.shape[-3:])
    keep = (torch.rand((hr.shape[0], 1, *grid), device=hr.device) > 0.15).to(hr.dtype)
    keep = F.interpolate(keep, size=hr.shape[-3:], mode="nearest")
    corrupted = first * keep
    reconstructed = model.reconstruct(corrupted)
    a, b = model(first), model(second)
    reconstruction = (reconstructed - hr).square().mean()
    invariance = (a - b).square().mean()
    centered_a, centered_b = a - a.mean(0), b - b.mean(0)
    if hr.shape[0] > 1:
        variance = (F.relu(0.2 - (centered_a.square().mean(0) + 1e-4).sqrt()).mean()
                    + F.relu(0.2 - (centered_b.square().mean(0) + 1e-4).sqrt()).mean()) / 2
        covariance_a = centered_a.T @ centered_a / (hr.shape[0] - 1)
        covariance_b = centered_b.T @ centered_b / (hr.shape[0] - 1)
        off_a = covariance_a - torch.diag_embed(covariance_a.diagonal())
        off_b = covariance_b - torch.diag_embed(covariance_b.diagonal())
        covariance = (off_a.square().sum() + off_b.square().sum()) / (2 * a.shape[1])
    else:
        variance = a.sum() * 0
        covariance = a.sum() * 0
    loss = reconstruction + 0.1 * invariance + 0.1 * variance + 0.01 * covariance
    return {"loss": loss, "reconstruction": reconstruction, "invariance": invariance,
            "variance": variance, "covariance": covariance,
            "feature_std": (centered_a.square().mean(0) + 1e-8).sqrt().mean()}


class DataMetric(nn.Module):
    """Frozen 3D encoder plus RBF-supported positive diagonal local geometry.

    ``centers`` and ``variance`` are in standardised feature coordinates.
    ``bandwidth`` is a scalar standard deviation per centre using mean squared
    feature distance. ``diagonal`` consumes already standardised features;
    ``encode``, ``state_cost`` and ``kinetic`` consume image volumes.

    Only encoder parameters are frozen. Autograd through it to input images is
    preserved, which is required when optimising conditional bridge paths.
    """

    def __init__(self, encoder: nn.Module, centers: Tensor, variance: Tensor,
                 bandwidth: Tensor, feature_mean: Tensor, feature_std: Tensor,
                 floor: float = 0.05):
        super().__init__()
        if not math.isfinite(floor) or floor <= 0:
            raise ValueError("Metric floor must be finite and positive")
        centers = torch.as_tensor(centers).detach().clone()
        variance = torch.as_tensor(variance, device=centers.device, dtype=centers.dtype).detach().clone()
        bandwidth = torch.as_tensor(bandwidth, device=centers.device, dtype=centers.dtype).flatten().detach().clone()
        mean = torch.as_tensor(feature_mean, device=centers.device, dtype=centers.dtype).flatten().detach().clone()
        std = torch.as_tensor(feature_std, device=centers.device, dtype=centers.dtype).flatten().detach().clone()
        if (centers.ndim != 2 or min(centers.shape) < 1 or variance.shape != centers.shape
                or bandwidth.shape != centers.shape[:1] or mean.shape != centers.shape[1:]
                or std.shape != mean.shape):
            raise ValueError("Inconsistent metric centre/statistic shapes")
        if not centers.is_floating_point():
            raise ValueError("Metric statistics must be floating point")
        if not all(torch.isfinite(v).all() for v in (centers, variance, bandwidth, mean, std)):
            raise ValueError("Metric statistics must be finite")
        if not ((variance >= 0).all() and (bandwidth > 0).all() and (std > 0).all()):
            raise ValueError("Variance must be nonnegative; std and bandwidth positive")
        self.encoder = encoder.eval().requires_grad_(False)
        self.floor = float(floor)
        for name, value in (("centers", centers), ("variance", variance), ("bandwidth", bandwidth),
                            ("feature_mean", mean), ("feature_std", std)):
            self.register_buffer(name, value)
        # Normalise each feature's metric to mean one on cluster centres. This
        # fixes a nuisance global scale while retaining location dependence.
        with torch.no_grad():
            normalizer = self._raw_diagonal(centers).mean(0).clamp_min(1e-6)
        self.register_buffer("normalizer", normalizer)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def encode(self, image: Tensor) -> Tensor:
        return (self.encoder(image) - self.feature_mean) / self.feature_std

    def _kernels(self, features: Tensor) -> tuple[Tensor, Tensor]:
        if features.ndim != 2 or features.shape[1] != self.centers.shape[1]:
            raise ValueError("Features must be [batch, feature_dim]")
        differences = features[:, None, :] - self.centers[None, :, :]
        log_kernel = -0.5 * differences.square().mean(-1) / self.bandwidth.square()[None, :]
        return differences, log_kernel

    def _raw_diagonal(self, features: Tensor) -> Tensor:
        differences, log_kernel = self._kernels(features)
        local_second = ((differences.square() + self.variance[None]) * log_kernel.exp()[..., None]).sum(1)
        return (local_second + self.floor).reciprocal()

    def diagonal(self, features: Tensor) -> Tensor:
        # Strict SPD feature metric with bounded entries. Bounds also keep rare
        # feature coordinates from dominating the finite-path objective.
        return (self._raw_diagonal(features) / self.normalizer).clamp(1e-3, 1e3)

    def state_cost_features(self, features: Tensor) -> Tensor:
        """Stable negative log mean RBF support, unbounded away from data.

        logsumexp avoids the vanishing gradients caused by clamping an RBF
        density after all kernels underflow. Cost is not a calibrated density.
        """
        _, log_kernel = self._kernels(features)
        return -torch.logsumexp(log_kernel, dim=1) + math.log(self.centers.shape[0])

    def state_cost(self, image: Tensor) -> Tensor:
        return self.state_cost_features(self.encode(image))

    def kinetic(self, x0: Tensor, x1: Tensor, dt: float | Tensor) -> Tensor:
        """Feature velocity energy under midpoint data-induced diagonal G."""
        first, second = self.encode(x0), self.encode(x1)
        dt = torch.as_tensor(dt, device=first.device, dtype=first.dtype)
        if not torch.isfinite(dt).all() or not (dt > 0).all():
            raise ValueError("dt must be finite and positive")
        if dt.ndim == 1:
            dt = dt[:, None]
        velocity = (second - first) / dt
        return (self.diagonal((first + second) / 2) * velocity.square()).mean(1)

    def export_config(self) -> dict:
        if not isinstance(self.encoder, Representation):
            raise TypeError("Portable metric payloads require a Representation encoder")
        return {"encoder": self.encoder.export_config(), "floor": self.floor,
                "clusters": self.centers.shape[0], "feature_dim": self.centers.shape[1]}


def fit_metric(encoder: Representation, feature_batches: Tensor | Iterable[Tensor],
               clusters: int = 32, seed: int = 42, iterations: int = 20,
               floor: float = 0.05) -> DataMetric:
    """Fit deterministic local RBF statistics from *training HR features only*.

    Features must be raw outputs of the supplied frozen-inference encoder.
    K-means runs on CPU in float32, independent of GPU/world size. The first
    centre is seeded, subsequent centres use farthest-point initialisation.
    A deepcopy prevents fitting from freezing the caller's pretraining model.
    """
    if isinstance(clusters, bool) or not isinstance(clusters, int) or clusters < 1 or iterations < 1:
        raise ValueError("clusters and iterations must be positive integers")
    values = (feature_batches if isinstance(feature_batches, Tensor)
              else torch.cat([part.detach().cpu() for part in feature_batches], dim=0))
    values = values.detach().to(device="cpu", dtype=torch.float32)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != encoder.feature_dim:
        raise ValueError("Need >=2 training feature vectors of encoder.feature_dim")
    if not torch.isfinite(values).all():
        raise ValueError("Training features must be finite")
    mean = values.mean(0)
    # A finite lower bound stops near-constant learned channels from amplifying
    # input derivatives by many orders of magnitude.
    std = values.std(0, unbiased=False).clamp_min(0.05)
    z = (values - mean) / std
    count, dimension = z.shape
    k = min(clusters, count)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    chosen = [int(torch.randint(count, (1,), generator=generator))]
    minimum = (z - z[chosen[0]]).square().sum(1)
    for _ in range(1, k):
        index = int(minimum.argmax())
        chosen.append(index)
        minimum = torch.minimum(minimum, (z - z[index]).square().sum(1))
    centers = z[chosen].clone()
    with torch.no_grad():
        for _ in range(iterations):
            distances = torch.cdist(z, centers).square()
            labels = distances.argmin(1)
            updated = centers.clone()
            for index in range(k):
                members = z[labels == index]
                if members.shape[0]:
                    updated[index] = members.mean(0)
            converged = torch.allclose(updated, centers, atol=1e-6, rtol=0)
            centers = updated
            if converged:
                break
        labels = torch.cdist(z, centers).argmin(1)
        variance = torch.empty_like(centers)
        global_variance = z.var(0, unbiased=False).clamp_min(0.01)
        for index in range(k):
            members = z[labels == index]
            variance[index] = ((members - centers[index]).square().mean(0).clamp_min(0.01)
                               if members.shape[0] > 1 else global_variance)
        bandwidth = variance.mean(1).sqrt().clamp_min(0.3)
    fitted_encoder = copy.deepcopy(encoder).to("cpu").eval().requires_grad_(False)
    return DataMetric(fitted_encoder, centers, variance, bandwidth, mean, std, floor=floor)


def metric_payload(metric: DataMetric) -> dict:
    """Tensor-only/state-dict payload compatible with torch weights_only=True."""
    return {"format_version": 1, "config": metric.export_config(),
            "state": {name: value.detach().cpu().clone() for name, value in metric.state_dict().items()}}


def metric_from_payload(payload: dict, device: str | torch.device = "cpu") -> DataMetric:
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported metric payload format")
    cfg, state = payload["config"], payload["state"]
    encoder = Representation(**cfg["encoder"])
    metric = DataMetric(encoder, state["centers"], state["variance"], state["bandwidth"],
                        state["feature_mean"], state["feature_std"], floor=cfg["floor"])
    metric.load_state_dict(state, strict=True)
    return metric.to(device).eval().requires_grad_(False)
