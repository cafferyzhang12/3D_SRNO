"""Small full-volume reconstruction networks for anisotropic microscopy."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def volume_inputs(state: Tensor, base: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Validate the image contract and broadcast a binary observation mask."""
    if state.ndim != 5 or state.shape[1] != 1:
        raise ValueError("Images must have shape (batch, 1, z, y, x).")
    if base.shape != state.shape or base.device != state.device:
        raise ValueError("base must have the same shape and device as state.")
    if not state.is_floating_point() or not base.is_floating_point():
        raise ValueError("Images must be floating-point tensors.")
    if mask.device != state.device:
        raise ValueError("mask must have the same device as state.")
    try:
        mask = torch.broadcast_to(mask, state.shape)
    except RuntimeError as error:
        raise ValueError("mask must broadcast to the image shape.") from error
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must be binary, with one at observed voxels.")
    return base.to(dtype=state.dtype), mask.to(dtype=state.dtype)


def batch_time(time: Tensor | float, reference: Tensor) -> Tensor:
    """Return a validated scalar time per batch item."""
    value = torch.as_tensor(time, device=reference.device, dtype=reference.dtype)
    if value.numel() == 1:
        value = value.reshape(1).expand(reference.shape[0])
    elif value.numel() == reference.shape[0]:
        value = value.reshape(reference.shape[0])
    else:
        raise ValueError("time must contain one scalar or one value per batch item.")
    if not torch.all(torch.isfinite(value) & (value >= 0) & (value <= 1)):
        raise ValueError("time must be finite and lie in [0, 1].")
    return value


def _groups(channels: int) -> int:
    for group_count in (4, 2, 1):
        if channels % group_count == 0 and channels // group_count >= 2:
            return group_count
    return 1


class _TimeResidualBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, time_width: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(inputs), inputs)
        self.conv1 = nn.Conv3d(inputs, outputs, 3, padding=1)
        self.time_projection = nn.Linear(time_width, outputs)
        self.norm2 = nn.GroupNorm(_groups(outputs), outputs)
        self.conv2 = nn.Conv3d(outputs, outputs, 3, padding=1)
        self.skip = nn.Identity() if inputs == outputs else nn.Conv3d(inputs, outputs, 1)

    def forward(self, image: Tensor, embedding: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(image)))
        hidden = hidden + self.time_projection(embedding)[:, :, None, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return self.skip(image) + hidden


class Reconstructor(nn.Module):
    """Time-conditioned 3D U-Net predicting clean HR intensities.

    Its residual is unbounded. Observed voxels are copied from ``base`` exactly.
    Interpolation to encoder shapes supports odd and non-cubic volume sizes.
    """

    def __init__(self, width: int = 16):
        super().__init__()
        if width < 2:
            raise ValueError("width must be at least two.")
        time_width = 4 * width
        self.register_buffer(
            "time_frequencies", torch.exp(torch.linspace(0, math.log(1000), 2 * width))
        )
        self.time_network = nn.Sequential(
            nn.Linear(time_width, time_width), nn.SiLU(), nn.Linear(time_width, time_width)
        )
        self.direction_embedding = nn.Embedding(2, time_width)
        self.input_projection = nn.Conv3d(3, width, 3, padding=1)
        self.encoder1 = _TimeResidualBlock(width, width, time_width)
        self.down1 = nn.Conv3d(width, 2 * width, 3, stride=2, padding=1)
        self.encoder2 = _TimeResidualBlock(2 * width, 2 * width, time_width)
        self.down2 = nn.Conv3d(2 * width, 4 * width, 3, stride=2, padding=1)
        self.bottleneck = _TimeResidualBlock(4 * width, 4 * width, time_width)
        self.decoder2 = _TimeResidualBlock(6 * width, 2 * width, time_width)
        self.decoder1 = _TimeResidualBlock(3 * width, width, time_width)
        self.output_projection = nn.Conv3d(width, 1, 3, padding=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, state: Tensor, base: Tensor, mask: Tensor, time: Tensor | float, direction: int = 0) -> Tensor:
        base, mask = volume_inputs(state, base, mask)
        times = batch_time(time, state)
        angles = times[:, None] * self.time_frequencies.to(dtype=state.dtype)[None, :]
        embedding = self.time_network(torch.cat((angles.sin(), angles.cos()), dim=1))
        embedding = embedding + self.direction_embedding.weight[direction][None, :]
        first = self.encoder1(self.input_projection(torch.cat((state, base, mask), 1)), embedding)
        second = self.encoder2(self.down1(first), embedding)
        hidden = self.bottleneck(self.down2(second), embedding)
        hidden = F.interpolate(hidden, size=second.shape[-3:], mode="trilinear", align_corners=False)
        hidden = self.decoder2(torch.cat((hidden, second), 1), embedding)
        hidden = F.interpolate(hidden, size=first.shape[-3:], mode="trilinear", align_corners=False)
        hidden = self.decoder1(torch.cat((hidden, first), 1), embedding)
        clean = base + self.output_projection(hidden)
        return torch.where(mask.bool(), base, clean)
