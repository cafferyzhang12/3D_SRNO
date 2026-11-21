"""Lossless spatial residual coordinates and a 3D acceleration network."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..models import _TimeResidualBlock, batch_time, volume_inputs
from .guidance import atlas_anchor_update
from .phase import transition


class HaarXY(nn.Module):
    """Orthonormal 4x4 Haar transform with no discarded spatial coefficients."""
    factor = 4
    channels = 16

    def __init__(self):
        super().__init__()
        root = math.sqrt(0.5)
        haar = torch.tensor([[.5, .5, .5, .5], [.5, .5, -.5, -.5],
                             [root, -root, 0, 0], [0, 0, root, -root]])
        self.register_buffer("basis", torch.kron(haar, haar))

    def encode(self, image):
        b, c, d, y, x = image.shape
        if c != 1 or y % self.factor or x % self.factor:
            raise ValueError("Haar expects one channel and XY sizes divisible by four")
        planes = image.transpose(1, 2).reshape(b*d, 1, y, x)
        blocks = F.pixel_unshuffle(planes, self.factor).reshape(b, d, self.channels, y//4, x//4)
        return torch.einsum("oc,bdcyx->bodyx", self.basis.to(image), blocks)

    def decode(self, coefficients):
        b, c, d, y, x = coefficients.shape
        if c != self.channels:
            raise ValueError("Unexpected number of Haar coefficients")
        blocks = torch.einsum("oc,bodyx->bdcyx", self.basis.to(coefficients), coefficients)
        planes = F.pixel_shuffle(blocks.reshape(b*d, c, y, x), self.factor)
        return planes.reshape(b, d, 1, 4*y, 4*x).transpose(1, 2)

    def free(self, mask):
        blocks = F.avg_pool3d(mask, kernel_size=(1, 4, 4), stride=(1, 4, 4))
        if not bool(((blocks == 0) | (blocks == 1)).all()):
            raise ValueError("Exact observations must cover complete Haar blocks")
        return 1 - blocks


class AccelerationNetwork(nn.Module):
    """Predict e in a=3*(e-z-(1-t)*v)/(1-t)^2 to scale acceleration matching."""
    def __init__(self, width=24, chart_features=False):
        super().__init__()
        self.codec = HaarXY()
        self.chart_features = chart_features
        channels = self.codec.channels
        embedding = 4 * width
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, math.log(1000), 2*width)))
        self.time_network = nn.Sequential(nn.Linear(embedding, embedding), nn.SiLU(), nn.Linear(embedding, embedding))
        self.input_projection = nn.Conv3d((5 if chart_features else 3) * channels, width, 3, padding=1)
        self.encoder = _TimeResidualBlock(width, width, embedding)
        self.down = nn.Conv3d(width, 2*width, 3, stride=(1,2,2), padding=1)
        self.mid = _TimeResidualBlock(2*width, 2*width, embedding)
        self.decoder = _TimeResidualBlock(3*width, width, embedding)
        self.output = nn.Conv3d(width, channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, position, velocity, condition, time, physical=None):
        time = batch_time(time, position)
        angles = time[:, None] * self.frequencies.to(position)[None]
        embedding = self.time_network(torch.cat((angles.sin(), angles.cos()), dim=1))
        inputs = (position, velocity, condition)
        if self.chart_features:
            if physical is None:
                raise ValueError("Chart momentum requires physical position and velocity features")
            inputs += physical
        state = self.input_projection(torch.cat(inputs, dim=1))
        skip = self.encoder(state, embedding)
        hidden = self.mid(self.down(skip), embedding)
        hidden = F.interpolate(hidden, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return self.output(self.decoder(torch.cat((hidden, skip), dim=1), embedding))

    @torch.no_grad()
    def restore(self, base, mask, steps=20, sigma=0.15, generator=None, geometry=None, geometry_guidance=None):
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("Expected positive integer steps")
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("Expected finite nonnegative reference noise")
        base, mask = volume_inputs(base, base, mask)
        free = self.codec.free(mask)
        condition = self.codec.encode(base)
        if self.chart_features and geometry is None:
            raise ValueError("Chart momentum sampling requires its fitted MLS atlas")
        frame = geometry.bind(condition, free) if self.chart_features else None
        bound = None
        if geometry_guidance is not None:
            if self.chart_features or not getattr(geometry, "source_routing", False):
                raise ValueError("Momentum force requires its fitted LR-routed MRI atlas")
            bound = geometry.bind(base)
        position = torch.zeros_like(condition)
        velocity = torch.zeros_like(condition)
        for step in range(steps):
            time = step / steps
            physical = frame.features(position, velocity) if frame is not None else None
            anchor = self(position, velocity, condition, time, physical) * free
            if bound is not None:
                anchor = anchor + atlas_anchor_update(codec=self.codec, bound=bound, base=base, mask=mask,
                                                       position=position, velocity=velocity, time=time,
                                                       options=geometry_guidance)
            # Integrate the stiff reference acceleration exactly while holding
            # this learned effective endpoint fixed over the current step.
            # In chart mode, the next query pushes both updated coordinates
            # through Φ and DΦ, without projecting away the normal residual.
            position, velocity = transition(position, velocity, anchor, 1 / steps,
                                            (steps - step) / steps, sigma, generator)
            position = position * free
            velocity = velocity * free
        if frame is not None:
            position = frame.to_image(position)
        if not torch.isfinite(position).all():
            raise FloatingPointError("Nonfinite momentum bridge output")
        return torch.where(mask.bool(), base, base + self.codec.decode(position))
