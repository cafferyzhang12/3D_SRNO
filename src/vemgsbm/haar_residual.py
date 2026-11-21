"""Native LR conditioning and lossless 16-band XY residuals for paired MRI."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .models import _TimeResidualBlock, batch_time
from .momentum.model import HaarXY
from .paths import optimize_path, sample_path


HAAR_MODES = {"haar_direct", "haar_bridge"}


class LRConditionEncoder(nn.Module):
    """A frozen copy of this encoder can describe LR sites without HR input."""

    def __init__(self, widths):
        super().__init__()
        a, b, c = widths
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Conv3d(cin, cout, 3, stride=stride, padding=1), nn.SiLU(),
                          nn.Conv3d(cout, cout, 3, padding=1))
            for cin, cout, stride in ((1, a, 1), (a, b, (1, 2, 2)), (b, c, (1, 2, 2)))
        ])

    def forward(self, lr):
        if lr.ndim != 5 or lr.shape[1] != 1:
            raise ValueError("Expected native LR of shape (B,1,Z,Y,X)")
        features = []
        for layer in self.layers:
            lr = layer(lr)
            features.append(lr)
        return features


class HaarResidualNetwork(nn.Module):
    def __init__(self, mode, widths=(48, 96, 128), readout="integral"):
        super().__init__()
        if mode not in HAAR_MODES or readout not in {"integral", "endpoint"}:
            raise ValueError("Unknown Haar MRI model or readout")
        self.mode, self.readout = mode, readout
        self.codec = HaarXY()
        self.lr_encoder = LRConditionEncoder(widths)
        a, b, c = widths
        embedding = 4 * a
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, math.log(1000), 2 * a)))
        self.time = nn.Sequential(nn.Linear(embedding, embedding), nn.SiLU(), nn.Linear(embedding, embedding))
        self.direction = nn.Embedding(2, embedding)
        self.state = nn.Conv3d(16, a, 3, padding=1)
        self.first = _TimeResidualBlock(a, a, embedding)
        self.down1 = nn.Conv3d(a, b, 3, stride=(1, 2, 2), padding=1)
        self.second = _TimeResidualBlock(b, b, embedding)
        self.down2 = nn.Conv3d(b, c, 3, stride=(1, 2, 2), padding=1)
        self.deep = _TimeResidualBlock(c, c, embedding)
        self.up2 = _TimeResidualBlock(c + b, b, embedding)
        self.up1 = _TimeResidualBlock(b + a, a, embedding)
        self.endpoint = nn.Conv3d(a, 16, 3, padding=1)
        self.drift = nn.Conv3d(a, 16, 3, padding=1) if mode == "haar_bridge" else None
        for head in (self.endpoint, self.drift):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def forward(self, state, condition, time, direction=0):
        angles = batch_time(time, state)[:, None] * self.frequencies.to(state)
        embedding = self.time(torch.cat((angles.sin(), angles.cos()), dim=1))
        embedding = embedding + self.direction.weight[direction]
        low, middle, deep = condition
        a = self.first(self.state(state) + low, embedding)
        b = self.second(self.down1(a) + middle, embedding)
        c = self.deep(self.down2(b) + deep, embedding)
        def align(x, skip):
            return torch.cat((F.interpolate(x, size=skip.shape[-3:], mode="trilinear",
                                            align_corners=False), skip), dim=1)
        hidden = self.up1(align(self.up2(align(c, b), embedding), a), embedding)
        return self.endpoint(hidden), self.drift(hidden) if self.drift is not None else None

    def zero_state(self, lr, base):
        if lr is None or lr.shape != (*base.shape[:3], base.shape[-2] // 4, base.shape[-1] // 4):
            raise ValueError("Real native LR must align with the XY x4 base and Z grid")
        return lr.new_zeros((len(lr), 16, *lr.shape[2:]))

    @torch.no_grad()
    def restore(self, lr, base, steps, sigma, generator=None):
        if not isinstance(steps, int) or steps < 1 or not math.isfinite(sigma) or sigma < 0:
            raise ValueError("Invalid Haar bridge sampler steps or sigma")
        state = self.zero_state(lr, base)
        condition = self.lr_encoder(lr)
        if self.mode == "haar_direct":
            residual, _ = self(state, condition, 0.)
        else:
            for i in range(steps):
                time = max(.02, min(.98, i / steps))
                endpoint, drift = self(state, condition, time)
                state = state + drift / steps
                if i + 1 < steps and sigma:
                    state = state + torch.randn(state.shape, dtype=state.dtype, device=state.device,
                                                generator=generator) * (sigma / math.sqrt(steps))
            residual = endpoint if self.readout == "endpoint" else state
        return base + self.codec.decode(residual)


class HaarObjective(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, lr, base, hr):
        model = self.model
        condition = model.lr_encoder(lr)
        zero = model.zero_state(lr, base)
        if model.mode == "haar_direct":
            clean, _ = model(zero, condition, 0.)
            mse = F.mse_loss(base + model.codec.decode(clean), hr)
            return {"loss": mse, "endpoint_pixel_mse": mse}
        target = model.codec.encode(hr - base)
        path, diagnostics = optimize_path(zero, target, torch.zeros_like(zero), None, self.config["path"])
        cutoff = self.config["train"].get("matching_time_min", .02)
        t = torch.rand(len(hr), device=hr.device) * (1 - 2 * cutoff) + cutoff
        noise = torch.randn_like(target)
        state, f = sample_path(path, t, noise, "forward")
        _, r = sample_path(path, t, noise, "backward")
        clean, forward = model(state, condition, t, 0)
        _, reverse = model(state, condition, t, 1)
        fm, rm = F.mse_loss(forward, f), F.mse_loss(reverse, r)
        mse = F.mse_loss(base + model.codec.decode(clean), hr)
        weight = self.config["train"].get("endpoint_weight", 1.)
        return {"loss": (fm + rm) * .5 + weight * mse, "endpoint_pixel_mse": mse,
                "forward_matching": fm, "backward_matching": rm,
                "path_mean_correction_rms": hr.new_tensor(diagnostics["mean_correction_rms"])}


def make_haar_model(config):
    return HaarResidualNetwork(config["mode"], config["model"]["widths"],
                               config["sampling"].get("readout", "integral"))
