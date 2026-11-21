"""Native LR conditioning and lossless 16-band XY residuals for paired MRI."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .models import _TimeResidualBlock, batch_time
from .momentum.model import HaarXY
from .paths import linear_path, optimize_path, sample_path


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

    def forward(self, lr, *, descriptor_only=False):
        if lr.ndim != 5 or lr.shape[1] != 1:
            raise ValueError("Expected native LR of shape (B,1,Z,Y,X)")
        if descriptor_only:
            return self.layers[0](lr)
        features = []
        for layer in self.layers:
            lr = layer(lr)
            features.append(lr)
        return features


class HaarResidualNetwork(nn.Module):
    def __init__(self, mode, widths=(48, 96, 128), readout="integral", spatial=None):
        super().__init__()
        if mode not in HAAR_MODES or readout not in {"integral", "endpoint"}:
            raise ValueError("Unknown Haar MRI model or readout")
        self.mode, self.readout = mode, readout
        self.spatial = spatial
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
        if spatial is not None:
            # Learn corrections to both unconstrained heads in residual space.
            # Start from the ungated identity map, including in low-support regions.
            self.geometry_embedding = nn.Sequential(nn.Conv3d(65, a, 1), nn.SiLU())
            self.geometry_delta = nn.Conv3d(a, 32, 1)
            self.geometry_gate = nn.Conv3d(2 * a, 32, 1)
            nn.init.zeros_(self.geometry_delta.weight)
            nn.init.zeros_(self.geometry_delta.bias)
        for head in (self.endpoint, self.drift):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def forward(self, state, condition, time, direction=0, bound=None):
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
        clean = self.endpoint(hidden)
        drift = self.drift(hidden) if self.drift is not None else None
        if self.spatial is not None:
            if bound is None:
                raise ValueError("Spatial Haar bridge requires an LR-conditioned chart")
            query = state if self.spatial.get("query", "endpoint") == "state" else clean
            spatial = self.geometry_embedding(bound.features(bound.unwhiten(query)))
            gate = self.geometry_gate(torch.cat((hidden, spatial), dim=1)).sigmoid()
            dc, dd = (gate * self.geometry_delta(spatial)).chunk(2, dim=1)
            clean = clean + bound.whiten(dc)
            drift = drift + bound.whiten(dd)
        return clean, drift

    def zero_state(self, lr, base):
        if lr is None or lr.shape != (*base.shape[:3], base.shape[-2] // 4, base.shape[-1] // 4):
            raise ValueError("Real native LR must align with the XY x4 base and Z grid")
        return lr.new_zeros((len(lr), 16, *lr.shape[2:]))

    @torch.no_grad()
    def restore(self, lr, base, steps, sigma, generator=None, atlas=None, noise_steps=None):
        if not isinstance(steps, int) or steps < 1 or not math.isfinite(sigma) or sigma < 0:
            raise ValueError("Invalid Haar bridge sampler steps or sigma")
        state = self.zero_state(lr, base)
        condition = self.lr_encoder(lr)
        bound = atlas.bind(lr) if atlas is not None else None
        if self.mode == "haar_direct":
            residual, _ = self(state, condition, 0.)
        else:
            for i in range(steps):
                time = max(.02, min(.98, i / steps))
                endpoint, drift = self(state, condition, time, bound=bound)
                state = state + drift / steps
                if i + 1 < steps and sigma:
                    noise = (torch.randn(state.shape, dtype=state.dtype, device=state.device, generator=generator)
                             if noise_steps is None else noise_steps[i])
                    state = state + noise * (sigma / math.sqrt(steps))
            residual = endpoint if self.readout == "endpoint" else state
        return base + self.codec.decode(bound.unwhiten(residual) if bound is not None else residual)


class HaarObjective(nn.Module):
    def __init__(self, model, config, atlas=None):
        super().__init__()
        self.model = model
        self.config = config
        self.atlas = atlas

    def forward(self, lr, base, hr, case_ids=None):
        model = self.model
        condition = model.lr_encoder(lr)
        zero = model.zero_state(lr, base)
        if model.mode == "haar_direct":
            clean, _ = model(zero, condition, 0.)
            mse = F.mse_loss(base + model.codec.decode(clean), hr)
            return {"loss": mse, "endpoint_pixel_mse": mse}
        target = model.codec.encode(hr - base)
        bound, path_metric = None, None
        if self.atlas is not None:
            if case_ids is None:
                raise ValueError("TRAIN spatial queries require case IDs for held-out fold routing")
            bound = self.atlas.bind(lr, case_ids)
            target = bound.whiten(target)
            if model.spatial.get("path_geometry", False):
                path_metric = bound.path_metric()
        path_cfg = self.config["path"]
        if path_metric is None:
            # Exact minimum of the mean and variance control action.
            path = linear_path(zero, target, torch.zeros_like(zero), path_cfg["knots"], path_cfg["sigma"])
            diagnostics = {"mean_correction_rms": 0., "accepted_steps": 0}
        else:
            # CondSOC may draw quadrature noise. Preserve the matching-time and
            # noise RNG stream shared with the network-only path ablation.
            devices = [target.device] if target.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                path, diagnostics = optimize_path(zero, target, torch.zeros_like(zero), path_metric, path_cfg)
        cutoff = self.config["train"].get("matching_time_min", .02)
        t = torch.rand(len(hr), device=hr.device) * (1 - 2 * cutoff) + cutoff
        noise = torch.randn_like(target)
        state, f = sample_path(path, t, noise, "forward")
        _, r = sample_path(path, t, noise, "backward")
        clean, forward = model(state, condition, t, 0, bound)
        _, reverse = model(state, condition, t, 1, bound)
        fm, rm = F.mse_loss(forward, f), F.mse_loss(reverse, r)
        mse = F.mse_loss(base + model.codec.decode(bound.unwhiten(clean) if bound is not None else clean), hr)
        weight = self.config["train"].get("endpoint_weight", 1.)
        terms = {"loss": (fm + rm) * .5 + weight * mse, "endpoint_pixel_mse": mse,
                 "forward_matching": fm, "backward_matching": rm,
                 "path_mean_correction_rms": hr.new_tensor(diagnostics["mean_correction_rms"])}
        if bound is not None:
            terms.update({key: torch.as_tensor(value, device=hr.device) for key, value in bound.diagnostics().items()})
            rms = target.square().mean().sqrt().clamp_min(1.e-8)
            terms["path_relative_correction_rms"] = terms["path_mean_correction_rms"] / rms
            terms["path_accepted_steps"] = hr.new_tensor(diagnostics["accepted_steps"])
        return terms


def make_haar_model(config):
    return HaarResidualNetwork(config["mode"], config["model"]["widths"],
                               config["sampling"].get("readout", "integral"), config.get("spatial_atlas"))
