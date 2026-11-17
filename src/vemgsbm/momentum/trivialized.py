"""Chart-trivialized kinetic bridges on lossless Haar residual volumes."""

import math

import torch
from torch import nn
from torch.nn.functional import one_hot

from ..atlas_geometry import fit_feature_atlas, route_features, trusted_mls_decode
from .phase import PhasePath, optimize_path


def chart_momentum_options(config):
    options = config.get("chart_momentum")
    if options is None:
        return None
    if (config["mode"] != "momentum" or config["data"].get("kind") != "paired_mri_png"
            or not (config.get("atlas") or {}).get("source_routing")):
        raise ValueError("Chart momentum requires paired MRI and an LR-routed MLS atlas")
    if not isinstance(options, dict) or set(options) - {"warp_strength", "normal_noise_ratio", "scale_floor"}:
        raise ValueError("Unknown chart momentum options")
    values = {"warp_strength": .75, "normal_noise_ratio": .75, "scale_floor": .015, **options}
    if (not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                for x in values.values())
            or not 0 < values["warp_strength"] <= 1 or not 0 < values["normal_noise_ratio"] <= 1
            or values["scale_floor"] <= 0):
        raise ValueError("Invalid MLS chart momentum strength, noise ratio or standardization floor")
    return values


class SpatialCells:
    """Average Haar channels within up to 2x2x2 disjoint spatial cells."""

    def __init__(self, free):
        shape = free.shape[-3:]
        axes = [torch.arange(n, device=free.device) * min(n, 2) // n for n in shape]
        labels = ((axes[0][:, None, None] * min(shape[1], 2) + axes[1][None, :, None])
                  * min(shape[2], 2) + axes[2][None, None, :])
        self.indicators = one_hot(labels.flatten(), math.prod(min(n, 2) for n in shape)).T.to(free)
        self.free = free
        count = free.flatten(2) @ self.indicators.T
        self.active = (count.squeeze(1) > 0).flatten()
        self.reduction = count.clamp_min(1)
        self.orthogonal_scale = self.reduction.reshape(-1, 1).sqrt()

    def describe(self, coefficients):
        pooled = ((coefficients * self.free).flatten(2) @ self.indicators.T) / self.reduction
        return pooled.transpose(1, 2).reshape(-1, coefficients.shape[1])

    def lift(self, descriptors):
        values = descriptors.reshape(len(self.free), self.indicators.shape[0], -1)
        return (values.transpose(1, 2) @ self.indicators).reshape(
            len(self.free), values.shape[-1], *self.free.shape[-3:]
        ) * self.free


class ChartMomentumAtlas(nn.Module):
    """Frozen MLS geometry fitted to coarse Haar moments of paired MRI volumes."""

    def __init__(self, payload, options):
        super().__init__()
        from neural_atlas.mls_atlas import MLSAtlas

        if payload.get("kind") != "haar_mls" or payload.get("routing", {}).get("kind") != "paired_lr_kernel":
            raise ValueError("Expected the fitted Haar MLS atlas with paired LR routing")
        self.atlas = MLSAtlas.from_checkpoint(payload["atlas"])
        self.options = options
        routing = payload["routing"]
        self.routing_settings = routing["settings"]
        for name in ("source_mean", "source_scale", "source_bank", "chart_probabilities"):
            self.register_buffer(name, routing[name].clone())
        transform = payload["atlas"]["transform"]
        self.register_buffer("feature_mean", torch.tensor(transform["mean"]))
        self.register_buffer("feature_scale", torch.tensor(transform["scale"]))
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def bind(self, condition, free):
        cells = SpatialCells(free)
        base = cells.describe(condition)
        normalized = (base - self.source_mean) / self.source_scale
        ids, prior, confidence = route_features(normalized, self.source_bank, self.chart_probabilities,
                                                self.routing_settings)
        # Selection is fixed within each trajectory. Changing basis while
        # integrating would require transition maps for both position and momentum.
        confidence = confidence * prior[:, 0] * cells.active
        return ChartMomentumFrame(self, ids[:, 0], cells, (base - self.feature_mean) / self.feature_scale,
                                  confidence)


class ChartMomentumFrame:
    """Full-rank normal-graph extension with a fixed LR-selected body frame.

    Cell coordinates tangential to the reference chart pass through unchanged.
    The remaining cell coordinates have a positive scale and a nonlinear
    normal displacement. Spatial variation within each Haar cell is retained.
    """

    def __init__(self, geometry, ids, cells, source, confidence):
        self.geometry, self.ids, self.cells, self.source = geometry, ids, cells, source
        atlas = geometry.atlas
        self.basis = atlas.basis[ids]
        self.centers, self.scales = atlas.centers[ids], atlas.scales[ids, None]
        self.lower, self.upper = atlas.domain_lo[ids], atlas.domain_hi[ids]
        self.confidence = confidence[:, None]
        self.warp = geometry.options["warp_strength"] * self.confidence
        self.normal_scale = 1 - (1 - geometry.options["normal_noise_ratio"]) * self.confidence
        self.source_coordinate = self._pull(source - self.centers) / self.scales
        self.source_bend, _, _ = self._bend(self.source_coordinate, derivative=False)

    def _pull(self, value):
        return (self.basis.mT @ value[..., None]).squeeze(-1)

    def _normal(self, value):
        return value - (self.basis @ self._pull(value)[..., None]).squeeze(-1)

    def _bend(self, coordinate, derivative):
        center = (self.lower + self.upper) * .5
        radius = ((self.upper - self.lower) * .5).clamp_min(1e-4)
        relative = ((coordinate - center) / radius).tanh()
        bounded = center + radius * relative
        decoded, jacobian, support = trusted_mls_decode(
            self.geometry.atlas, self.ids, bounded, self.geometry.routing_settings["correction_limit"],
            derivative=derivative,
        )
        graph_basis = self.basis * self.scales[..., None]
        linear = self.centers + (graph_basis @ bounded[..., None]).squeeze(-1)
        bend = self._normal(decoded - linear)
        if derivative:
            normal_jacobian = jacobian - graph_basis
            normal_jacobian = normal_jacobian - self.basis @ (self.basis.mT @ normal_jacobian)
            jacobian = normal_jacobian * (1 - relative.square())[:, None, :]
        return bend, jacobian, support

    def _coordinate(self, moment):
        return self.source_coordinate + self._pull(moment) / self.scales

    def to_body(self, position):
        """Recover the full body position analytically from a Haar residual."""
        moment = self.cells.describe(position) / self.geometry.feature_scale
        coordinate = self._coordinate(moment)
        bend, _, _ = self._bend(coordinate, derivative=False)
        normal = self._normal(moment)
        corrected = (normal - self.warp * (bend - self.source_bend)) / self.normal_scale
        return position + self.cells.lift((corrected - normal) * self.geometry.feature_scale)

    def to_image(self, position):
        moment = self.cells.describe(position) / self.geometry.feature_scale
        coordinate = self._coordinate(moment)
        bend, _, _ = self._bend(coordinate, derivative=False)
        shift = (self.normal_scale - 1) * self._normal(moment) + self.warp * (bend - self.source_bend)
        return position + self.cells.lift(shift * self.geometry.feature_scale)

    def _phase(self, moment, vector):
        bend, jacobian, support = self._bend(self._coordinate(moment), derivative=True)
        tangent_velocity = self._pull(vector) / self.scales
        bend_velocity = (jacobian @ tangent_velocity[..., None]).squeeze(-1)
        shift = (self.normal_scale - 1) * self._normal(moment) + self.warp * (bend - self.source_bend)
        shift_velocity = (self.normal_scale - 1) * self._normal(vector) + self.warp * bend_velocity
        return bend, bend_velocity, shift, shift_velocity, support

    def features(self, position, momentum):
        """Physical position and velocity, using the MLS Jacobian for DΦ(q)p."""
        scale = self.geometry.feature_scale
        moment = self.cells.describe(position) / scale
        vector = self.cells.describe(momentum) / scale
        _, _, shift, shift_velocity, _ = self._phase(moment, vector)
        return position + self.cells.lift(shift * scale), momentum + self.cells.lift(shift_velocity * scale)

    def _cell_cost(self, position, momentum, _free):
        scale = self.cells.orthogonal_scale * self.geometry.feature_scale
        moment, vector = position / scale, momentum / scale
        bend, bend_velocity, shift, shift_velocity, support = self._phase(moment, vector)
        residual = self._normal(self.source + moment + shift - self.centers) - bend
        normal_velocity = self._normal(vector + shift_velocity) - bend_velocity
        state = residual.square().mean(-1)
        velocity = normal_velocity.square().mean(-1)
        weight = self.confidence[:, 0] * (.1 + .9 * support)
        count = self.cells.active.sum().clamp_min(1)
        return ((weight * (2 * state / (1 + (1 + state).sqrt()))).sum() / count,
                (weight * velocity).sum() / count)

    def optimize(self, endpoint, config):
        # Geometry depends only on cell means. Use unit-norm mean modes so the
        # projected Brownian noise is standard normal and control energy exact.
        scale = self.cells.orthogonal_scale
        compact = self.cells.describe(endpoint) * scale
        free = self.cells.active[:, None].to(endpoint)
        fitted, details = optimize_path(compact, free, self._cell_cost, config,
                                        control_scale=compact.numel() / endpoint.numel())
        shifts = torch.stack([self.cells.lift(shift / scale) for shift in fitted.shifts])
        details["path_shift_rms"] = float(shifts.square().mean().sqrt())
        return PhasePath(endpoint, shifts, fitted.sigma, self.cells.free), details


@torch.no_grad()
def _collect_cell_features(codec, loader, limit, context, paired=False):
    hr_rows, lr_rows, remaining, failure = [], [], limit, None
    try:
        for batch in loader:
            if not remaining:
                break
            hr = batch["hr"][:remaining].to(context.device)
            free = codec.free(batch["mask"][:remaining].to(context.device))
            cells = SpatialCells(free)
            hr_rows.append(cells.describe(codec.encode(hr))[cells.active].cpu())
            if paired:
                base = batch["base"][:remaining].to(context.device)
                lr_rows.append(cells.describe(codec.encode(base))[cells.active].cpu())
            remaining -= len(hr)
    except Exception as exc:
        failure = f"Chart-momentum descriptors: {exc}"
    context.check_errors(failure)
    empty = torch.empty(0, codec.channels)
    local = (torch.cat(hr_rows) if hr_rows else empty, torch.cat(lr_rows) if lr_rows else empty)
    if not context.is_distributed:
        return local
    rows = [None] * context.world_size
    torch.distributed.all_gather_object(rows, local)
    return tuple(torch.cat([item[i] for item in rows]) for i in range(2))


def fit_chart_momentum_atlas(codec, loader, val_loader, cfg, context, options):
    bank_patches = cfg["atlas"]["bank_patches"]
    if not isinstance(bank_patches, int) or isinstance(bank_patches, bool) or bank_patches < 1:
        raise ValueError("Atlas bank_patches must be a positive integer")
    per_rank = bank_patches // context.world_size + int(context.rank < bank_patches % context.world_size)
    bank, source = _collect_cell_features(codec, loader, per_rank, context, paired=True)
    val, _ = _collect_cell_features(codec, val_loader, len(val_loader.dataset), context)

    def fit():
        payload = fit_feature_atlas(bank, val, cfg["atlas"], cfg["train"]["seed"], source,
                                    scale_floor=options["scale_floor"])
        payload["kind"] = "haar_mls"
        payload["diagnostics"]["descriptor"] = "haar_2x2x2_cell_means"
        return payload

    return context.main_call(fit, broadcast_result=True)
