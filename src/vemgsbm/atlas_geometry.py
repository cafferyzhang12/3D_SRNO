"""MRI bridge costs from the shared MLS chart atlas in learned 3D features."""

import hashlib
import json
import math

import numpy as np
import torch
from torch import nn

from .metric import Representation


def _routing_options(value):
    if value is False or value is None:
        return None
    if value is not True and not isinstance(value, dict):
        raise ValueError("source_routing must be a bool or a mapping")
    settings = dict(candidates=3, bandwidth=None, temperature=1., normal_weight=1.,
                    metric_floor=.05, jvp_step=.01, correction_limit=1.)
    supplied = {} if value is True else dict(value)
    if set(supplied) - set(settings):
        raise ValueError(f"Unknown source_routing options: {sorted(set(supplied) - set(settings))}")
    settings.update(supplied)
    if isinstance(settings["candidates"], bool) or int(settings["candidates"]) != settings["candidates"] or settings["candidates"] < 1:
        raise ValueError("source_routing candidates must be a positive integer")
    settings["candidates"] = int(settings["candidates"])
    for key in ("temperature", "normal_weight", "metric_floor", "jvp_step", "correction_limit"):
        value = float(settings[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"source_routing {key} must be finite and positive")
        settings[key] = value
    if settings["bandwidth"] is not None:
        settings["bandwidth"] = float(settings["bandwidth"])
        if not math.isfinite(settings["bandwidth"]) or settings["bandwidth"] <= 0:
            raise ValueError("source_routing bandwidth must be finite and positive")
    return settings


@torch.no_grad()
def _fit_source_routing(atlas, target_features, source_features, settings, scale_floor=.05):
    """Kernel regression of paired TRAIN LR features onto overlapping HR charts.

    Both banks must contain the same TRAIN patches in the same order. No VAL
    or TEST LR/HR features enter source statistics, bandwidth, or chart labels.
    """
    from scipy.spatial import cKDTree

    source = torch.as_tensor(source_features).detach().float().cpu()
    target = torch.as_tensor(target_features).detach().float().cpu()
    if (source.ndim != 2 or source.shape != target.shape or not len(source)
            or not torch.isfinite(source).all() or not torch.isfinite(target).all()):
        raise ValueError("Source routing needs finite, aligned TRAIN LR and HR feature matrices")
    mean = source.mean(0)
    scale = source.std(0, unbiased=False).clamp_min(scale_floor)
    bank = (source - mean) / scale
    settings = dict(settings)
    if settings["bandwidth"] is None:
        if len(bank) == 1:
            bandwidth = 1.
        else:
            # Mean-square feature distance has units independent of feature_dim.
            nearest = cKDTree(bank.numpy()).query(bank.numpy(), k=min(9, len(bank)))[0]
            bandwidth = float(np.median(nearest[:, -1]) / math.sqrt(bank.shape[1]))
        settings["bandwidth"] = max(bandwidth, .05)
    count = min(settings["candidates"], atlas.chart_count)
    location = atlas.locate(target, top_k=count, candidate_count=max(count, 8))
    logits = -.5 * location["normalized_residual"].square()
    # Chart fits outside their measured support receive less label mass.
    logits = logits + torch.where(location["in_domain"], 0., math.log(.05))
    labels = torch.zeros(len(source), atlas.chart_count)
    labels.scatter_(1, location["chart_ids"], logits.softmax(-1))
    fingerprint = hashlib.sha256(source.numpy().tobytes() + labels.numpy().tobytes()
                                 + json.dumps(settings, sort_keys=True).encode()).hexdigest()
    return {"version": 1, "kind": "paired_lr_kernel", "settings": settings, "sha256": fingerprint,
            "source_mean": mean, "source_scale": scale, "source_bank": bank,
            "chart_probabilities": labels}


def fit_mri_atlas(encoder, train_features, val_features, options, seed, source_features=None):
    payload = fit_feature_atlas(train_features, val_features, options, seed, source_features)
    payload["encoder_config"] = encoder.export_config()
    payload["encoder"] = {name: tensor.detach().cpu().clone() for name, tensor in encoder.state_dict().items()}
    payload["diagnostics"]["encoder_parameters"] = sum(p.numel() for p in encoder.parameters())
    return payload


def fit_feature_atlas(train_features, val_features, options, seed, source_features=None, *, scale_floor=.05):
    """Fit the shared MLS geometry and paired LR router to explicit feature rows."""
    from neural_atlas.atlas import build_atlas
    from neural_atlas.data import GeometryRows, Observations, PreparedData, fit_transform
    from neural_atlas.mls_atlas import fit_mls_atlas

    routing_settings = _routing_options(options.get("source_routing", source_features is not None))
    if routing_settings is not None and source_features is None:
        raise ValueError("Source-conditioned MLS needs aligned TRAIN LR source_features")
    if routing_settings is not None and (source_features.shape != train_features.shape
                                         or not torch.isfinite(source_features).all()):
        raise ValueError("Source routing needs finite, aligned TRAIN LR and HR feature matrices")
    values = torch.cat((train_features, val_features)).float().cpu().numpy()
    train = np.arange(len(train_features))
    val = np.arange(len(train), len(values))
    if not np.isfinite(values).all() or not len(val):
        raise ValueError("MRI atlas needs finite TRAIN and held-out VAL features")
    mask = Observations(values)
    transform = fit_transform(values, mask, train, "feature", "MRI patch features")
    transform.scale = np.maximum(transform.scale, scale_floor)
    data = PreparedData(
        x=GeometryRows(values, mask, transform), mask=mask, train=train, val=val,
        test=np.empty(0, dtype=int), transform=transform, metadata={}, truth={},
        meta={"dataset": "paired_mri_png"}, fingerprint=hashlib.sha256(values[train].tobytes()).hexdigest(),
    )
    meta = build_atlas(data, dict(options, seed=seed))
    atlas = fit_mls_atlas(data, meta, allow_unsupported=True, **options["mls"])
    held_out = torch.from_numpy(data.x[val])
    location = atlas.locate(held_out, top_k=1, candidate_count=4)
    snapshot = atlas.snapshot()
    # Keep the portable snapshot tensor-only for weights_only checkpoint loading.
    snapshot["meta"] = {
        key: torch.from_numpy(atlas.meta[key].copy())
        for key in ("centers", "basis", "scales", "domain_lo", "domain_hi")
    } | {"version": atlas.meta["version"]}
    payload = {
        "kind": "mls",
        "atlas": snapshot,
        "diagnostics": {
            "train_features": len(train), "validation_features": len(val),
            "chart_count": atlas.chart_count, "chart_dimension": atlas.d,
            "unsupported_charts": len(meta["centers"]) - atlas.chart_count,
            "active_anchors": int(atlas.anchor_valid.sum()),
            "atlas_version": atlas.meta["version"],
            "val_atlas_rmse": float(location["residual"].square().mean().sqrt()),
            "val_atlas_coverage": float(location["coverage"].float().mean()),
            "dimension_selection": meta["fit_metadata"]["dimension_selection"],
        },
    }
    if routing_settings is not None:
        payload["routing"] = _fit_source_routing(
            atlas, torch.from_numpy(data.x[train]), source_features, routing_settings, scale_floor=scale_floor)
        payload["diagnostics"].update(source_routing="paired_lr_kernel",
                                      source_routing_train_pairs=len(train),
                                      source_routing_version=payload["routing"]["sha256"],
                                      source_routing_bandwidth=payload["routing"]["settings"]["bandwidth"])
    return payload


def route_features(source, bank, chart_probabilities, settings):
    """Select chart candidates from TRAIN-paired LR descriptor likelihoods."""
    distance = torch.cdist(source, bank).square() / source.shape[-1]
    bandwidth = settings["bandwidth"]
    weights = (-distance / (2 * bandwidth**2 * settings["temperature"])).softmax(-1)
    probability = weights @ chart_probabilities
    count = min(settings["candidates"], chart_probabilities.shape[1])
    prior, ids = probability.topk(count, dim=-1)
    prior = prior / prior.sum(-1, keepdim=True).clamp_min(1e-12)
    confidence = (1 + distance.amin(-1) / bandwidth**2).reciprocal()
    return ids, prior, confidence


class MRIAtlasGeometry(nn.Module):
    def __init__(self, payload):
        super().__init__()
        from neural_atlas.mls_atlas import MLSAtlas

        self.encoder = Representation(**payload["encoder_config"])
        self.encoder.load_state_dict(payload["encoder"])
        self.atlas = MLSAtlas.from_checkpoint(payload["atlas"])
        transform = payload["atlas"]["transform"]
        self.register_buffer("feature_mean", torch.tensor(transform["mean"]))
        self.register_buffer("feature_scale", torch.tensor(transform["scale"]))
        self.source_routing = "routing" in payload
        if self.source_routing:
            routing = payload["routing"]
            if routing.get("version") != 1 or routing.get("kind") != "paired_lr_kernel":
                raise ValueError("Unsupported MRI source routing snapshot")
            self.routing_settings = _routing_options(routing["settings"])
            if self.routing_settings["bandwidth"] is None:
                raise ValueError("Fitted source routing must include a positive bandwidth")
            for key in ("source_mean", "source_scale", "source_bank", "chart_probabilities"):
                self.register_buffer(key, torch.as_tensor(routing[key]).float().clone())
            if (self.source_bank.ndim != 2 or not len(self.source_bank)
                    or self.source_bank.shape[1] != self.feature_mean.numel()
                    or self.source_mean.shape != self.feature_mean.shape
                    or self.source_scale.shape != self.feature_scale.shape
                    or self.chart_probabilities.shape != (len(self.source_bank), self.atlas.chart_count)
                    or not all(torch.isfinite(getattr(self, key)).all() for key in
                               ("source_mean", "source_scale", "source_bank", "chart_probabilities"))
                    or (self.source_scale <= 0).any() or (self.chart_probabilities < 0).any()
                    or not torch.allclose(self.chart_probabilities.sum(-1), torch.ones(len(self.source_bank)), atol=1e-5)):
                raise ValueError("Invalid source routing bank or chart probabilities")
        self.eval().requires_grad_(False)

    def train(self, mode=True):
        # Frozen geometry remains deterministic while image gradients are enabled.
        return super().train(False)

    def encode(self, images):
        return (self.encoder(images) - self.feature_mean) / self.feature_scale

    @torch.no_grad()
    def bind(self, target):
        if self.source_routing:
            source = (self.encoder(target) - self.source_mean) / self.source_scale
            if not torch.isfinite(source).all():
                raise ValueError("Source routing received nonfinite LR features")
            ids, prior, confidence = route_features(source, self.source_bank, self.chart_probabilities,
                                                    self.routing_settings)
            return BoundSourceMRIAtlas(self, ids, prior, confidence)
        features = self.encode(target)
        location = self.atlas.locate(features, top_k=1, candidate_count=4)
        return BoundMRIAtlas(self, location["chart_ids"][:, 0])


class BoundMRIAtlas:
    """Keep the target chart fixed during each paired path optimization."""

    def __init__(self, geometry, chart_ids):
        self.geometry = geometry
        self.chart_ids = chart_ids

    def _coordinates(self, features):
        atlas, ids = self.geometry.atlas, self.chart_ids
        coordinate = atlas.encode_local(ids, features)
        coordinate = coordinate.clamp(atlas.domain_lo[ids], atlas.domain_hi[ids])
        anchors, radii = atlas.anchor_z[ids], atlas.bandwidth[ids]
        relative = (coordinate[:, None] - anchors) / radii[..., None]
        distances = relative.square().sum(-1).masked_fill(~atlas.anchor_valid[ids], torch.inf)
        nearest = distances.argmin(-1)
        rows = torch.arange(len(ids), device=ids.device)
        center = anchors[rows, nearest]
        radius = radii[rows, nearest] * atlas.settings["support_limit"] * atlas.settings["blend_radius"]
        factor = (radius / (coordinate - center).norm(dim=-1).clamp_min(1e-7)).clamp(max=1)
        return center + (coordinate - center) * factor[:, None]

    def _state_energy(self, features, decoded):
        # A smooth robust loss retains gradients outside the fitted chart.
        error = (features - decoded).square().mean(-1)
        return 2 * (error / (1 + (1 + error).sqrt()))

    def _normal_energy(self, midpoint, velocity):
        from neural_atlas.geometry import tangent_projection

        atlas, ids = self.geometry.atlas, self.chart_ids
        coordinate = self._coordinates(midpoint)
        decoded, jacobian = atlas.decode_and_jacobian(ids, coordinate)
        normal = velocity - tangent_projection(jacobian, velocity)
        # Trust the tangent less as the path departs from the fitted surface.
        reliability = (1 + (midpoint - decoded).square().sum(-1) /
                       atlas.scales[ids].square()).reciprocal()
        return reliability * normal.square().mean(-1)

    def state_cost(self, images):
        features = self.geometry.encode(images)
        decoded = self.geometry.atlas.decode(self.chart_ids, self._coordinates(features))
        return self._state_energy(features, decoded)

    def kinetic(self, first, second, dt):
        a, b = self.geometry.encode(torch.cat((first, second))).chunk(2)
        return self._normal_energy((a + b) / 2, (b - a) / dt)

    def phase_cost(self, images, image_velocity):
        # Central difference applies the feature-map Jacobian to image velocity.
        step = 0.01
        midpoint, left, right = self.geometry.encode(torch.cat(
            (images, images - step * image_velocity, images + step * image_velocity)
        )).chunk(3)
        decoded = self.geometry.atlas.decode(self.chart_ids, self._coordinates(midpoint))
        return (self._state_energy(midpoint, decoded).mean(),
                self._normal_energy(midpoint, (right - left) / (2 * step)).mean())

    def momentum_cost(self, base, codec):
        def cost(position, velocity, free):
            return self.phase_cost(base + codec.decode(position * free), codec.decode(velocity * free))
        return cost


def trusted_mls_decode(atlas, ids, coordinate, correction_limit, derivative=True):
    """MLS normal graph with a bounded, support-weighted nonlinear part."""
    decoded, jacobian = (atlas.decode_and_jacobian(ids, coordinate) if derivative
                         else (atlas.decode(ids, coordinate), None))
    basis = atlas.basis[ids] * atlas.scales[ids, None, None]
    linear = atlas.centers[ids] + torch.einsum("mdi,mi->md", basis, coordinate)
    normal = decoded - linear
    limit = correction_limit * atlas.scales[ids, None]
    shrink = (1 + normal.square().sum(-1, keepdim=True) / limit.square()).rsqrt()
    bounded = normal * shrink
    delta = coordinate[:, None] - atlas.anchor_z[ids]
    radius = atlas.bandwidth[ids] * atlas.settings["blend_radius"]
    activity = .95 * torch.exp(-(delta / radius[..., None]).square().sum(-1))
    activity = activity * atlas.anchor_valid[ids]
    product = (1 - activity).prod(-1)
    support = 1 - product
    trusted = linear + support[:, None] * bounded
    trusted_jacobian = None
    if derivative:
        normal_jacobian = jacobian - basis
        bounded_jacobian = normal_jacobian * shrink[..., None] - (
            normal[..., None] * shrink.pow(3)[..., None]
            * torch.einsum("md,mdi->mi", normal, normal_jacobian)[:, None, :]
            / limit.square()[..., None])
        support_derivative = product[:, None] * (
            -2 * delta / radius.square()[..., None]
            * (activity / (1 - activity))[..., None]).sum(1)
        trusted_jacobian = (basis + support[:, None, None] * bounded_jacobian
                            + bounded[..., None] * support_derivative[:, None, :])
    return trusted, trusted_jacobian, support


class BoundSourceMRIAtlas:
    """LR-selected overlapping charts, fixed for a complete training/sampling path.

    The MLS decoder has a bounded, support-weighted normal correction. Its
    analytic Jacobian includes both modifications. Chart-coordinate velocities
    use the induced Gram matrix J^T J, plus a soft ambient normal penalty.
    This is feature geometry; the compressive encoder does not define a
    nondegenerate Riemannian metric on the entire image space.
    """

    def __init__(self, geometry, chart_ids, chart_prior, source_confidence):
        self.geometry, self.chart_ids = geometry, chart_ids
        self.chart_prior, self.source_confidence = chart_prior, source_confidence
        self.settings = geometry.routing_settings

    def _trusted_decode(self, ids, coordinate, derivative=True):
        return trusted_mls_decode(self.geometry.atlas, ids, coordinate, self.settings["correction_limit"], derivative)

    def _local_geometry(self, features, derivative=True):
        atlas, ids = self.geometry.atlas, self.chart_ids
        if features.ndim != 2 or len(features) != len(ids):
            raise ValueError("Bound MRI geometry requires the bound source batch size")
        batch, candidates = ids.shape
        flat_ids = ids.flatten()
        expanded = features[:, None].expand(-1, candidates, -1).reshape(-1, features.shape[-1])
        raw = atlas.encode_local(flat_ids, expanded)
        center = (atlas.domain_hi[flat_ids] + atlas.domain_lo[flat_ids]) / 2
        halfwidth = ((atlas.domain_hi[flat_ids] - atlas.domain_lo[flat_ids]) / 2).clamp_min(1e-4)
        relative = ((raw - center) / halfwidth).tanh()
        coordinate = center + halfwidth * relative
        decoded, jacobian, support = self._trusted_decode(flat_ids, coordinate, derivative=derivative)
        residual = (expanded - decoded).square().mean(-1)
        reliability = (1 + residual / atlas.scales[flat_ids].square()).reciprocal()
        trust = (support * reliability).reshape(batch, candidates)
        weight = self.chart_prior * (.01 + trust)
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-12)
        return {
            "residual": residual.reshape(batch, candidates), "weight": weight,
            "trust": trust * self.source_confidence[:, None],
            "jacobian": (jacobian.reshape(batch, candidates, atlas.ambient_dim, atlas.d)
                         if derivative else None),
        }

    def _state_energy(self, local):
        residual = local["residual"]
        robust = 2 * residual / (1 + (1 + residual).sqrt())
        # Retain attraction and its input gradient even outside anchor support.
        # Low confidence reduces strength without creating a detached zero cost.
        return (local["weight"] * robust).sum(-1) * (.05 + .95 * self.source_confidence)

    def _direction_energy(self, local, velocity):
        jacobian = local["jacobian"]
        gram = jacobian.transpose(-2, -1) @ jacobian
        # Recover tangent coefficients in the chart's induced metric. A tiny
        # scale-relative ridge makes the projection stable; solve_ex avoids a
        # CUDA synchronization for an error check on every path evaluation.
        ridge = gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12) * 1e-6
        system = gram + ridge[..., None, None] * torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        rhs = torch.einsum("bkdi,bd->bki", jacobian, velocity)
        coordinate_velocity = torch.linalg.solve_ex(system, rhs[..., None], check_errors=False).result[..., 0]
        tangent = torch.einsum("bki,bkij,bkj->bk", coordinate_velocity, gram, coordinate_velocity)
        lifted = torch.einsum("bkdi,bki->bkd", jacobian, coordinate_velocity)
        normal = (velocity[:, None] - lifted).square().sum(-1)
        chart_energy = (tangent + self.settings["normal_weight"] * normal) / velocity.shape[-1]
        ambient = velocity.square().mean(-1)
        # Unsupported charts revert smoothly to the ambient feature metric.
        trusted = local["trust"] * chart_energy + (1 - local["trust"]) * ambient[:, None]
        return ((local["weight"] * trusted).sum(-1)
                + self.settings["metric_floor"] * ambient)

    def state_cost(self, images):
        return self._state_energy(self._local_geometry(self.geometry.encode(images), derivative=False))

    def kinetic(self, first, second, dt):
        a, b = self.geometry.encode(torch.cat((first, second))).chunk(2)
        delta = torch.as_tensor(dt, dtype=a.dtype, device=a.device)
        if delta.ndim > 1 or (delta.ndim and delta.shape != (len(a),)) or not torch.isfinite(delta).all() or (delta <= 0).any():
            raise ValueError("dt must be finite and positive, scalar or shape [B]")
        if delta.ndim:
            delta = delta[:, None]
        return self._direction_energy(self._local_geometry((a + b) / 2), (b - a) / delta)

    def phase_cost(self, images, image_velocity):
        step = self.settings["jvp_step"]
        midpoint, left, right = self.geometry.encode(torch.cat(
            (images, images - step * image_velocity, images + step * image_velocity)
        )).chunk(3)
        local = self._local_geometry(midpoint)
        return self._state_energy(local).mean(), self._direction_energy(local, (right - left) / (2 * step)).mean()

    def drift_cost(self, state, error):
        return self.drift_costs(state, error)[0]

    def drift_costs(self, state, *errors):
        if not errors or any(error.shape != state.shape for error in errors):
            raise ValueError("drift_costs requires image errors with the state shape")
        # The learned matching target and metric location are fixed. Gradients
        # flow through each error JVP, never into state or frozen atlas weights.
        fixed_state = state.detach()
        with torch.no_grad():
            local = self._local_geometry(self.geometry.encode(fixed_state))
        step, costs = self.settings["jvp_step"], []
        for error in errors:
            left, right = self.geometry.encode(torch.cat(
                (fixed_state - step * error, fixed_state + step * error))).chunk(2)
            costs.append(self._direction_energy(local, (right - left) / (2 * step)))
        return tuple(costs)

    def momentum_cost(self, base, codec):
        def cost(position, velocity, free):
            return self.phase_cost(base + codec.decode(position * free), codec.decode(velocity * free))
        return cost


@torch.no_grad()
def collect_features(encoder, loader, limit, context, paired=False):
    encoder.eval()
    parts, source_parts, remaining, error = [], [], limit, None
    try:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("Feature limit must be a nonnegative integer")
        for batch in loader:
            if not remaining:
                break
            features = encoder(batch["hr"][:remaining].to(context.device))
            parts.append(features.cpu())
            if paired:
                source = encoder(batch["base"][:remaining].to(context.device))
                if source.shape != features.shape:
                    raise ValueError("TRAIN HR/source patch batches must be aligned")
                source_parts.append(source.cpu())
            remaining -= len(features)
            if not remaining:
                break
        local = torch.cat(parts) if parts else torch.empty(0, encoder.feature_dim)
        if paired:
            source = torch.cat(source_parts) if source_parts else torch.empty(0, encoder.feature_dim)
            local = (local, source)
    except Exception as exc:
        error = f"MRI atlas feature extraction: {exc}"
    context.check_errors(error)
    if not context.is_distributed:
        return local
    gathered = [None] * context.world_size
    torch.distributed.all_gather_object(gathered, local)
    return tuple(torch.cat([pair[index] for pair in gathered]) for index in range(2)) if paired else torch.cat(gathered)


def load_mri_atlas(cfg, digest, identity, device):
    from .runtime import load_checkpoint, project_path

    saved = load_checkpoint(project_path(cfg["representation"]["checkpoint"]))
    payload = saved.get("metric")
    if saved.get("stage") != "representation" or not payload or payload.get("kind") != "mls":
        raise ValueError("MLS bridge requires a completed MRI representation and chart fit")
    if (saved["manifest_sha256"] != digest or saved["data_identity"] != identity
            or saved["config"]["data"] != cfg["data"] or saved["config"]["atlas"] != cfg["atlas"]):
        raise ValueError("MRI atlas and bridge input contracts differ")
    requested_routing = _routing_options(cfg["atlas"].get("source_routing")) is not None
    if requested_routing != ("routing" in payload):
        raise ValueError("MRI atlas source routing payload and configuration differ")
    return payload, MRIAtlasGeometry(payload).to(device)
