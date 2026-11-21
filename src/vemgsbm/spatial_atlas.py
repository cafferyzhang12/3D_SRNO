"""Cross-fitted local residual graphs on the native LR grid of XYx4 MRI."""
import argparse
import copy
import hashlib

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn

from .haar_residual import LRConditionEncoder
from .momentum.model import HaarXY
from .runtime import atomic_save, fingerprint, load_checkpoint, setup


def _sites(x):
    return x.movedim(1, -1).reshape(-1, x.shape[1])


def _volume(x, shape):
    return x.reshape(shape[0], *shape[2:], -1).movedim(-1, 1)


def _balanced_sites(lr, budget, generator):
    plane = lr[0]
    edge = torch.zeros_like(plane)
    edge[:, 1:] += (plane[:, 1:] - plane[:, :-1]).abs()
    edge[:, :, 1:] += (plane[:, :, 1:] - plane[:, :, :-1]).abs()
    foreground = plane > 2 / 255
    border = (edge > 0) & foreground
    if border.any():
        border &= edge >= torch.quantile(edge[border], .75)
    groups = [(mask.flatten().nonzero()[:, 0]) for mask in
              (~foreground, foreground & ~border, border)]
    groups = [ids[torch.randperm(len(ids), generator=generator)] for ids in groups if len(ids)]
    # Interleave strata. A small or absent stratum gives its unused budget to the others.
    ranks = torch.cat([torch.arange(len(ids)) for ids in groups])
    return torch.cat(groups)[ranks.argsort(stable=True)[:budget]]


@torch.no_grad()
def build_spatial_atlas(direct_ckpt, train_set, settings, device):
    if train_set.split != "train" or direct_ckpt["config"]["mode"] != "haar_direct":
        raise ValueError("Build the spatial atlas from TRAIN and a direct-regression checkpoint")
    cfg = direct_ckpt["config"]
    if train_set.scale != 4 or tuple(cfg["data"]["patch_size"]) != train_set.patch_size:
        raise ValueError("Spatial atlas requires matching direct model and XYx4 patches")
    nsites = int(settings.get("sites_per_case", 256))
    npatches = int(settings.get("patches_per_case", 16))
    if min(nsites, npatches) < 1:
        raise ValueError("Positive spatial atlas sampling budgets required")
    ds = copy.copy(train_set)
    ds.samples_per_epoch, ds.samples_per_volume = None, min(npatches, nsites)
    ds.seed = int(settings.get("seed", 42))
    ds.set_epoch(0)
    widths = cfg["model"]["widths"]
    encoder = LRConditionEncoder(widths).to(device).eval()
    prefix = "lr_encoder."
    weights = {k[len(prefix):]: v.cpu() for k, v in direct_ckpt["model"].items() if k.startswith(prefix)}
    encoder.load_state_dict(weights)
    codec = HaarXY().to(device)
    gen = torch.Generator().manual_seed(ds.seed)
    descriptors, residuals, case_indices = [], [], []
    for idx in range(len(ds)):
        item = ds[idx]
        slot = idx % ds.samples_per_volume
        quota = nsites // ds.samples_per_volume + (slot < nsites % ds.samples_per_volume)
        sites = _balanced_sites(item["lr"], quota, gen)
        lr = item["lr"][None].to(device)
        descriptor = encoder(lr, descriptor_only=True)
        residual = codec.encode((item["hr"] - item["base"])[None].to(device))
        descriptors.append(_sites(descriptor)[sites].cpu())
        residuals.append(_sites(residual)[sites].cpu())
        case_indices.append(torch.full((len(sites),), idx // ds.samples_per_volume, dtype=torch.int32))
    return {"version": 1, "descriptors": torch.cat(descriptors), "residuals": torch.cat(residuals),
            "case_indices": torch.cat(case_indices), "widths": widths, "encoder": weights,
            "train_case_ids": [c.id for c in ds.cases], "train_fingerprints": [c.fingerprint for c in ds.cases],
            "sampling": {"seed": ds.seed, "sites_per_case": nsites, "patches_per_case": npatches}}


def _fit_charts(source, residuals, settings, dim, generator):
    from neural_atlas.mls_atlas import polynomial_powers

    avg, scale = source.mean(0), source.std(0, unbiased=False).clamp_min(1e-5)
    source = (source - avg) / scale
    ncenters = min(len(source), int(settings.get("centers", 128)))
    k = min(int(settings.get("neighbors", 64)), len(source))
    if min(ncenters, k) < 1:
        raise ValueError("Spatial atlas needs positive neighborhoods and centers")
    centers = source[torch.randperm(len(source), generator=generator)[:ncenters]]
    distances, ids = cKDTree(source.numpy()).query(centers.numpy(), k=np.arange(k) + 1)
    dist = torch.from_numpy(distances)
    radius = dist[:, -1:].clamp_min(.01)
    w = torch.softmax(-2 * (dist / radius).square(), dim=1)
    effective = w.square().sum(-1).reciprocal()
    ambient = residuals[ids].double()
    mean = (w[..., None] * ambient).sum(1)
    centered = ambient - mean[:, None]
    covariance = centered.mT @ (w[..., None] * centered)
    eigenvalues, directions = torch.linalg.eigh(covariance)
    basis = directions[:, :, -dim:].flip(-1)
    q = centered @ basis
    bandwidth = (eigenvalues[:, -dim:].flip(-1).clamp_min(1e-8)).sqrt()
    qn = q / bandwidth[:, None]
    # Same monomial ordering as neural_atlas. Store the quadratic coefficients
    # as a symmetric form for cheap analytic values and Jacobians at all sites.
    powers = polynomial_powers(dim, 2)
    pairs = np.array([np.repeat(np.arange(dim), p) for p in powers if p.sum() == 2])
    i, j = pairs.T
    matrix = torch.cat((torch.ones_like(qn[..., :1]), qn, qn[..., i] * qn[..., j]), dim=-1)
    design = matrix * w.sqrt()[..., None]
    normal = centered - q @ basis.mT
    u, s, vt = torch.linalg.svd(design, full_matrices=False)
    condition = s[:, 0] / s[:, -1].clamp_min(1e-12)
    ridge = float(settings.get("ridge", 1.e-4))
    if ridge <= 0:
        raise ValueError("Quadratic fit requires positive ridge")
    pca_supported = (effective > dim) & (eigenvalues[:, -dim] > 1.e-8)
    enough = ((effective >= matrix.shape[-1]) & (condition <= settings.get("condition_limit", 1.e6))
              & pca_supported)
    shrink = s / (s.square() + ridge * s[:, :1].square())
    coeff = (vt.mT * shrink[:, None, :]) @ (u.mT @ (normal * w.sqrt()[..., None]))
    coeff *= enough[:, None, None]
    form = mean.new_zeros(ncenters, dim, dim, 16)
    form[:, i, j] += .5 * coeff[:, 1+dim:]
    form[:, j, i] += .5 * coeff[:, 1+dim:]
    # Fit a TRAIN-only affine LR-to-coordinate estimator for the reference
    # diffusion. No neural bridge state enters this estimate.
    local_desc = source[ids].double()
    desc_mean = (local_desc * w[..., None]).sum(1)
    x = local_desc - desc_mean[:, None]
    gram = x.mT @ (x * w[..., None])
    identity = torch.eye(gram.shape[-1], dtype=gram.dtype)
    lr_to_q = torch.linalg.solve(gram + ridge * identity, x.mT @ (qn * w[..., None]))
    fitted = dict(centers=centers, radius=radius, mean=mean, basis=basis, bandwidth=bandwidth,
                  offset=coeff[:, 0], linear=coeff[:, 1:1+dim], form=form, covariance=covariance,
                  desc_mean=desc_mean, lr_to_q=lr_to_q, effective=effective, condition=condition,
                  quadratic_supported=enough, pca_supported=pca_supported)
    return avg, scale, {k: v.float() for k, v in fitted.items()}


class SpatialAtlas(nn.Module):
    def __init__(self, payload, settings):
        super().__init__()
        if payload.get("version") != 1:
            raise ValueError("Unknown spatial MRI atlas version")
        self.settings = dict(settings)
        self.method = settings.get("method", "mls")
        self.dim = int(settings.get("dim", 4))
        self.reference = settings.get("reference", "identity")
        if self.method not in {"pca", "mls", "mean"} or self.dim not in (2, 4, 6, 8):
            raise ValueError("Invalid spatial geometry method or dimension")
        if self.reference not in ("identity", "geometry", "shuffled"):
            raise ValueError("Unknown spatial reference")
        self.encoder = LRConditionEncoder(payload["widths"])
        self.encoder.load_state_dict(payload["encoder"])
        ids = payload["train_case_ids"]
        nf = min(len(ids), int(settings.get("folds", 5)))
        if nf < 2 or len(ids) != len(set(ids)):
            raise ValueError("Cross-fitted atlas needs two or more distinct TRAIN cases")
        seed = int(settings.get("seed", 42))
        ordered = sorted(ids, key=lambda case: hashlib.sha256(f"{seed}:{case}".encode()).digest())
        self.folds = {case: index % nf for index, case in enumerate(ordered)}
        source = payload["descriptors"].float().cpu()
        target = payload["residuals"].float().cpu()
        row_folds = torch.tensor([self.folds[case] for case in ids])[payload["case_indices"].long()]
        if source.ndim != 2 or target.shape != (len(source),16) or not (source.isfinite().all() and target.isfinite().all()):
            raise ValueError("Invalid TRAIN descriptor/residual records")
        chunks, avgs, scales, offsets = {}, [], [], [0]
        for fold in range(nf + 1):
            mask = row_folds != fold
            gen = torch.Generator().manual_seed(seed + fold)
            avg, scale, fit = _fit_charts(source[mask], target[mask], settings, self.dim, gen)
            offsets.append(offsets[-1] + len(fit["centers"]))
            if settings.get("shuffle_routing", False):
                # Derange fitted surfaces while keeping actual LR centers, radii
                # and probabilities. A cyclic shift avoids accidental fixed points.
                order = torch.randperm(len(fit["mean"]), generator=gen)
                perm = torch.empty_like(order)
                perm[order] = order.roll(1)
                fit = {k: v if k in {"centers","radius"} else v[perm] for k,v in fit.items()}
            for k, v in fit.items():
                chunks.setdefault(k, []).append(v)
            avgs.append(avg)
            scales.append(scale)
        self.offsets = offsets
        for key, parts in chunks.items():
            self.register_buffer(key, torch.cat(parts))
        self.register_buffer("desc_avg", torch.stack(avgs))
        self.register_buffer("desc_scale", torch.stack(scales))
        random_matrix = torch.randn(16, 16, generator=torch.Generator().manual_seed(seed + 12001))
        self.register_buffer("rotation", torch.linalg.qr(random_matrix).Q)
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def bind(self, lr, exclude_cases=None):
        n = len(lr)
        if exclude_cases is not None and (len(exclude_cases) != n or any(case not in self.folds for case in exclude_cases)):
            raise ValueError("TRAIN queries must identify each member of the cross-fitted case bank")
        features = self.encoder(lr, descriptor_only=True)
        chunk = int(self.settings.get("query_chunk", 2048))
        selected, weights, trusts, factors = [], [], [], []
        # Folds can contain different numbers of charts on tiny training sets.
        top = min(self.settings.get("candidates", 3), min(np.diff(self.offsets)))
        if chunk < 1 or top < 1:
            raise ValueError("Positive retrieval chunk and candidate counts required")
        for b in range(n):
            fold = len(self.offsets) - 2 if exclude_cases is None else self.folds[exclude_cases[b]]
            start, end = self.offsets[fold:fold+2]
            for site in _sites(features[b:b+1]).split(chunk):
                x = (site - self.desc_avg[fold]) / self.desc_scale[fold]
                distance, ix = torch.cdist(x, self.centers[start:end]).topk(top, largest=False)
                ids = ix + start
                d2 = (distance / self.radius[ids, 0]).square()
                prior = (-.5 * d2).softmax(1)
                trust = (1 + d2).reciprocal()
                selected.append(ids)
                weights.append(prior)
                trusts.append(trust)
                if self.reference != "identity":
                    qn = ((x[:, None] - self.desc_mean[ids]).unsqueeze(-2) @ self.lr_to_q[ids]).squeeze(-2)
                    _, jac = self._decode(ids, qn.clamp(-3, 3))
                    factors.append(self._factor(ids, prior, trust, jac))
        return BoundSpatialAtlas(self, torch.cat(selected), torch.cat(weights), torch.cat(trusts),
                                 torch.cat(factors) if factors else None, lr.shape)

    def _decode(self, ids, qn):
        # Jacobian with respect to standardized tangent coordinates; its span
        # equals the physical-coordinate tangent space.
        basis = self.basis[ids] * self.bandwidth[ids][..., None, :]
        values = self.mean[ids]
        if self.method == "mean":
            return values, basis
        values = values + self.pca_supported[ids][..., None] * (basis @ qn[..., None]).squeeze(-1)
        if self.method == "pca":
            return values, basis
        form = self.form[ids]
        half_grad = torch.einsum("...i,...ijd->...jd", qn, form)
        linear = self.linear[ids]
        values = values + self.offset[ids] + ((linear + half_grad) * qn[..., None]).sum(-2)
        return values, basis + (linear + 2 * half_grad).mT

    def _factor(self, ids, prior, trust, jac):
        q = torch.linalg.qr(jac, mode="reduced").Q
        projection = q @ q.mT
        covariance = self.covariance[ids]
        tangential_trace = (projection * covariance).sum((-2, -1))
        total = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
        a = tangential_trace.clamp_min(0) / self.dim
        b = (total - tangential_trace).clamp_min(0) / (16 - self.dim)
        eye = torch.eye(16, device=ids.device)
        matrix = (1.e-6 * eye + (prior[..., None, None] *
                  (b[..., None, None] * eye + (a-b)[..., None, None] * projection)).sum(1))
        matrix = matrix / matrix.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None]
        # Low effective sample size, weak descriptor support and fallback fits
        # shrink the covariance continuously to identity while preserving trace.
        support = self._confidence(ids, prior, trust)
        eigen = torch.linalg.eigvalsh(matrix)
        upper = (3. / (eigen[:, -1] - 1).clamp_min(1.e-6)).clamp(max=1.)
        lower = (.9 / (1 - eigen[:, 0]).clamp_min(1.e-6)).clamp(max=1.)
        weight = torch.minimum(support, torch.minimum(upper, lower))
        matrix = eye + weight[:, None, None] * (matrix - eye)
        if self.reference == "shuffled":
            matrix = self.rotation @ matrix @ self.rotation.T
        return torch.linalg.cholesky((matrix + matrix.mT) / 2)

    def _confidence(self, ids, prior, trust):
        support = (self.effective[ids] / max(2*self.dim, 16)).clamp(max=1)
        if self.method != "mean":
            support = support * self.pca_supported[ids]
        if self.method == "mls":
            support = support * (.25 + .75 * self.quadratic_supported[ids])
        return (prior * trust * support).sum(1)


class BoundSpatialAtlas:
    def __init__(self, atlas, ids, weights, trust, factor, shape):
        self.atlas, self.ids, self.weights, self.trust = atlas, ids, weights, trust
        self.factor, self.shape = factor, shape

    def diagnostics(self):
        def average(buffer):
            return (buffer[self.ids] * self.weights).sum(1).mean()
        return {"effective_neighbors": average(self.atlas.effective),
                "quadratic_design_condition": average(self.atlas.condition),
                "quadratic_fallback_fraction": average(1 - self.atlas.quadratic_supported),
                "pca_to_mean_fallback_fraction": average(1 - self.atlas.pca_supported)}

    def whiten(self, r):
        if self.factor is None:
            return r
        z = torch.linalg.solve_triangular(self.factor, _sites(r).unsqueeze(-1), upper=False)
        return _volume(z.squeeze(-1), self.shape)

    def unwhiten(self, z):
        if self.factor is None:
            return z
        return _volume((self.factor @ _sites(z)[..., None]).squeeze(-1), self.shape)

    def _components(self, raw, velocity=None, mean_time=None):
        a = self.atlas
        chunk = int(a.settings.get("query_chunk", 2048))
        rows = _sites(raw)
        vrows = None if velocity is None else _sites(velocity)
        parts = []
        for start in range(0, len(rows), chunk):
            region = slice(start, start + chunk)
            ids, w = self.ids[region], self.weights[region]
            x = rows[region, None]
            coord = ((x - a.mean[ids]).unsqueeze(-2) @ a.basis[ids]).squeeze(-2) / a.bandwidth[ids]
            pred, jac = a._decode(ids, coord.clamp(-3,3))
            if mean_time is not None:
                state_error = ((x - pred) * mean_time).square().mean(-1)
                parts.append((w * state_error).sum(1, keepdim=True))
                continue
            vector = (x if vrows is None else vrows[region,None]) - pred
            if a.method == "mean":
                tangent = torch.zeros_like(vector)
            else:
                # Basis has full column rank. QR here has continuous, finite
                # derivatives for chart coordinates in the supported domain.
                q = torch.linalg.qr(jac, mode="reduced").Q
                tangent = (q @ (q.mT @ vector[...,None])).squeeze(-1) * a.pca_supported[ids][..., None]
            if velocity is None:
                confidence = a._confidence(ids, w, self.trust[region])
                mix = lambda values: (w[...,None] * values).sum(1)
                target = mix(pred)
                parts.append(torch.cat((target, target - x[:,0], mix(tangent),
                                        mix(vector-tangent), confidence[:,None]), dim=1))
            else:
                parts.append((w * (vector-tangent).square().mean(-1)).sum(1,keepdim=True))
        return _volume(torch.cat(parts), self.shape)

    def features(self, raw):
        return self._components(raw)

    def path_metric(self):
        return self

    def mean_state_cost(self, state, time):
        mean = self.unwhiten(state)
        return self._components(mean / time, mean_time=time).flatten(1).mean(1)

    def kinetic_time(self, first, second, dt, time):
        midpoint = self.unwhiten((first + second) / 2)
        velocity = self.unwhiten((second - first) / dt)
        # For t*g(q(t)), subtract g(q) before projecting the motion onto
        # the normal space. t*J(q)*q_dot lies wholly in the tangent.
        return self._components(midpoint / time, velocity).flatten(1).mean(1)


def load_spatial_atlas(payload, settings):
    return SpatialAtlas(payload, settings)


def bank_provenance(path, identity, settings):
    return {"train_identity": [row for row in identity if row["split"] == "train"],
            "descriptor_checkpoint_sha256": fingerprint(path),
            "fit_settings": {key: settings.get(key, default) for key, default in
                             (("seed", 42), ("sites_per_case", 256), ("patches_per_case", 16))}}


def main():
    from .datasets import training_data
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case-inventory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sites-per-case", type=int, default=256)
    parser.add_argument("--patches-per-case", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    saved = load_checkpoint(args.checkpoint)
    train, _, digest, identity = training_data(saved["config"], args.case_inventory)
    if digest != saved["manifest_sha256"]:
        raise ValueError("Direct checkpoint and bank inventory identities differ")
    payload = build_spatial_atlas(saved, train, vars(args), setup(args.seed, args.device))
    atomic_save({"provenance": bank_provenance(args.checkpoint, identity, vars(args)), "payload": payload}, args.output)
    print(f"Saved {len(payload['residuals'])} TRAIN sites from {len(train.cases)} cases", flush=True)


if __name__ == "__main__":
    main()
