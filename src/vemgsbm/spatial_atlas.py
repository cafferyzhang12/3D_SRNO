"""Cross-fitted local residual graphs on the native LR grid of XYx4 MRI."""
import argparse
import copy
import hashlib

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F

from .haar_residual import LRConditionEncoder
from .momentum.model import HaarXY
from .runtime import atomic_save, fingerprint, load_checkpoint, setup
from .jetatlas import SCJetAtlas, JetAtlasRegularizer, SoftAtlasRouter


def _sites(x):
    return x.movedim(1, -1).reshape(-1, x.shape[1])


def _volume(x, shape):
    return x.reshape(shape[0], *shape[2:], -1).movedim(-1, 1)


@torch.no_grad()
def _structure_labels(reference, grid_shape):
    """Detached TRAIN HR strata on the Haar grid: background/weak/texture/edge.

    Local variance uses a 7x7 XY window. No pooling or finite difference mixes
    slices. These training strata do not change the whole-case metric mask.
    """
    if (reference.ndim != 5 or reference.shape[:2] != (1, 1)
            or tuple(reference.shape[2:]) != (grid_shape[0], 4*grid_shape[1], 4*grid_shape[2])):
        raise ValueError("Structural MLS sampling needs aligned B1ZYX HR at four times the Haar XY grid")
    image = reference.detach().float().reshape(-1, 1, *reference.shape[-2:])
    if not torch.isfinite(image).all():
        raise ValueError("Structural MLS reference must be finite")
    mean = F.avg_pool2d(F.pad(image, (3, 3, 3, 3), mode="replicate"), 7, stride=1)
    second = F.avg_pool2d(F.pad(image.square(), (3, 3, 3, 3), mode="replicate"), 7, stride=1)
    variance = (49/48*(second-mean.square())).clamp_min(0)
    gradient = (F.pad((image[..., 1:, :]-image[..., :-1, :]).abs(), (0, 0, 0, 1))
                + F.pad((image[..., 1:]-image[..., :-1]).abs(), (0, 1, 0, 0)))
    foreground = F.avg_pool2d(image, 4, stride=4).flatten() > 2/255
    variance = F.avg_pool2d(variance, 4, stride=4).flatten()
    gradient = F.avg_pool2d(gradient, 4, stride=4).flatten()
    edge = torch.zeros_like(foreground)
    if foreground.any():
        edge = foreground & (gradient > 0) & (gradient >= torch.quantile(gradient[foreground], .9))
    labels = torch.zeros_like(foreground, dtype=torch.long)
    labels[foreground] = 2
    labels[foreground & (variance < 1.e-4)] = 1
    labels[edge] = 3
    return labels


@torch.no_grad()
def _stratified_sites(labels, budget, fractions):
    """Balanced training objective, intentionally not a uniform-loss estimator.

    Sample without replacement; reassign missing/undersized stratum quotas to
    remaining sites. Thus black patches and tiny patches always remain valid.
    """
    budget = min(budget, len(labels))
    if budget == len(labels):
        return torch.arange(len(labels), device=labels.device)
    exact = np.asarray(fractions)*budget
    quotas = np.floor(exact).astype(int)
    for index in np.argsort(-(exact-quotas), kind="stable")[:budget-int(quotas.sum())]:
        quotas[index] += 1
    selected = []
    used = torch.zeros_like(labels, dtype=torch.bool)
    for group, quota in enumerate(quotas):
        available = (labels == group).nonzero()[:, 0]
        ids = available[torch.randperm(len(available), device=labels.device)[:int(quota)]]
        used[ids] = True
        selected.append(ids)
    missing = budget-sum(len(ids) for ids in selected)
    if missing:
        remaining = (~used).nonzero()[:, 0]
        selected.append(remaining[torch.randperm(len(remaining), device=labels.device)[:missing]])
    return torch.cat(selected)


def _xy_neighbor_pairs(sites, grid_shape):
    """Positive XY neighbors of anchors, never a row wrap or a Z neighbor."""
    _, height, width = grid_shape
    right = sites[sites.remainder(width) < width-1]
    down = sites[sites.div(width, rounding_mode="floor").remainder(height) < height-1]
    return torch.cat((right, down)), torch.cat((right+1, down+width))


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


def _fit_charts(source, residuals, settings, dim, generator, source_weights=None):
    avg, scale = source.mean(0), source.std(0, unbiased=False).clamp_min(1e-5)
    if source_weights is not None:
        # Apply descriptor weights after standardization so normalization does
        # not silently cancel the relative weight of the paired LR features.
        scale = scale / source_weights
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
    i, j = torch.triu_indices(dim, dim)
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
                  quadratic_supported=enough, pca_supported=pca_supported,
                  domain_lo=qn.amin(1), domain_hi=qn.amax(1))
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


@torch.no_grad()
def prepare_training_mls(dataset, model, settings):
    """Bank of paired TRAIN residual sites; no descriptor model and no VAL/TEST fitting."""
    from .runtime import project_path
    from .mri_case_cache import acquire_lock

    if dataset.split != "train":
        raise ValueError("MLS regularizer banks must be built from TRAIN")
    nsites, npatches = int(settings.get("sites_per_case",256)), int(settings.get("patches_per_case",4))
    if min(nsites,npatches) < 1:
        raise ValueError("Positive MLS sampling budgets required")
    workers = settings.get("bank_workers", 0)
    if type(workers) is not int or not 0 <= workers <= 8:
        raise ValueError("bank_workers must be an integer from 0 to 8")
    seed = int(settings.get("seed",42))
    paired = settings.get("geometry_mode", "normal") == "paired_curved"
    identity = {"version":2 if paired else 1,"train":[[c.id,c.fingerprint] for c in dataset.cases],
                "mean":model.band_mean.cpu().tolist(),"std":model.band_std.cpu().tolist(),
                "whitened":model.use_band_whitening,"sites_per_case":nsites,"patches_per_case":npatches,"seed":seed}
    path = project_path(settings["bank"])
    path.parent.mkdir(parents=True,exist_ok=True)
    with acquire_lock(path.with_suffix(".lock")):
        if path.is_file():
            saved = load_checkpoint(path)
            if saved.get("identity") != identity:
                raise ValueError("TRAIN MLS bank identity differs; choose a new bank path")
            return saved
        ds = copy.copy(dataset)
        ds.samples_per_epoch, ds.samples_per_volume = None, min(npatches,nsites)
        ds.seed = seed
        ds.set_epoch(0)
        generator = torch.Generator().manual_seed(seed)
        records, bases, cases = [], [], []
        device = model.band_mean.device
        # Dataset crops depend on (seed, epoch, index), and DataLoader yields
        # them in index order. Worker count therefore cannot change bank rows
        # or the independent balanced-site generator. Spawn avoids forking an
        # already initialized CUDA runtime in the prebuild process.
        loader_options = (dict(prefetch_factor=2, multiprocessing_context="spawn") if workers else {})
        loader = torch.utils.data.DataLoader(
            ds, batch_size=None, num_workers=workers,
            generator=torch.Generator().manual_seed(seed+1729), **loader_options)
        if paired:
            print(f"TRAIN MLS paired bank: sampling {len(ds)} patches from {len(ds.cases)} cases "
                  f"with {workers} workers", flush=True)
        for idx, item in enumerate(loader):
            slot = idx % ds.samples_per_volume
            quota = nsites//ds.samples_per_volume + (slot < nsites%ds.samples_per_volume)
            sites = _balanced_sites(item["lr"],quota,generator)
            device_sites = sites.to(device)
            coefficients = model.codec.encode((item["hr"]-item["base"])[None].to(device))
            values = _sites(model.normalize_haar(coefficients))[device_sites].cpu()
            records.append(values)
            if paired:
                bases.append(_sites(model.codec.encode(item["base"][None].to(device)))[device_sites].cpu())
            cases.append(torch.full((len(values),),idx//ds.samples_per_volume,dtype=torch.long))
            if paired and ((idx+1) % 256 == 0 or idx+1 == len(ds)):
                print(f"TRAIN MLS paired bank: {idx+1}/{len(ds)} patches", flush=True)
        payload = dict(identity=identity,residuals=torch.cat(records),case_indices=torch.cat(cases),
                       train_case_ids=[c.id for c in ds.cases])
        if paired:
            payload.update(base_coefficients=torch.cat(bases),
                           band_scale=(model.band_std.detach().cpu()+1.e-6 if model.use_band_whitening
                                       else torch.ones(16)),
                           provenance={"split":"train", "paired_source":"base_haar",
                                       "train_fingerprints":[c.fingerprint for c in ds.cases]})
        atomic_save(payload,path)
        return payload


class TrainingMLSAtlas(nn.Module):
    """Cross-fitted residual charts, queried only by detached paired TRAIN truth.

    Legacy ``normal`` charts use whitened residuals. ``paired_curved`` charts
    use physical residual coefficients for all PCA, coordinates and decoded
    errors, with whitened residual/base descriptors used only for retrieval.
    The paired base supplies the observed LR context; no degradation operator
    or synthetic downsampling relationship is assumed.
    """
    _decode = SpatialAtlas._decode

    def __init__(self,payload,settings):
        super().__init__()
        self.settings, self.method = dict(settings), "mls"
        self.geometry_mode = settings.get("geometry_mode", "normal")
        if self.geometry_mode not in {"normal", "paired_curved"}:
            raise ValueError("Unknown training MLS geometry_mode")
        self.loss_sampling = settings.get("loss_sampling", "uniform")
        if self.loss_sampling not in {"uniform", "structure_stratified"}:
            raise ValueError("MLS loss_sampling must be uniform or structure_stratified")
        self.structure_fractions = tuple(settings.get("structure_fractions", (.1, .2, .5, .2)))
        if (len(self.structure_fractions) != 4 or not np.isfinite(self.structure_fractions).all()
                or min(self.structure_fractions) < 0 or not np.isclose(sum(self.structure_fractions), 1.)):
            raise ValueError("MLS structure_fractions must be four nonnegative fractions summing to one")
        self.neighbor_weight = float(settings.get("neighbor_weight", 0.))
        if not np.isfinite(self.neighbor_weight) or self.neighbor_weight < 0:
            raise ValueError("MLS neighbor_weight must be finite and nonnegative")
        if self.geometry_mode != "paired_curved" and (self.loss_sampling != "uniform" or self.neighbor_weight):
            raise ValueError("Structural sampling and neighbor loss require paired_curved MLS")
        self.paired_base_weight = float(settings.get("paired_base_weight", 1.))
        if not np.isfinite(self.paired_base_weight) or self.paired_base_weight < 0:
            raise ValueError("paired_base_weight must be finite and nonnegative")
        self.use_paired_base = self.geometry_mode == "paired_curved" and self.paired_base_weight > 0
        self.dim = int(settings.get("dim",4))
        ids = payload["train_case_ids"]
        nf = min(len(ids),int(settings.get("folds",5)))
        if nf < 2 or not 1 <= self.dim < 16 or len(set(ids)) != len(ids):
            raise ValueError("TRAIN MLS needs distinct cases, >=2 folds and 1..15 tangent dimensions")
        seed = int(settings.get("seed",42))
        ordered = sorted(ids,key=lambda case: hashlib.sha256(f"{seed}:{case}".encode()).digest())
        self.folds = {case:i%nf for i,case in enumerate(ordered)}
        residuals = payload["residuals"].detach().cpu().float()
        cases = payload["case_indices"].detach().cpu()
        if (residuals.ndim != 2 or residuals.shape[1] != 16 or not len(residuals)
                or not residuals.isfinite().all() or cases.shape != (len(residuals),)
                or cases.is_floating_point() or cases.dtype == torch.bool
                or cases.min() < 0 or cases.max() >= len(ids)):
            raise ValueError("Invalid aligned TRAIN residual records and case indices")
        if payload.get("provenance", {}).get("split", "train") != "train":
            raise ValueError("Training MLS charts require TRAIN provenance")
        if len(cases.unique()) != len(ids):
            raise ValueError("Every TRAIN case needs at least one sampled residual")
        row_fold = torch.tensor([self.folds[case] for case in ids])[cases.long()]
        source, source_weights = residuals, None
        if self.use_paired_base:
            base = payload.get("base_coefficients")
            if base is None or base.shape != residuals.shape or not base.isfinite().all():
                raise ValueError("Paired MLS requires aligned finite TRAIN base_coefficients")
            source = torch.cat((residuals, base.detach().cpu().float()), -1)
            source_weights = torch.cat((torch.ones(16), torch.full((16,), self.paired_base_weight)))
        physical_scale = torch.as_tensor(payload.get("band_scale", torch.ones(16))).detach().cpu().float().flatten()
        if physical_scale.shape != (16,) or not physical_scale.isfinite().all() or (physical_scale <= 0).any():
            raise ValueError("MLS bank band_scale must contain 16 positive finite physical scales")
        self.register_buffer("physical_scale", physical_scale)
        fit_target = residuals * physical_scale if self.geometry_mode == "paired_curved" else residuals
        limit = int(settings.get("max_fit_sites_per_fold", 65536))
        if limit < 1:
            raise ValueError("max_fit_sites_per_fold must be positive")
        chunks, avgs, scales, self.offsets = {}, [], [], [0]
        self.fit_sites_per_fold = []
        for fold in range(nf):
            gen = torch.Generator().manual_seed(seed+fold)
            rows = (row_fold != fold).nonzero()[:, 0]
            if len(rows) > limit:
                rows = rows[torch.randperm(len(rows), generator=gen)[:limit]]
            self.fit_sites_per_fold.append(len(rows))
            avg,scale,fit = _fit_charts(source[rows],fit_target[rows],settings,self.dim,gen,source_weights)
            self.offsets.append(self.offsets[-1]+len(fit["centers"]))
            for key,value in fit.items():
                chunks.setdefault(key,[]).append(value)
            avgs.append(avg)
            scales.append(scale)
        for key,parts in chunks.items():
            self.register_buffer(key,torch.cat(parts))
        self.register_buffer("query_mean",torch.stack(avgs))
        self.register_buffer("query_std",torch.stack(scales))
        self.eval().requires_grad_(False)

    def reconstruction_loss(self, prediction, target, case_ids, band_scale, base_coefficients=None,
                            reference_image=None):
        terms = self.reconstruction_terms(prediction, target, case_ids, band_scale,
                                          base_coefficients, reference_image)
        return terms["local_mse"] + self.neighbor_weight*terms["neighbor_mse"]

    def reconstruction_terms(self, prediction, target, case_ids, band_scale, base_coefficients=None,
                             reference_image=None):
        """Target-anchored curved normal discrepancy in physical Haar MSE units.

        For chart decoder g, this measures ||(p-y)-(g(q_p)-g(q_y))||^2 / 16.
        Coordinates are Euclidean projections onto the *physical* PCA basis.
        Both decodes are bounded to the fitted coordinate domain. The target
        anchor cancels chart approximation bias exactly at p=y, while the
        predicted coordinates retain gradients through the quadratic decoder.
        The companion image/endpoint MSE supplies tangential supervision.
        The optional neighbor term penalizes XY differences of these *curved
        defects*, not image gradients, in their common physical ambient space.
        It is target-anchored and vanishes for exact predictions even across
        charts. An anchor budget queries at most three times as many sites
        when neighbors are enabled, with bounded chart-query chunks.
        """
        if self.geometry_mode != "paired_curved":
            raise ValueError("reconstruction_loss requires geometry_mode='paired_curved'")
        if (prediction.shape != target.shape or prediction.ndim < 3 or prediction.shape[1] != 16
                or not len(prediction) or len(case_ids) != len(prediction)
                or any(case not in self.folds for case in case_ids)):
            raise ValueError("MLS reconstruction needs matching predictions and identified TRAIN targets")
        if self.use_paired_base and (base_coefficients is None or base_coefficients.shape != target.shape):
            raise ValueError("MLS reconstruction requires paired base_coefficients matching the residual grid")
        structural = self.loss_sampling == "structure_stratified"
        if structural and reference_image is None:
            raise ValueError("structure_stratified MLS needs the detached TRAIN reference_image")
        if (structural or self.neighbor_weight or reference_image is not None) and prediction.ndim != 5:
            raise ValueError("Structural MLS losses require B16ZYX Haar volumes")
        if reference_image is not None and len(reference_image) != len(prediction):
            raise ValueError("MLS reference_image batch differs")
        # Model whitening adds epsilon. Callers pass that exact physical scale,
        # or ones when whitening is disabled. The residual mean cancels in all
        # differences, so it need not enter this metric or the fitted charts.
        scale = torch.as_tensor(band_scale, device=prediction.device, dtype=torch.float32).detach().flatten()
        if (scale.shape != (16,) or not torch.isfinite(scale).all() or (scale <= 0).any()
                or not torch.allclose(scale, self.physical_scale, rtol=1.e-5, atol=1.e-9)):
            raise ValueError("MLS physical band_scale differs from the fitted TRAIN bank")
        budget = int(self.settings.get("loss_sites_per_patch",256))
        chunk = int(self.settings.get("query_chunk",2048))
        coordinate_limit = float(self.settings.get("coordinate_limit",3.))
        if min(budget, chunk) < 1 or not np.isfinite(coordinate_limit) or coordinate_limit <= 0:
            raise ValueError("Positive MLS query budgets and coordinate_limit required")
        totals, neighbor_totals = [], []
        coverage = prediction.new_zeros(4, dtype=torch.float32)
        sampled_sites, pair_count = 0, 0
        # Keep chart algebra in float32 under AMP: tiny physical coefficients
        # and their quadratic discrepancies otherwise underflow in float16.
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            for b,case in enumerate(case_ids):
                truth = _sites(target[b:b+1]).detach().float()
                pred = _sites(prediction[b:b+1]).float()
                base = (_sites(base_coefficients[b:b+1]).detach().float() if self.use_paired_base else None)
                labels = (_structure_labels(reference_image[b:b+1], prediction.shape[2:])
                          if reference_image is not None else None)
                sites = (_stratified_sites(labels, budget, self.structure_fractions) if structural else
                         torch.arange(len(truth),device=truth.device) if budget >= len(truth) else
                         torch.randperm(len(truth),device=truth.device)[:budget])
                sampled_sites += len(sites)
                if labels is not None:
                    coverage += torch.bincount(labels[sites], minlength=4).to(coverage)
                left, right = (_xy_neighbor_pairs(sites, prediction.shape[2:]) if self.neighbor_weight else
                               (sites[:0], sites[:0]))
                query_sites = torch.unique(torch.cat((sites, right)), sorted=True)
                defects, supports = [], []
                for indices in query_sites.split(chunk):
                    with torch.no_grad():
                        fold = self.folds[case]
                        lo,hi = self.offsets[fold:fold+2]
                        x = truth[indices]
                        descriptor = torch.cat((x,base[indices]),-1) if base is not None else x
                        query = (descriptor-self.query_mean[fold])/self.query_std[fold]
                        distance, selected = torch.cdist(query,self.centers[lo:hi]).min(1)
                        selected = selected+lo
                        trust = (1+(distance/self.radius[selected,0]).square()).reciprocal()
                        support = self.pca_supported[selected]*trust
                        origin, basis = self.mean[selected], self.basis[selected]
                        bandwidth = self.bandwidth[selected]
                        lower = self.domain_lo[selected].clamp(-coordinate_limit, coordinate_limit)
                        upper = self.domain_hi[selected].clamp(-coordinate_limit, coordinate_limit)
                        x = x*scale
                        q_target = ((x-origin).unsqueeze(-2) @ basis).squeeze(-2)/bandwidth
                        decoded_target, _ = self._decode(selected,q_target.clamp(lower,upper))
                    physical_pred = pred[indices]*scale
                    q_pred = ((physical_pred-origin).unsqueeze(-2) @ basis).squeeze(-2)/bandwidth
                    decoded_pred, _ = self._decode(selected,q_pred.clamp(lower,upper))
                    defect = (physical_pred-x)-(decoded_pred-decoded_target)
                    defects.append(defect)
                    supports.append(support)
                defects, supports = torch.cat(defects), torch.cat(supports)
                anchor_ids = torch.searchsorted(query_sites, sites)
                totals.append(defects[anchor_ids].square().mean(-1)*supports[anchor_ids])
                if len(left):
                    a, b = torch.searchsorted(query_sites, left), torch.searchsorted(query_sites, right)
                    # Divide the difference by two, retaining physical MSE units.
                    # Both endpoints must be supported by their TRAIN chart.
                    neighbor_totals.append((defects[a]-defects[b]).div(2).square().mean(-1)
                                           * torch.minimum(supports[a], supports[b]))
                    pair_count += len(left)
        terms = dict(local_mse=torch.cat(totals).mean(),
                     neighbor_mse=(torch.cat(neighbor_totals).mean() if neighbor_totals
                                   else prediction.reshape(-1)[0].float()*0.),
                     sampled_sites=prediction.new_tensor(sampled_sites, dtype=torch.float32),
                     neighbor_pairs=prediction.new_tensor(pair_count, dtype=torch.float32))
        terms.update({"sample_fraction_"+name: coverage[i]/max(sampled_sites, 1)
                      for i, name in enumerate(("background", "weak", "texture", "edge"))})
        return terms

    def normal_loss(self,prediction,target,case_ids):
        from .atlas_geometry import normal_residual_energy

        if self.geometry_mode != "normal":
            raise ValueError("Paired curved MLS uses reconstruction_loss with physical scales and paired base coefficients")
        if prediction.shape != target.shape or len(case_ids) != len(prediction) or any(c not in self.folds for c in case_ids):
            raise ValueError("MLS normal loss needs matching predictions and identified TRAIN targets")
        totals = []
        budget = int(self.settings.get("loss_sites_per_patch",256))
        chunk = int(self.settings.get("query_chunk",2048))
        if min(budget,chunk) < 1:
            raise ValueError("Positive MLS regularizer query budgets required")
        for b,case in enumerate(case_ids):
            truth = _sites(target[b:b+1]).detach()
            pred = _sites(prediction[b:b+1])
            sites = torch.randperm(len(truth),device=truth.device)[:budget]
            for indices in sites.split(chunk):
                with torch.no_grad():
                    fold = self.folds[case]
                    lo,hi = self.offsets[fold:fold+2]
                    x = truth[indices]
                    query = (x-self.query_mean[fold])/self.query_std[fold]
                    selected = torch.cdist(query,self.centers[lo:hi]).argmin(1)+lo
                    qn = ((x-self.mean[selected]).unsqueeze(-2) @ self.basis[selected]).squeeze(-2)
                    qn = (qn/self.bandwidth[selected]).clamp(-3,3)
                    _,jacobian = self._decode(selected,qn)
                    tangent = torch.linalg.qr(jacobian,mode="reduced").Q
                    supported = self.pca_supported[selected]
                totals.append(normal_residual_energy(pred[indices]-x,tangent)*supported)
        return torch.cat(totals).mean()


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
