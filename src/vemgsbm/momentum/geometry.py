"""Local affine PCA in the spatial residual coefficients of training HR pairs."""
from dataclasses import dataclass

import torch


def tokens(volume):
    return volume.movedim(1, -1).reshape(-1, volume.shape[1])


@dataclass
class LocalChart:
    center: torch.Tensor
    basis: torch.Tensor
    confidence: torch.Tensor

    def cost(self, position, velocity, free):
        weights = self.confidence * free.reshape(-1)
        # Subtract the in-plane energy instead of explicitly constructing I-UU^T.
        def normal_energy(vectors):
            coordinates = torch.einsum("nc,ncr->nr", vectors, self.basis)
            residual = vectors.square().sum(-1) - coordinates.square().sum(-1)
            return (weights * residual.clamp_min(0)).sum() / (free.sum().clamp_min(1) * vectors.shape[-1])

        return normal_energy(tokens(position) - self.center), normal_energy(tokens(velocity))


@torch.no_grad()
def fit_atlas(bank, centers, neighbors, tangent_rank, seed):
    """Cluster training tokens, then estimate each local subspace by k-NN PCA."""
    bank = bank.float().cpu()
    if bank.ndim != 2 or not torch.isfinite(bank).all():
        raise ValueError("Expected finite training feature tokens")
    n, dim = bank.shape
    if not 0 < tangent_rank < dim or n <= tangent_rank + 1:
        raise ValueError("Insufficient training tokens for local PCA")
    if centers < 1 or neighbors <= tangent_rank + 1:
        raise ValueError("PCA needs positive centers and enough neighbors")
    generator = torch.Generator().manual_seed(seed)
    k = min(centers, n)
    means = bank[torch.randperm(n, generator=generator)[:k]].clone()
    for _ in range(8):
        assignment = torch.cdist(bank, means).argmin(-1)
        sums = torch.zeros_like(means).index_add_(0, assignment, bank)
        counts = torch.bincount(assignment, minlength=k)[:, None]
        means = torch.where(counts > 0, sums / counts.clamp_min(1), means)
    distances = torch.cdist(means, bank)
    indices = distances.topk(min(neighbors, n), largest=False).indices
    local = bank[indices]
    origins = local.mean(1)
    centered = local - origins[:, None]
    covariance = centered.transpose(1, 2) @ centered / (local.shape[1] - 1)
    values, vectors = torch.linalg.eigh(covariance.double())
    basis = vectors[:, :, -tangent_rank:].float().contiguous()
    leading = values[:, -tangent_rank]
    next_value = values[:, -tangent_rank-1]
    confidence = ((leading - next_value) / (leading + 1e-7)).clamp(0.05, 1).float()
    return {"keys": means, "centers": origins, "basis": basis, "confidence": confidence,
            "eigenvalues": values.float(), "training_tokens": n}


@torch.no_grad()
def assign_chart(atlas, reference_position):
    """Choose a local chart at the reference midpoint for each spatial token.

    Freeze the choice during path optimization so the local objective is smooth.
    The costs penalize normal motion without projecting out those directions.
    """
    position = tokens(reference_position)
    nearest = torch.cdist(position, atlas["keys"]).argmin(-1)
    return LocalChart(atlas["centers"][nearest], atlas["basis"][nearest], atlas["confidence"][nearest])
