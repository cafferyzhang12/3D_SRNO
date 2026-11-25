"""Direction-conditioned neural drift, not a clean-endpoint predictor."""
import math
import torch
from torch import nn
from .models import Reconstructor, volume_inputs
from .hybrid import HybridWaveletNAFNet25D


class DriftNetwork(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.backbone = Reconstructor(width)

    def forward(self, state, base, mask, time, direction="forward"):
        if direction not in {"forward", "backward"}:
            raise ValueError("Unknown drift direction")
        prediction = self.backbone(state, base, mask, time, int(direction == "backward"))
        return (prediction - base) * (1 - mask)


def diffusion_drift_weights(reference_sigma, inference_sigma):
    """Weights for a new diffusion with the trained one-time marginals.

    Both drifts use positive simulation time. Current velocity is (f-r)/2
    and the score multiplied by reference_sigma**2 is f+r.
    """
    if not math.isfinite(reference_sigma) or reference_sigma <= 0:
        raise ValueError("Diffusion correction requires positive finite training sigma")
    if not math.isfinite(inference_sigma) or inference_sigma < 0:
        raise ValueError("Inference sigma must be finite and nonnegative")
    ratio = (inference_sigma / reference_sigma) ** 2
    return (1 + ratio) / 2, (ratio - 1) / 2


def diffusion_adjusted_model(model, reference_sigma, inference_sigma):
    own, opposite = diffusion_drift_weights(reference_sigma, inference_sigma)
    if opposite == 0:
        return model

    def drift(state, base, mask, time, direction):
        other_direction = "backward" if direction == "forward" else "forward"
        forward_time = model(state, base, mask, time, direction)
        reversed_time = model(state, base, mask, time, other_direction)
        return own * forward_time + opposite * reversed_time

    return drift


@torch.no_grad()
def integrate(model, initial, base, mask, steps=20, sigma=0.1, direction="forward", generator=None,
              terminal_noise=True, state_guidance=None, transition_guidance=None):
    """Euler-Maruyama in forward or reversed time; no HR available at inference.

    Optional guidance acts on provisional interior states. Transition guidance
    also receives the previous state, before the drift and noise updates.
    Both directions use a positive simulation step. The backward network is
    trained against the reversed-time drift, so no second sign flip is applied.
    Times are supplied in original (LR=0, HR=1) coordinates.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("SDE steps must be a positive integer")
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("SDE sigma must be finite and nonnegative")
    if direction not in {"forward", "backward"}:
        raise ValueError("Unknown drift direction")
    base, mask = volume_inputs(initial, base, mask)
    if not bool(torch.isfinite(initial).all() & torch.isfinite(base).all()):
        raise ValueError("SDE input images must be finite")
    state = torch.where(mask.bool(), base, initial).clone()
    dt = 1.0 / steps
    for index in range(steps):
        time = index / steps if direction == "forward" else 1 - index / steps
        # Endpoint Gaussian conditional drifts have singular representations;
        # endpoint time embeddings are evaluated at the training cutoff.
        time = min(0.98, max(0.02, time))
        previous = state
        drift = model(state, base, mask, time, direction)
        noise = torch.randn(state.shape, dtype=state.dtype, device=state.device, generator=generator)
        state = state + dt * drift
        # Draw even when discarded so the next tile sees the original RNG stream.
        if terminal_noise or index + 1 < steps:
            state = state + sigma * dt**0.5 * noise * (1 - mask)
        state = torch.where(mask.bool(), base, state)
        if (state_guidance is not None or transition_guidance is not None) and index + 1 < steps:
            next_time = (index + 1) / steps
            if direction == "backward":
                next_time = 1 - next_time
            if transition_guidance is not None:
                state = transition_guidance(state, next_time, dt, mask, previous)
            else:
                state = state_guidance(state, next_time, dt, mask)
            state = torch.where(mask.bool(), base, state)
        if not torch.isfinite(state).all():
            raise FloatingPointError("Nonfinite GSBM SDE trajectory")
    return state
