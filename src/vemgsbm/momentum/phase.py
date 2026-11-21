"""Integrated Brownian bridges and geometry-dependent mean optimization.

The reference is dZ=V dt, dV=sigma dW conditioned on terminal position.
The acceleration follows 3MSBM equation (54), with fixed paired endpoints.
This implementation derives joint transition moments by Gaussian conditioning.
"""
from dataclasses import dataclass
import math

import torch


def _transition_coefficients(elapsed, remaining):
    """Gains and covariance entries before multiplying by sigma squared."""
    fraction = elapsed / remaining
    # Factorized forms avoid cancellation of the vanishing position variance.
    gain_z = fraction**2 * (3 - fraction) / 2
    gain_v = 3 * fraction * (2 - fraction) / (2 * remaining)
    var_z = elapsed**3 * (1 - fraction)**2 * (4 - fraction) / 12
    cov = elapsed**2 * (1 - fraction) * (2 - 4*fraction + fraction**2) / 4
    var_v = elapsed * (1 - 3*fraction * (2 - fraction)**2 / 4)
    return gain_z, gain_v, var_z, cov, var_v


def transition(position, velocity, endpoint, elapsed, remaining, sigma, generator=None):
    """Exact conditional transition with a fixed endpoint over this step."""
    if not 0 < elapsed <= remaining:
        raise ValueError("Require 0 < elapsed <= remaining")
    g_z, g_v, var_z, cov, var_v = _transition_coefficients(elapsed, remaining)
    residual = endpoint - position - remaining * velocity
    mean_v = velocity + g_v * residual
    if elapsed == remaining:
        # The terminal position is pinned. Terminal velocity is free.
        eps = torch.randn(velocity.shape, device=velocity.device, dtype=velocity.dtype, generator=generator)
        return endpoint, mean_v + sigma * math.sqrt(var_v) * eps
    mean_z = position + elapsed * velocity + g_z * residual
    l_z = math.sqrt(var_z)
    l_vz = cov / l_z
    l_v = math.sqrt(max(0.0, var_v - l_vz**2))
    noise_z = torch.randn(position.shape, device=position.device, dtype=position.dtype, generator=generator)
    noise_v = torch.randn(velocity.shape, device=velocity.device, dtype=velocity.dtype, generator=generator)
    return mean_z + sigma * l_z * noise_z, mean_v + sigma * (l_vz * noise_z + l_v * noise_v)


def _time(time, reference):
    value = torch.as_tensor(time, device=reference.device, dtype=reference.dtype)
    if value.numel() == 1:
        value = value.expand(len(reference))
    return value.reshape(len(reference), *([1] * (reference.ndim - 1)))


def _shape_functions(t):
    """Two smooth shifts preserving positions and reference endpoint velocities."""
    f = (t * (1 - t))**2
    derivative = 2 * t * (1 - t) * (1 - 2*t)
    # u_delta = q'' + 3q/(1-t)^2 + 3q'/(1-t), expanded at the endpoint.
    control = 2 - 6*t + 3*t*t
    g = 2*t - 1
    return ((f, f*g), (derivative, derivative*g + 2*f),
            (control, control*g + 4*derivative + 6*t*t*(1-t)))


@dataclass
class PhasePath:
    endpoint: torch.Tensor
    shifts: torch.Tensor
    sigma: float
    free: torch.Tensor

    def sample(self, time, noise_z=None, noise_v=None):
        """Sample correlated (position, velocity) and the scaled acceleration target."""
        t = _time(time, self.endpoint)
        if not bool(((t >= 0) & (t <= 1)).all()):
            raise ValueError("Time must lie in [0, 1]")
        g_z, g_v, var_z, cov, var_v = _transition_coefficients(t, 1.0)
        functions, derivatives, controls = _shape_functions(t)
        first, second = self.shifts.unbind(0)
        mean_z = g_z * self.endpoint + first * functions[0] + second * functions[1]
        mean_v = g_v * self.endpoint + first * derivatives[0] + second * derivatives[1]
        correction = first * controls[0] + second * controls[1]
        l_z = var_z.clamp_min(0).sqrt()
        l_vz = torch.where(l_z > 0, cov / l_z.clamp_min(1e-20), 0)
        l_v = (var_v - l_vz.square()).clamp_min(0).sqrt()
        noise_z = torch.randn_like(mean_z) if noise_z is None else noise_z
        noise_v = torch.randn_like(mean_v) if noise_v is None else noise_v
        z = (mean_z + self.sigma * l_z * noise_z) * self.free
        v = (mean_v + self.sigma * (l_vz * noise_z + l_v * noise_v)) * self.free
        # The acceleration equals 3*(anchor - z - (1-t)*v)/(1-t)^2.
        anchor = (self.endpoint + (1 - t).square() * correction / 3) * self.free
        return z, v, anchor, correction * self.free


@torch.enable_grad()
def optimize_path(endpoint, free, chart, config, *, control_scale=1.0):
    """Optimize fixed-covariance bridge means under soft data geometry costs.

    The control cost is sigma^2 times KL relative to the conditioned reference,
    normalized per coordinate. A deterministic shift adds only u_delta to its
    acceleration. Geometry uses common Gaussian samples across local iterations.
    """
    endpoint = endpoint.detach() * free
    shifts = endpoint.new_zeros((2, *endpoint.shape), requires_grad=True)
    path = PhasePath(endpoint, shifts, float(config["sigma"]), free)
    optimizer = torch.optim.Adam([shifts], lr=float(config["lr"]))
    points = int(config["quadrature"])
    times = (torch.arange(points, device=endpoint.device, dtype=endpoint.dtype) + 0.5) / points
    noise_z = torch.randn((points, *endpoint.shape), device=endpoint.device, dtype=endpoint.dtype)
    noise_v = torch.randn_like(noise_z)
    geometry_cost = chart if callable(chart) else chart.cost
    geometry_ramp = float(config.get("geometry_ramp_power", 0.))
    if not math.isfinite(geometry_ramp) or geometry_ramp < 0:
        raise ValueError("Momentum path geometry ramp must be finite and nonnegative")

    def cost():
        totals = []
        for i, t in enumerate(times):
            z, v, _, correction = path.sample(t, noise_z[i], noise_v[i])
            normal_position, normal_velocity = geometry_cost(z, v, free)
            control = 0.5 * control_scale * correction.square().mean()
            weight = t.pow(geometry_ramp)
            totals.append(torch.stack((control, weight * normal_position, weight * normal_velocity)))
        terms = torch.stack(totals).mean(0)
        total = terms[0] + config["state_weight"] * terms[1] + config["normal_weight"] * terms[2]
        return total, terms

    with torch.no_grad():
        initial, best_terms = cost()
        if not torch.isfinite(initial):
            raise FloatingPointError("Nonfinite reference momentum path objective")
        best = float(initial)
        best_shifts = shifts.detach().clone()
    backtracks = config.get("backtracking_steps", 0)
    if isinstance(backtracks, bool) or not isinstance(backtracks, int) or backtracks < 0:
        raise ValueError("Momentum path backtracking steps must be a nonnegative integer")
    accepted, shrinks = 0, 0
    for _ in range(config["iterations"]):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = cost()
        loss.backward()
        previous = shifts.detach().clone() if backtracks else None
        torch.nn.utils.clip_grad_norm_([shifts], 1.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            shifts.clamp_(-config["max_shift"], config["max_shift"])
            if backtracks:
                update = shifts - previous
            candidate, terms = cost()
            for attempt in range(backtracks):
                if torch.isfinite(candidate) and candidate <= loss:
                    break
                shifts.copy_(previous + update * (.5 ** (attempt + 1)))
                candidate, terms = cost()
                shrinks += 1
            if backtracks and (not torch.isfinite(candidate) or candidate > loss):
                shifts.copy_(previous)
                continue
            if not torch.isfinite(candidate):
                raise FloatingPointError("Nonfinite momentum path objective")
            if float(candidate) < best:
                best, best_terms = float(candidate), terms
                best_shifts = shifts.detach().clone()
                accepted += 1
    return PhasePath(endpoint, best_shifts, path.sigma, free), {
        "path_before": float(initial), "path_after": best,
        "path_control": float(best_terms[0]), "path_normal_position": float(best_terms[1]),
        "path_normal_velocity": float(best_terms[2]), "path_shift_rms": float(best_shifts.square().mean().sqrt()),
        "path_objective_improvement": float(initial) - best, "path_improving_steps": accepted,
        "path_backtracks": shrinks,
    }
