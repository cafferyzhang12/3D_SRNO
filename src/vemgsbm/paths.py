"""Gaussian conditional stochastic-control paths and bridge-matching targets.

Independent implementation of the Gaussian-path and conditional-control formulas
in Liu et al., *Generalized Schrödinger Bridge Matching*, ICLR 2024, Eq. (8).
Reference implementation inspected: facebookresearch/generalized-schrodinger-
bridge-matching/gsbm/gaussian_path.py (CC BY-NC 4.0, Meta Platforms).

This adaptation fixes the observed LR/HR pairing and optimizes interior mean
knots. An optional bounded log-scale spline also optimizes Gaussian variance,
gamma(t)=sigma*sqrt(t*(1-t))*exp(h(t)), with h(0)=h(1)=0. It is a
restricted paired conditional variational GSBM, not a reproduction of the full
unpaired algorithm, learned endpoint recoupling or path-
integral debiasing. The conditional controls below are analytic; a neural drift
is subsequently fitted to them by bridge matching, without IPF.

For fixed Brownian variance, E[|u|^2]/2 = |m_dot|^2/2 + C(t). We subtract the
Brownian C(t) baseline, independent of optimized parameters, including when
optimizing variance. Using h(0)=h(1)=0, integration by parts expresses the
residual variance action as the nonnegative density
0.5*gamma^2*h_dot^2 + sigma^2*sinh(h)^2/(2*t*(1-t)). This avoids cancellation
and spurious negative variance costs from coarse quadrature of singular terms.
This avoids the endpoint divergence of the Dirac-conditioned reference. All image
energies are normalized per unobserved voxel. The optional metric kinetic term
is an explicit MFM-inspired extension of the conditional-control objective.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor


def _times(time: Tensor | float, reference: Tensor, *, interior: bool = False) -> Tensor:
    value = torch.as_tensor(time, dtype=reference.dtype, device=reference.device)
    if value.numel() == 1:
        value = value.reshape(1).expand(reference.shape[0])
    elif value.numel() == reference.shape[0]:
        value = value.reshape(reference.shape[0])
    else:
        raise ValueError("time must be scalar or have one element per batch item")
    valid = torch.isfinite(value) & (value > 0 if interior else value >= 0)
    valid &= value < 1 if interior else value <= 1
    if not bool(valid.all()):
        raise ValueError("drift time must be in (0,1)" if interior else "time must be in [0,1]")
    return value


def _expand(value: Tensor, reference: Tensor) -> Tensor:
    return value.reshape(-1, *([1] * (reference.ndim - 1)))


def _mean_free(value: Tensor, missing: Tensor) -> Tensor:
    axes = tuple(range(1, value.ndim))
    return (value * missing).sum(axes) / missing.sum(axes).clamp_min(1)


@dataclass
class PathBatch:
    """A batch of endpoint-fixed Gaussian paths on image coordinates.

    ``mean_knots`` contains only interior absolute mean images, shape
    (batch, knot_count-2, channels, z, y, x). ``mask`` marks exact observations.
    Times always refer to physical forward time (0=LR source, 1=HR target),
    including when requesting the drift of the reversed-time process.
    """

    source: Tensor
    target: Tensor
    mask: Tensor
    mean_knots: Tensor
    sigma: float = 0.1
    std_knots: Tensor | None = None
    std_bound: float = 0.75

    def __post_init__(self):
        if self.source.ndim != 5 or not self.source.is_floating_point():
            raise ValueError("images must be floating (batch, channels, z, y, x) tensors")
        if self.target.shape != self.source.shape or self.target.device != self.source.device:
            raise ValueError("source and target must have identical shape and device")
        if self.target.dtype != self.source.dtype:
            raise ValueError("source and target must have identical dtype")
        if not bool(torch.isfinite(self.source).all() & torch.isfinite(self.target).all()):
            raise ValueError("endpoint images must be finite")
        try:
            self.mask = torch.broadcast_to(self.mask.to(self.source), self.source.shape)
        except RuntimeError as error:
            raise ValueError("mask must broadcast to image shape") from error
        if not bool(((self.mask == 0) | (self.mask == 1)).all()):
            raise ValueError("mask must be binary")
        if not torch.equal(self.source * self.mask, self.target * self.mask):
            raise ValueError("paired endpoints differ at exactly observed voxels")
        if (self.mean_knots.ndim != self.source.ndim + 1
                or self.mean_knots.shape[0] != self.source.shape[0]
                or self.mean_knots.shape[1] < 1
                or self.mean_knots.shape[2:] != self.source.shape[1:]
                or self.mean_knots.device != self.source.device
                or self.mean_knots.dtype != self.source.dtype):
            raise ValueError("mean_knots must be (batch, >=1, channels, z, y, x)")
        if not bool(torch.isfinite(self.mean_knots).all()):
            raise ValueError("mean knots must be finite")
        self.sigma = float(self.sigma)
        if not math.isfinite(self.sigma) or self.sigma < 0:
            raise ValueError("sigma must be finite and nonnegative")
        if not math.isfinite(self.std_bound) or self.std_bound <= 0:
            raise ValueError("std_bound must be finite and positive")
        if self.std_knots is not None:
            if (self.std_knots.shape != self.mean_knots.shape[:2]
                    or self.std_knots.device != self.source.device
                    or self.std_knots.dtype != self.source.dtype
                    or not bool(torch.isfinite(self.std_knots).all())):
                raise ValueError("std_knots must be finite (batch, knot_count-2) matching image dtype/device")

    @property
    def base(self) -> Tensor:
        return self.source

    @property
    def knots(self) -> Tensor:
        interior = self.mask[:, None] * self.source[:, None] + (1 - self.mask[:, None]) * self.mean_knots
        return torch.cat((self.source[:, None], interior, self.target[:, None]), dim=1)

    def mean_and_velocity(self, time: Tensor | float) -> tuple[Tensor, Tensor]:
        """Piecewise-linear mean and its exact a.e. forward-time derivative."""
        t = _times(time, self.source)
        knots = self.knots
        segments = knots.shape[1] - 1
        scaled = t * segments
        left = scaled.floor().long().clamp(max=segments - 1)
        weight = _expand(scaled - left, self.source)
        batch = torch.arange(self.source.shape[0], device=self.source.device)
        lo, hi = knots[batch, left], knots[batch, left + 1]
        mean = lo + weight * (hi - lo)
        velocity = (hi - lo) * segments
        # Return exact endpoint tensors rather than floating interpolation drift.
        mean = torch.where(_expand(t == 0, mean), self.source, mean)
        mean = torch.where(_expand(t == 1, mean), self.target, mean)
        return mean, velocity * (1 - self.mask)

    def mean(self, time: Tensor | float) -> Tensor:
        return self.mean_and_velocity(time)[0]

    def std_and_derivative(self, time: Tensor | float) -> tuple[Tensor, Tensor]:
        """Gaussian std and forward-time derivative, on interior times only."""
        t = _times(time, self.source, interior=True)
        baseline = self.sigma * (t * (1 - t)).sqrt()
        baseline_dot = self.sigma * (1 - 2 * t) / (2 * (t * (1 - t)).sqrt())
        if self.std_knots is None:
            return baseline, baseline_dot
        raw = self.std_bound * torch.tanh(self.std_knots)
        zero = torch.zeros_like(raw[:, :1])
        knots = torch.cat((zero, raw, zero), dim=1)
        segments = knots.shape[1] - 1
        scaled = t * segments
        left = scaled.floor().long().clamp(max=segments - 1)
        batch = torch.arange(self.source.shape[0], device=self.source.device)
        lo, hi = knots[batch, left], knots[batch, left + 1]
        log_scale = lo + (scaled - left) * (hi - lo)
        log_scale_dot = (hi - lo) * segments
        scale = log_scale.exp()
        return baseline * scale, scale * (baseline_dot + baseline * log_scale_dot)

    def detached(self) -> "PathBatch":
        return PathBatch(self.source.detach(), self.target.detach(), self.mask.detach(),
                         self.mean_knots.detach().clone(), self.sigma,
                         None if self.std_knots is None else self.std_knots.detach().clone(), self.std_bound)


def linear_path(source: Tensor, target: Tensor, mask: Tensor, knots: int = 5,
                sigma: float = 0.1) -> PathBatch:
    if isinstance(knots, bool) or not isinstance(knots, int) or knots < 3:
        raise ValueError("knots must be an integer >=3")
    t = torch.linspace(0, 1, knots, device=source.device, dtype=source.dtype)[1:-1]
    t = t.reshape(1, -1, *([1] * (source.ndim - 1)))
    interior = source[:, None] * (1 - t) + target[:, None] * t
    return PathBatch(source, target, mask, interior, sigma)


def sample_path(path: PathBatch, time: Tensor | float, noise: Tensor | None = None,
                direction: str = "forward") -> tuple[Tensor, Tensor]:
    """Sample a marginal and its analytic forward or reversed-time drift.

    For forward drift: m_dot + (gamma_dot-sigma²/(2*gamma))*(x-m)/gamma.
    For reversed-time drift: -m_dot + (-gamma_dot-sigma²/(2*gamma))*(x-m)/gamma.
    These are Gaussian stochastic controls, not clean-image regression targets.
    Caller must use interior times. For a linear mean they reduce exactly to
    (target-x)/(1-t) and (source-x)/t, respectively.
    """
    if direction not in {"forward", "backward", "fwd", "bwd"}:
        raise ValueError("direction must be forward or backward")
    t = _times(time, path.source, interior=True)
    mean, velocity = path.mean_and_velocity(t)
    if noise is None:
        noise = torch.randn_like(mean)
    if noise.shape != mean.shape or noise.device != mean.device:
        raise ValueError("noise must have image shape and device")
    if not bool(torch.isfinite(noise).all()):
        raise ValueError("noise must be finite")
    missing = 1 - path.mask
    std, std_dot = path.std_and_derivative(t)
    perturbation = _expand(std, mean) * noise.to(mean.dtype) * missing
    state = mean + perturbation
    if path.sigma == 0:
        drift = velocity if direction in {"forward", "fwd"} else -velocity
    else:
        correction = path.sigma**2 / (2 * std)
        if direction in {"forward", "fwd"}:
            coefficient = (std_dot - correction) / std
            drift = velocity + _expand(coefficient, mean) * perturbation
        else:
            coefficient = (-std_dot - correction) / std
            drift = -velocity + _expand(coefficient, mean) * perturbation
    return state, drift * missing


def _objective(path: PathBatch, metric: Any, times: Tensor, noise: Tensor,
               config: dict) -> tuple[Tensor, dict[str, Tensor]]:
    """Mean control energy plus Gaussian expected state cost and metric action."""
    batch, count = path.source.shape[0], times.numel()
    state_costs, variance_costs = [], []
    missing = 1 - path.mask
    knots = path.knots
    segments = knots.shape[1] - 1
    ramp_power = float(config.get("geometry_ramp_power", 0))
    # Exact integration over EVERY piecewise-linear segment. A coarser state
    # quadrature must never leave an unpenalized mean-motion segment.
    controls = [0.5 * _mean_free(((knots[:, i + 1] - knots[:, i]) * segments).square(), missing)
                for i in range(segments)]
    mean_control = torch.stack(controls, dim=1).mean()
    for index, t in enumerate(times):
        if path.std_knots is not None and path.sigma > 0:
            std, std_dot = path.std_and_derivative(t)
            baseline = path.sigma * (t * (1 - t)).sqrt()
            h = (std / baseline).log()
            h_dot = std_dot / std - (1 - 2 * t) / (2 * t * (1 - t))
            # Equivalent integrated variance-control residual after eliminating
            # the zero-boundary total derivative. Every quadrature term >=0.
            residual = (.5 * std.square() * h_dot.square()
                        + path.sigma**2 * h.sinh().square() / (2 * t * (1 - t)))
            valid = (missing.flatten(1).sum(1) > 0).to(residual.dtype)
            variance_costs.append(residual * valid)
        if metric is not None and float(config.get("state_weight", 0.1)) != 0:
            state, _ = sample_path(path, t, noise=noise[:, index])
            value = metric.state_cost(state)
            if value.shape != (batch,):
                raise ValueError("metric.state_cost must return one scalar per image")
            state_costs.append(value * t.pow(ramp_power))
    variance_control = torch.stack(variance_costs, dim=1).mean() if variance_costs else mean_control.new_zeros(())
    control = mean_control + variance_control
    state_cost = torch.stack(state_costs, dim=1).mean() if state_costs else control.new_zeros(())
    kinetic = control.new_zeros(())
    if metric is not None and float(config.get("metric_weight", 0.05)) != 0:
        step = 1 / (knots.shape[1] - 1)
        values = [metric.kinetic(knots[:, i], knots[:, i + 1], step)
                  * ((i + 0.5) * step) ** ramp_power
                  for i in range(knots.shape[1] - 1)]
        if any(value.shape != (batch,) for value in values):
            raise ValueError("metric.kinetic must return one scalar per image")
        kinetic = 0.5 * torch.stack(values, dim=1).mean()
    total = (float(config.get("control_weight", 1.0)) * control
             + float(config.get("state_weight", 0.1)) * state_cost
             + float(config.get("metric_weight", 0.05)) * kinetic)
    return total, {"control": control, "mean_control": mean_control,
                   "variance_control": variance_control, "state": state_cost, "metric": kinetic}


@torch.enable_grad()
def optimize_path(source: Tensor, target: Tensor, mask: Tensor, metric: Any,
                  config: dict | None = None) -> tuple[PathBatch, dict[str, float]]:
    """Optimize Gaussian means for the supplied *paired* conditional endpoints.

    Common-random-number midpoint quadrature is used during local optimization.
    Mean and optional variance knots receive gradients; the metric is frozen.
    When backtracking is enabled, each step must decrease the same frozen-noise
    objective. An exhausted search returns the last accepted iterate.
    Returned tensors are detached before neural bridge matching.
    """
    cfg = {} if config is None else dict(config)
    iterations = cfg.get("iterations", 5)
    points = cfg.get("quadrature_points", cfg.get("quadrature", 5))
    learning_rate = float(cfg.get("learning_rate", cfg.get("lr", 0.03)))
    backtrack_steps = cfg.get("backtrack_steps", 0)
    maximum = cfg.get("max_correction")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if isinstance(points, bool) or not isinstance(points, int) or points < 1:
        raise ValueError("quadrature_points must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    if isinstance(backtrack_steps, bool) or not isinstance(backtrack_steps, int) or backtrack_steps < 0:
        raise ValueError("backtrack_steps must be a nonnegative integer")
    if maximum is not None:
        maximum = float(maximum)
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError("max_correction must be finite and positive")
    for key in ("control_weight", "state_weight", "metric_weight"):
        value = float(cfg.get(key, {"control_weight": 1., "state_weight": .1, "metric_weight": .05}[key]))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    ramp_power = float(cfg.get("geometry_ramp_power", 0))
    if not math.isfinite(ramp_power) or ramp_power < 0:
        raise ValueError("geometry_ramp_power must be finite and nonnegative")
    initial = linear_path(source.detach(), target.detach(), mask.detach(),
                          cfg.get("knots", 5), cfg.get("sigma", 0.1))
    variable = torch.nn.Parameter(initial.mean_knots.clone())
    std_variable = (torch.nn.Parameter(torch.zeros_like(variable[:, :, 0, 0, 0, 0]))
                    if cfg.get("optimize_std", False) and initial.sigma > 0 else None)
    path = PathBatch(initial.source, initial.target, initial.mask, variable, initial.sigma,
                     std_variable, float(cfg.get("std_bound", .75)))
    times = (torch.arange(points, device=source.device, dtype=source.dtype) + .5) / points
    noise = torch.randn(source.shape[0], points, *source.shape[1:], device=source.device, dtype=source.dtype)
    variables = [variable] + ([] if std_variable is None else [std_variable])
    optimizer = torch.optim.Adam(variables, lr=learning_rate)
    with torch.no_grad():
        baseline, base_terms = _objective(path, metric, times, noise, cfg)
    if not bool(torch.isfinite(baseline)):
        raise FloatingPointError("Nonfinite initial conditional-control objective")
    best_value = float(baseline.detach())
    best_variables = [parameter.detach().clone() for parameter in variables]
    best_terms = {key: float(value.detach()) for key, value in base_terms.items()}
    initial_value, initial_terms = best_value, dict(best_terms)
    accepted_steps, attempted_steps, trial_evaluations = 0, 0, 0
    for _ in range(iterations):
        attempted_steps += 1
        optimizer.zero_grad(set_to_none=True)
        value, _ = _objective(path, metric, times, noise, cfg)
        gradients = torch.autograd.grad(value, variables)
        for parameter, gradient in zip(variables, gradients):
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("Nonfinite conditional-control gradient")
            parameter.grad = gradient
        torch.nn.utils.clip_grad_norm_(variables, float(cfg.get("gradient_clip", 10.0)), error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            if maximum is not None:
                variable.copy_(initial.mean_knots + (variable - initial.mean_knots).clamp(-maximum, maximum))
            displacement = ([parameter - anchor for parameter, anchor in zip(variables, best_variables)]
                            if backtrack_steps else ())
            for retry in range(backtrack_steps + 1):
                scale = 0.5**retry
                if retry:
                    for parameter, anchor, delta in zip(variables, best_variables, displacement):
                        parameter.copy_(anchor + scale * delta)
                candidate, terms = _objective(path, metric, times, noise, cfg)
                trial_evaluations += 1
                score = float(candidate)
                if not math.isfinite(score):
                    if backtrack_steps:
                        continue
                    raise FloatingPointError("Nonfinite optimized conditional-control objective")
                if score < best_value:
                    best_value = score
                    best_variables = [parameter.detach().clone() for parameter in variables]
                    best_terms = {key: float(item) for key, item in terms.items()}
                    accepted_steps += 1
                    optimizer.param_groups[0]["lr"] *= scale
                    break
            else:
                if backtrack_steps:
                    break
    result = PathBatch(initial.source, initial.target, initial.mask, best_variables[0], initial.sigma,
                       best_variables[1] if std_variable is not None else None, path.std_bound).detached()
    diagnostics = {"objective_before": initial_value, "objective_after": best_value,
                   "objective_improvement": initial_value - best_value,
                   "iterations": float(iterations), "attempted_steps": float(attempted_steps),
                   "accepted_steps": float(accepted_steps), "trial_evaluations": float(trial_evaluations),
                   "mean_correction_rms": float((result.mean_knots - initial.mean_knots).square().mean().sqrt())}
    diagnostics["optimized_std"] = float(std_variable is not None)
    diagnostics.update({f"{key}_before": value for key, value in initial_terms.items()})
    diagnostics.update({f"{key}_after": value for key, value in best_terms.items()})
    return result, diagnostics

