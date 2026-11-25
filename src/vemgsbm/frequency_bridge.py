"""Frequency-split Gaussian bridge and score-corrected probability flow.

The data pairing is fixed. These are supervised Gaussian bridge marginals,
not a claim that an unpaired Schrödinger coupling has been optimized.
"""
import math

import torch

from .models import batch_time


def _sigma(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError("Bridge sigma must be finite and nonnegative")


def bridge_sigma(time, sigma_max=.08):
    """Exactly zero at both boundaries, in whitened high-frequency units."""
    _sigma(sigma_max)
    t = torch.as_tensor(time)
    if not t.is_floating_point():
        t = t.float()
    if not torch.isfinite(t).all() or bool(((t < 0) | (t > 1)).any()):
        raise ValueError("Bridge time must lie in [0,1]")
    return sigma_max * (t * (1-t)).sqrt()


def sample_frequency_bridge(target, sigma_max=.08, time=None, *, initial=None,
                            noise=None, endpoint_fraction=.1):
    """Sample continuous TRAIN times; return (Xt, t, supervised epsilon).

    Channel zero is LL and receives no noise. Endpoint examples supervise the
    deployment start and terminal readout; epsilon is zero at those endpoints.
    """
    _sigma(sigma_max)
    if target.ndim != 5 or target.shape[1] != 16:
        raise ValueError("Frequency bridge expects B16ZYX Haar states")
    if not 0 <= endpoint_fraction <= 1:
        raise ValueError("Invalid endpoint sampling fraction")
    if time is None:
        t = torch.rand(len(target), device=target.device, dtype=target.dtype)
        endpoint = torch.rand_like(t) < endpoint_fraction
        t = torch.where(endpoint, (torch.rand_like(t) >= .5).to(t), t)
    else:
        t = batch_time(time, target)
    tau = t[:, None, None, None, None]
    initial = torch.zeros_like(target) if initial is None else initial
    noise = torch.randn_like(target) if noise is None else noise
    if initial.shape != target.shape or noise.shape != target.shape:
        raise ValueError("Bridge state and noise shapes must match")
    noise = torch.cat((torch.zeros_like(noise[:, :1]), noise[:, 1:]), 1)
    state = (1-tau)*initial + tau*target + bridge_sigma(tau, sigma_max)*noise
    active = (tau > 0) & (tau < 1)
    return state, t, noise * active


def frequency_velocity(state, endpoint, epsilon, time, sigma_max=.08, *,
                       temperature=0., time_min=1.e-3):
    """Marginal current velocity plus controlled-diffusion score correction.

    For X=tY+s*sqrt(t(1-t))*eps, the exact conditional flow target is
    Y+s*(1-2t)/(2sqrt(t(1-t)))*eps. Its conditional expectation is the PF
    velocity. A controlled diffusion g(t)=temperature*s*sqrt(t(1-t)) adds
    g²*score/2, with score=-E[eps|X]/(s*sqrt(t(1-t))).
    """
    _sigma(sigma_max)
    if not math.isfinite(temperature) or not 0 <= temperature <= .3:
        raise ValueError("SDE temperature must lie in [0,.3]")
    if not 0 < time_min < .5:
        raise ValueError("time_min must lie in (0,.5)")
    t = batch_time(time, state).clamp(time_min, 1-time_min)[:, None, None, None, None]
    root = (t*(1-t)).sqrt()
    coefficient = sigma_max * (1-2*t)/(2*root)
    # g(t)^2*score/2; no division by sigma, including sigma_max=0.
    coefficient = coefficient - .5*temperature**2*sigma_max*root
    hf = endpoint[:, 1:] + coefficient*epsilon[:, 1:]
    return torch.cat((endpoint[:, :1], hf), 1)


def integrate_frequency_bridge(predict, initial, steps=8, *, sigma_max=.08,
                               sampler_type="ode", solver="heun", temperature=.1,
                               generator=None, noise_steps=None, curvature=None,
                               shrink=None, return_prefix=False, time_min=1.e-3):
    """Heun PF-ODE / low-temperature score-corrected SDE, safe at endpoints.

    Time integration ends before one, followed by a division-free endpoint
    prediction. ``initial`` is a state at ``time_min``. Zero initialization
    gives a deterministic reconstruction; a sampled interior initial state is
    needed when using the PF-ODE as a generative sampler from a point source.
    LL is predicted solely from LR and stays entirely deterministic.
    """
    _sigma(sigma_max)
    if type(steps) is not int or steps < 1 or initial.ndim != 5 or initial.shape[1] != 16:
        raise ValueError("Expected positive steps and B16ZYX initial state")
    if sampler_type not in {"ode", "sde"} or solver not in {"euler", "heun"}:
        raise ValueError("Unknown frequency bridge sampler or solver")
    if not math.isfinite(temperature) or not 0 <= temperature <= .3:
        raise ValueError("SDE temperature must lie in [0,.3]")
    if not 0 < time_min < .5:
        raise ValueError("time_min must lie in (0,.5)")
    if not torch.isfinite(initial).all():
        raise ValueError("Initial state must be finite")
    if noise_steps is not None and len(noise_steps) != steps-1:
        raise ValueError("Expected steps-1 integration noise tensors")
    state = initial
    dt = (1-2*time_min)/steps
    temp = temperature if sampler_type == "sde" else 0.
    for index in range(steps-1):
        t = time_min + index*dt
        clean, eps = predict(state, t)
        if sigma_max:
            velocity = frequency_velocity(state, clean, eps, t, sigma_max,
                                          temperature=temp, time_min=time_min)
        else:
            # Exact deterministic bridge drift when interpolation noise is off.
            velocity = (clean-state)/(1-t)
        proposal = state + dt*velocity
        if solver == "heun":
            next_t = t+dt
            next_clean, next_eps = predict(proposal, next_t)
            next_velocity = (frequency_velocity(proposal, next_clean, next_eps, next_t,
                                sigma_max, temperature=temp, time_min=time_min)
                             if sigma_max else (next_clean-proposal)/(1-next_t))
            proposal = state + .5*dt*(velocity+next_velocity)
        if temp and sigma_max:
            noise = (torch.randn(state.shape, dtype=state.dtype, device=state.device,
                                 generator=generator) if noise_steps is None else noise_steps[index])
            if noise.shape != state.shape or not torch.isfinite(noise).all():
                raise ValueError("Invalid integration noise")
            g = temp*sigma_max*math.sqrt(t*(1-t))
            noise = torch.cat((torch.zeros_like(noise[:, :1]), noise[:, 1:]), 1)
            proposal = proposal + g*math.sqrt(dt)*noise
            if curvature is not None:
                h = curvature(state, t)
                # Smooth vector clipping leaves finite derivatives.
                norm = h.square().sum(1, keepdim=True).sqrt().clamp_min(1.e-8)
                h = h/(1+norm)
                h = torch.cat((torch.zeros_like(h[:, :1]), h[:, 1:]), 1)
                proposal = proposal + .5*dt*g*g*h
        # Independent deterministic LL, even if a callback carries HF noise.
        proposal = torch.cat(((t+dt)*clean[:, :1], proposal[:, 1:]), 1)
        if shrink is not None:
            proposal = shrink(proposal, t+dt)
            proposal = torch.cat(((t+dt)*clean[:, :1], proposal[:, 1:]), 1)
        if not torch.isfinite(proposal).all():
            raise FloatingPointError("Nonfinite frequency bridge trajectory")
        state = proposal
    last_t = time_min+(steps-1)*dt
    if return_prefix:
        return state.detach(), last_t
    clean, _ = predict(state, last_t)
    if shrink is not None:
        shrunk = shrink(clean, 1.)
        clean = torch.cat((clean[:, :1], shrunk[:, 1:]), 1)
    if not torch.isfinite(clean).all():
        raise FloatingPointError("Nonfinite frequency bridge endpoint")
    return clean
