"""Public reconstruction adapter preserving the established inference API."""
import math
from collections.abc import Mapping

import torch

from .network import integrate


def sample_gaussian_bridge(target, sigma_0=.03, sigma_bridge=.08, time=None,
                           initial=None, noise=None, time_min=1.e-4):
    """Paired Brownian bridge in whitened Haar units; TRAIN never sees t=0/1."""
    if not 1.e-4 <= time_min < .5:
        raise ValueError("Training times require 1e-4 <= time_min < .5")
    if any(not math.isfinite(s) or s < 0 for s in (sigma_0, sigma_bridge)):
        raise ValueError("Bridge standard deviations must be finite and nonnegative")
    t = (torch.rand(len(target), device=target.device) * (1 - 2*time_min) + time_min
         if time is None else torch.as_tensor(time, device=target.device, dtype=target.dtype))
    if not torch.isfinite(t).all() or bool(((t < time_min) | (t > 1-time_min)).any()):
        raise ValueError("Training time is outside the safe interval")
    tau = t.reshape(-1, *([1] * (target.ndim-1)))
    initial = sigma_0 * torch.randn_like(target) if initial is None else initial
    noise = torch.randn_like(target) if noise is None else noise
    if initial.shape != target.shape or noise.shape != target.shape:
        raise ValueError("Bridge states must have matching shapes")
    state = (1-tau)*initial + tau*target + sigma_bridge*(tau*(1-tau)).sqrt()*noise
    return state, t, initial


def endpoint_velocity(state, endpoint, time):
    """Forward bridge drift; only evaluated strictly before the terminal readout."""
    t = torch.as_tensor(time, dtype=state.dtype, device=state.device)
    if not torch.isfinite(t).all() or bool(((t < 0) | (t >= 1)).any()):
        raise ValueError("Endpoint drift requires 0 <= t < 1")
    return (endpoint-state) / (1-t.reshape(-1, *([1]*(state.ndim-1))))


def integrate_endpoint(predict, initial, steps=8, *, sampler_type="ode", sigma_bridge=.08,
                       sigma_infer=None, solver="euler", generator=None, noise_steps=None,
                       return_prefix=False):
    """Endpoint ODE / Brownian SDE with an explicit division-free last step.

    The requested noise-free forward drift is an endpoint ODE, not the exact
    probability-flow ODE of a noisy bridge (which also needs a score correction).
    Heun uses 2*N-1 network evaluations; Euler uses N, including N=1.
    """
    if type(steps) is not int or steps < 1 or sampler_type not in {"ode", "sde"}:
        raise ValueError("Invalid endpoint sampler or step count")
    if solver not in {"euler", "heun"} or (sampler_type == "sde" and solver != "euler"):
        raise ValueError("Heun is supported only for deterministic ODE")
    if not math.isfinite(sigma_bridge) or sigma_bridge < 0:
        raise ValueError("Invalid bridge noise")
    if sampler_type == "sde" and sigma_infer is not None and sigma_infer != sigma_bridge:
        raise ValueError("SDE sigma_infer must equal sigma_bridge")
    sigma = sigma_bridge if sampler_type == "sde" else 0.
    if noise_steps is not None and len(noise_steps) != steps-1:
        raise ValueError("Expected one noise tensor per nonterminal step")
    state, dt = initial, 1./steps
    for i in range(steps-1):
        t = i/steps
        velocity = endpoint_velocity(state, predict(state, t), t)
        proposal = state + dt*velocity
        if solver == "heun":
            next_t = (i+1)/steps  # always <1
            next_v = endpoint_velocity(proposal, predict(proposal, next_t), next_t)
            state = state + .5*dt*(velocity+next_v)
        else:
            state = proposal
        if sigma:
            noise = (torch.randn(state.shape, device=state.device, dtype=state.dtype, generator=generator)
                     if noise_steps is None else noise_steps[i])
            state = state + sigma*math.sqrt(dt)*noise
    if return_prefix:
        return state.detach(), (steps-1)/steps
    return predict(state, (steps-1)/steps)


@torch.no_grad()
def sample_induced_coupling(model, lr, base, steps=8, generator=None):
    """Frozen stage-one teacher coupling, in its checkpoint's whitening units."""
    if getattr(model, "predict_target", None) != "x1":
        raise ValueError("Induced couplings require a stage-one x1 checkpoint")
    condition = model.lr_encoder(lr)
    initial = model.initial_state(lr, base, generator)
    endpoint = integrate_endpoint(lambda x,t: model(x, condition, t)[0], initial, steps,
                                  sigma_bridge=model.sigma_bridge)
    return initial.detach(), endpoint.detach()


def geometry_guidance_options(options):
    """Validate and normalize optional online MRI atlas guidance."""
    if options is None:
        return None
    if not isinstance(options, Mapping) or set(options) - {"strength", "start", "max_update", "normal_strength"}:
        raise ValueError("geometry_guidance supports strength, normal_strength, start and max_update")
    values = {"strength": 0.0, "start": 0.3, "max_update": 0.002, **options}
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
           for x in values.values()):
        raise ValueError("Geometry guidance options must be finite numbers")
    normal_strength = values.get("normal_strength", 0.)
    if min(values["strength"], normal_strength) < 0 or not 0 <= values["start"] < 1 or values["max_update"] <= 0:
        raise ValueError("Geometry guidance requires nonnegative strengths, 0 <= start < 1, and max_update > 0")
    return values if values["strength"] > 0 or normal_strength > 0 else None


def _atlas_guidance(chart, options):
    strength, start, maximum = (options[key] for key in ("strength", "start", "max_update"))
    normal_strength = options.get("normal_strength", 0.)

    def guide(state, time, dt, mask, previous=None):
        if time <= start:
            return state
        ramp = ((time - start) / (1 - start)) ** 2
        free = 1 - mask
        with torch.enable_grad():
            image = state.detach().requires_grad_(True)
            if normal_strength:
                position, normal = chart.transition_costs(previous.detach(), image)
                cost = strength * position + normal_strength * normal
            else:
                cost = chart.state_cost(image)
            # The cost is per patch. Match the voxel-mean units of the image
            # path action without coupling the corrections of different items.
            free_count = free.flatten(1).sum(1)
            gradient, = torch.autograd.grad((cost * free_count).sum(), image)
        update = (-dt * ramp if normal_strength else -dt * strength * ramp) * gradient
        return state + maximum * torch.tanh(update / maximum) * free

    return guide


def reconstruct(model, base, mask, mode="gsbm", geometry=None, steps=20,
                sigma=0.1, generator=None, terminal_noise=True, geometry_guidance=None,
                momentum_guidance=None, native_lr=None, noise_steps=None,
                sampler_type=None, solver="euler", initial_state=None):
    options = geometry_guidance_options(geometry_guidance)
    if mode in {"haar_direct", "haar_bridge", "external_2d"}:
        if native_lr is None or bool(mask.any()) or options or momentum_guidance:
            raise ValueError("This MRI model requires native LR and paired XY reconstruction")
        if mode == "external_2d":
            return model.restore(native_lr, base, steps, sigma, generator)
        extra = {"noise_steps": noise_steps} if noise_steps is not None else {}
        if sampler_type is not None:
            extra.update(sampler_type=sampler_type, solver=solver)
        if initial_state is not None:
            extra["initial_state"] = initial_state
        return model.restore(native_lr, base, steps, sigma, generator, atlas=geometry, **extra)
    if mode == "momentum":
        if options is not None:
            raise ValueError("Online MLS guidance requires a GSBM model")
        geometry_input = {"geometry": geometry} if geometry is not None else {}
        if momentum_guidance is not None:
            if geometry is None:
                raise ValueError("Momentum MLS guidance requires a fitted MRI atlas")
            geometry_input["geometry_guidance"] = momentum_guidance
        return model.restore(base, mask, steps=steps, sigma=sigma, generator=generator, **geometry_input)
    if mode != "gsbm":
        raise ValueError("Use a new GSBM checkpoint with this isolated project")
    if momentum_guidance is not None:
        raise ValueError("Momentum MLS guidance requires a momentum model")
    guidance = None
    if options is not None:
        if geometry is None:
            raise ValueError("Online MLS guidance requires the fitted MRI atlas")
        with torch.no_grad():
            guidance = _atlas_guidance(geometry.bind(base), options)
        terminal_noise = False
    tangent_aware = options is not None and options.get("normal_strength", 0.) > 0
    return integrate(model, base, base, mask, steps, sigma, generator=generator,
                     terminal_noise=terminal_noise, state_guidance=None if tangent_aware else guidance,
                     transition_guidance=guidance if tangent_aware else None)
