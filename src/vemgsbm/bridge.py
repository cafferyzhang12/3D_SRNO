"""Public reconstruction adapter preserving the established inference API."""
import math
from collections.abc import Mapping

import torch

from .network import integrate


def geometry_guidance_options(options):
    """Validate and normalize optional online MRI atlas guidance."""
    if options is None:
        return None
    if not isinstance(options, Mapping) or set(options) - {"strength", "start", "max_update"}:
        raise ValueError("geometry_guidance supports strength, start and max_update")
    values = {"strength": 0.0, "start": 0.3, "max_update": 0.002, **options}
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
           for x in values.values()):
        raise ValueError("Geometry guidance options must be finite numbers")
    if values["strength"] < 0 or not 0 <= values["start"] < 1 or values["max_update"] <= 0:
        raise ValueError("Geometry guidance requires strength >= 0, 0 <= start < 1, and max_update > 0")
    return values if values["strength"] > 0 else None


def _atlas_guidance(chart, options):
    strength, start, maximum = (options[key] for key in ("strength", "start", "max_update"))

    def guide(state, time, dt, mask):
        if time <= start:
            return state
        ramp = ((time - start) / (1 - start)) ** 2
        free = 1 - mask
        with torch.enable_grad():
            image = state.detach().requires_grad_(True)
            cost = chart.state_cost(image)
            # The cost is per patch. Match the voxel-mean units of the image
            # path action without coupling the corrections of different items.
            free_count = free.flatten(1).sum(1)
            gradient, = torch.autograd.grad((cost * free_count).sum(), image)
        update = -dt * strength * ramp * gradient
        return state + maximum * torch.tanh(update / maximum) * free

    return guide


def reconstruct(model, base, mask, mode="gsbm", geometry=None, steps=20,
                sigma=0.1, generator=None, terminal_noise=True, geometry_guidance=None):
    options = geometry_guidance_options(geometry_guidance)
    if mode == "momentum":
        if options is not None:
            raise ValueError("Online MLS guidance requires a GSBM model")
        geometry_input = {"geometry": geometry} if geometry is not None else {}
        return model.restore(base, mask, steps=steps, sigma=sigma, generator=generator, **geometry_input)
    if mode != "gsbm":
        raise ValueError("Use a new GSBM checkpoint with this isolated project")
    guidance = None
    if options is not None:
        if geometry is None:
            raise ValueError("Online MLS guidance requires the fitted MRI atlas")
        with torch.no_grad():
            guidance = _atlas_guidance(geometry.bind(base), options)
        terminal_noise = False
    return integrate(model, base, base, mask, steps, sigma, generator=generator,
                     terminal_noise=terminal_noise, state_guidance=guidance)
