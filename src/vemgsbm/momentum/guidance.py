"""Soft geometry forces in the lossless MRI phase space."""
from collections.abc import Mapping
import math

import torch


def momentum_geometry_options(config):
    options = config.get("sampling", {}).get("momentum_geometry")
    if options is None:
        return None
    if (config["mode"] != "momentum" or config["data"].get("kind") != "paired_mri_png"
            or not (config.get("atlas") or {}).get("source_routing")
            or config.get("chart_momentum") is not None):
        raise ValueError("Momentum guidance requires the LR-routed MRI feature atlas and Haar phase space")
    defaults = dict(position_strength=.02, velocity_strength=.02, start=.25, max_acceleration=.2)
    if not isinstance(options, Mapping) or set(options) - set(defaults):
        raise ValueError("Unknown momentum geometry options")
    values = {**defaults, **options}
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in values.values()):
        raise ValueError("Momentum geometry options must be finite numbers")
    position, velocity = values["position_strength"], values["velocity_strength"]
    if (min(position, velocity) < 0 or position + velocity == 0 or not 0 <= values["start"] < 1
            or values["max_acceleration"] <= 0):
        raise ValueError("Invalid momentum geometry strength, start time or acceleration bound")
    return values


def atlas_anchor_update(codec, bound, base, mask, position, velocity, time, options):
    """Convert a chart attraction and normal damping force to the learned anchor's units."""
    if time <= options["start"]:
        return 0.

    free = 1 - mask
    free_count = free.flatten(1).sum(-1)
    with torch.enable_grad():
        image = (base + codec.decode(position)).detach().requires_grad_(True)
        speed = codec.decode(velocity).detach().requires_grad_(True)
        position_cost, velocity_cost = bound.phase_costs(image, speed)
        position_gradient, = torch.autograd.grad((position_cost * free_count).sum(), image, retain_graph=True)
        velocity_gradient, = torch.autograd.grad((velocity_cost * free_count).sum(), speed)

    ramp = ((time - options["start"]) / (1 - options["start"])) ** 2
    force = -ramp * (options["position_strength"] * position_gradient
                     + options["velocity_strength"] * velocity_gradient)
    bound_force = options["max_acceleration"]
    acceleration = bound_force * torch.tanh(force / bound_force) * free
    # The transition already integrates x'=v and a=3*(anchor-x-r*v)/r².
    return (1 - time)**2 / 3 * codec.encode(acceleration)
