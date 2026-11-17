"""Checks for lossless MLS chart motion and its kinetic bridge."""

import pytest
import torch

from neural_atlas.mls_atlas import MLSAtlas
from vemgsbm.momentum.model import AccelerationNetwork
from vemgsbm.momentum.phase import PhasePath
from vemgsbm.momentum.trivialized import ChartMomentumAtlas


@pytest.fixture
def geometry():
    # The graph y = z² bends in channel 1, with a flat base direction in channel 0.
    coeff = torch.zeros(1, 1, 3, 16)
    coeff[..., 2, 1] = .65
    chart = MLSAtlas(
        {"centers": torch.zeros(1, 16), "basis": torch.eye(16)[None, :, :1],
         "scales": torch.ones(1), "domain_lo": torch.full((1, 1), -2.),
         "domain_hi": torch.full((1, 1), 2.)},
        {"anchor_z": torch.zeros(1, 1, 1), "bandwidth": torch.ones(1, 1),
         "anchor_valid": torch.ones(1, 1, dtype=torch.bool),
         "coefficients": coeff, "exponents": torch.tensor([[0], [1], [2]])},
        {"degree": 2, "blend_radius": 1.},
    )
    chart.transform_metadata = {"mean": [0.] * 16, "scale": [1.] * 16}
    return ChartMomentumAtlas(
        {"kind": "haar_mls", "atlas": chart.snapshot(),
         "routing": {"kind": "paired_lr_kernel", "settings": {"bandwidth": 1., "temperature": 1.,
                     "candidates": 1, "correction_limit": .5},
                     "source_bank": torch.zeros(1, 16), "source_mean": torch.zeros(16),
                     "source_scale": torch.ones(16), "chart_probabilities": torch.ones(1, 1)}},
        {"warp_strength": .75, "normal_noise_ratio": .75},
    )


def test_chart_roundtrip_and_physical_velocity(geometry):
    rng = torch.Generator().manual_seed(73)
    position = torch.randn((2, 16, 3, 4, 5), generator=rng) * .2
    position[:, 0] += .35
    free = torch.ones_like(position[:, :1])
    free[:, :, 0, 0] = 0
    position = position * free
    direction = torch.zeros_like(position)
    direction[:, 0] = free[:, 0]
    frame = geometry.bind(torch.zeros_like(position), free)
    physical, velocity = frame.features(position, direction)
    torch.testing.assert_close(frame.to_body(physical), position, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(frame.to_image(torch.zeros_like(position)), torch.zeros_like(position))
    torch.testing.assert_close(physical, frame.to_image(position))
    assert (physical - position).abs().max() > .01
    assert velocity[:, 1].abs().max() > .05
    eps = 1e-3
    expected = (frame.to_image(position + eps * direction) - frame.to_image(position - eps * direction)) / (2 * eps)
    torch.testing.assert_close(velocity, expected, atol=6e-5, rtol=1e-3)
    # Mean-free spatial detail is carried unchanged by the chart extension.
    correction = physical - position
    torch.testing.assert_close(correction, frame.cells.lift(frame.cells.describe(correction)), atol=1e-7, rtol=1e-5)


def test_bridge_geometry_changes_motion_and_keeps_physical_endpoint(geometry):
    reference = torch.zeros(1, 16, 1, 2, 2)
    target = reference.clone()
    target[:, 0] = .6
    target[:, 1] = .45
    free = torch.ones_like(target[:, :1])
    frame = geometry.bind(reference, free)
    endpoint = frame.to_body(target)
    config = dict(sigma=.0, quadrature=3, iterations=8, lr=.01, max_shift=.5, backtracking_steps=5,
                  state_weight=.3, normal_weight=.1)
    path, metrics = frame.optimize(endpoint, config)
    flat = PhasePath(target, torch.zeros_like(path.shifts), .0, free)
    torch.testing.assert_close(frame.to_image(path.sample(0.)[0]), reference)
    torch.testing.assert_close(frame.to_image(path.sample(1.)[0]), target)
    assert metrics["path_after"] < metrics["path_before"]
    assert (frame.to_image(path.sample(.5)[0]) - flat.sample(.5)[0]).abs().max() > .001


def test_chart_momentum_sampler_has_no_last_position_noise(geometry):
    model = AccelerationNetwork(width=4, chart_features=True)
    base = torch.zeros(1, 1, 3, 8, 8)
    mask = torch.zeros_like(base)
    mask[:, :, 0] = 1
    with torch.no_grad():
        model.output.bias[0] = .3
    kwargs = dict(steps=1, sigma=.15, geometry=geometry)
    first = model.restore(base, mask, generator=torch.Generator().manual_seed(1), **kwargs)
    second = model.restore(base, mask, generator=torch.Generator().manual_seed(2), **kwargs)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    torch.testing.assert_close(first[:, :, 0], base[:, :, 0], atol=0, rtol=0)
    assert first[:, :, 1:].abs().max() > 0
    with pytest.raises(ValueError, match="requires its fitted MLS atlas"):
        model.restore(base, mask, steps=1)
