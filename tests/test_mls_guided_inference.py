"""Regression checks for bounded online atlas guidance in the forward sampler."""

import torch
import pytest

from vemgsbm.bridge import reconstruct
from vemgsbm.network import DriftNetwork
from vemgsbm.runtime import build_models, sampling_noise


class RecordingGeometry:
    def __init__(self):
        self.bind_inputs = []
        self.queried = []

    def bind(self, base):
        self.bind_inputs.append(base.clone())
        return self

    def state_cost(self, state):
        self.queried.append(state.detach().clone())
        return state.square().flatten(1).mean(1)


def zero_drift(state, *_):
    return torch.zeros_like(state)


def test_online_geometry_guidance_is_lr_conditioned_bounded_and_interior_only():
    base = torch.full((1, 1, 3, 4, 4), 0.5)
    mask = torch.zeros_like(base)
    mask[..., 0] = 1
    geometry = RecordingGeometry()
    options = {"strength": 10, "start": 0.3, "max_update": 0.002}
    with torch.no_grad():
        result = reconstruct(zero_drift, base, mask, geometry=geometry, geometry_guidance=options,
                             sigma=0, steps=4)
    assert len(geometry.bind_inputs) == 1
    torch.testing.assert_close(geometry.bind_inputs[0], base)
    assert len(geometry.queried) == 2  # at t=.5 and .75, excluding the terminal state
    torch.testing.assert_close(geometry.queried[0], base)
    torch.testing.assert_close(result[mask.bool()], base[mask.bool()])
    assert torch.all(result[~mask.bool()] < base[~mask.bool()])
    assert (result - base).abs().max() <= 2 * options["max_update"] + 1e-7
    assert not result.requires_grad


def test_guidance_ablation_preserves_rng_and_has_no_last_step_noise():
    base = torch.full((1, 1, 3, 4, 4), 0.5)
    mask = torch.zeros_like(base)
    geometry = RecordingGeometry()
    plain_generator = torch.Generator().manual_seed(133)
    guided_generator = torch.Generator().manual_seed(133)
    plain = reconstruct(zero_drift, base, mask, sigma=0.1, steps=1,
                        generator=plain_generator, terminal_noise=False)
    guided = reconstruct(zero_drift, base, mask, geometry=geometry, sigma=0.1, steps=1,
                         generator=guided_generator, geometry_guidance={"strength": 0.02})
    torch.testing.assert_close(plain, guided)
    assert torch.equal(plain_generator.get_state(), guided_generator.get_state())
    assert not geometry.queried
    reconstruct(zero_drift, base, mask, geometry=geometry, sigma=0, steps=4,
                geometry_guidance={"strength": 0})
    assert len(geometry.bind_inputs) == 1


def test_checkpoint_requires_mls_payload_only_when_guidance_is_enabled():
    cfg = {"mode": "gsbm", "data": {"kind": "paired_mri_png"}, "atlas": {"source_routing": True},
           "sampling": {"geometry_guidance": {"strength": 0}}, "model": {"width": 2}}
    checkpoint = {"config": cfg, "model": DriftNetwork(2).state_dict()}
    assert build_models(checkpoint, "cpu")[1] is None
    cfg["sampling"]["geometry_guidance"]["strength"] = 0.02
    with pytest.raises(ValueError, match="missing its MRI MLS geometry"):
        build_models(checkpoint, "cpu")


def test_transition_guidance_receives_previous_state_and_preserves_tangent_motion():
    class Chart(RecordingGeometry):
        def transition_costs(self, previous, state):
            self.queried.append((previous.clone(), state.detach().clone()))
            normal = (state[..., 1] - previous[..., 1]).square().flatten(1).mean(1)
            return torch.zeros_like(normal), normal

    def drift(state, *_):
        return torch.ones_like(state)

    base = torch.zeros((1, 1, 3, 4, 4))
    chart = Chart()
    mask = torch.zeros_like(base)
    mask[..., 0] = 1
    result = reconstruct(drift, base, mask, geometry=chart, sigma=0., steps=4,
                         geometry_guidance={"normal_strength": 0.1, "max_update": 0.1})
    assert len(chart.queried) == 2
    before, after = chart.queried[0]
    torch.testing.assert_close(after[..., 1] - before[..., 1], torch.full_like(after[..., 1], .25))
    torch.testing.assert_close(result[..., 0], base[..., 0])
    torch.testing.assert_close(result[..., 2:], torch.ones_like(result[..., 2:]))
    assert (result[..., 1] < 1.).all()


def test_inference_sigma_is_separate_and_validated():
    cfg = {"sampling": {"sigma": .10}}
    assert sampling_noise(cfg) == .10
    cfg["sampling"]["inference_sigma"] = .05
    assert sampling_noise(cfg) == .05
    assert cfg["sampling"]["sigma"] == .10
    cfg["sampling"]["inference_sigma"] = float("nan")
    with pytest.raises(ValueError, match="Inference diffusion sigma"):
        sampling_noise(cfg)
