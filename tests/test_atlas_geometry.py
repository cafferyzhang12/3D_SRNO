"""Source-only atlas routing, trusted MLS derivatives and matching invariants."""

import io
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from neural_atlas.mls_atlas import MLSAtlas
from vemgsbm.atlas_geometry import (BoundMRIAtlas, BoundSourceMRIAtlas, MRIAtlasGeometry,
                                    _fit_source_routing, _routing_options, collect_features)
from vemgsbm.metric import Representation


@pytest.fixture(autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class FlattenEncoder(nn.Module):
    feature_dim = 2

    def forward(self, images):
        return images.flatten(1)


def image(values):
    return torch.tensor(values, dtype=torch.float32).reshape(-1, 1, 1, 1, 2)


def fixture_payload(curvature=0., source_routing=True):
    meta = {"version": "test", "centers": torch.tensor([[0., -1.], [0., 1.]]),
            "basis": torch.tensor([[[1.], [0.]], [[1.], [0.]]]),
            "scales": torch.ones(2), "domain_lo": torch.full((2, 1), -2.),
            "domain_hi": torch.full((2, 1), 2.)}
    coefficients = torch.zeros(2, 2, 3, 2)
    coefficients[..., 2, 1] = curvature
    buffers = {"anchor_z": torch.zeros(2, 2, 1), "bandwidth": torch.ones(2, 2),
               "anchor_valid": torch.tensor([[True, False], [True, False]]),
               "coefficients": coefficients, "exponents": torch.tensor([[0], [1], [2]])}
    atlas = MLSAtlas(meta, buffers, {"degree": 2, "blend_radius": 1., "support_limit": .9})
    atlas.transform_metadata = {"mean": [0., 0.], "scale": [1., 1.]}
    encoder = Representation(width=2, feature_dim=2)
    payload = {"kind": "mls", "atlas": atlas.snapshot(), "encoder_config": encoder.export_config(),
               "encoder": encoder.state_dict()}
    if source_routing:
        # Reverse the geometric HR ordering deliberately. Correct routing must
        # follow the paired LR->HR relation, not locate the current LR as HR.
        payload["routing"] = _fit_source_routing(
            atlas, torch.tensor([[0., 1.], [0., -1.]]), torch.tensor([[-1., 0.], [1., 0.]]),
            _routing_options(dict(candidates=2, bandwidth=.15, normal_weight=3.)))
    return payload


def fixture_geometry(**kwargs):
    geometry = MRIAtlasGeometry(fixture_payload(**kwargs))
    geometry.encoder = FlattenEncoder()
    return geometry


def test_paired_lr_routes_to_corresponding_hr_charts_and_stays_fixed():
    geometry = fixture_geometry()
    bound = geometry.bind(image([[-1., 0.], [1., 0.]]))
    assert isinstance(bound, BoundSourceMRIAtlas)
    assert bound.chart_ids[:, 0].tolist() == [1, 0]
    # HR chart labels intentionally overlap, even when LR neighbors are exact.
    assert (bound.chart_prior[:, 0] > .7).all()
    on_target = image([[0., 1.], [0., -1.]])
    opposite = image([[0., -1.], [0., 1.]])
    assert (bound.state_cost(on_target) < bound.state_cost(opposite)).all()
    ids, prior = bound.chart_ids.clone(), bound.chart_prior.clone()
    bound.state_cost(opposite)
    torch.testing.assert_close(bound.chart_ids, ids, rtol=0, atol=0)
    torch.testing.assert_close(bound.chart_prior, prior, rtol=0, atol=0)


def test_legacy_payload_keeps_hr_bound_chart_behavior():
    geometry = fixture_geometry(source_routing=False)
    assert not geometry.source_routing
    bound = geometry.bind(image([[0., 1.], [0., -1.]]))
    assert isinstance(bound, BoundMRIAtlas)
    assert bound.chart_ids.tolist() == [1, 0]
    torch.testing.assert_close(bound.state_cost(image([[0., 1.], [0., -1.]])), torch.zeros(2))


def test_trusted_decoder_jacobian_matches_autograd_and_normal_correction_is_bounded():
    geometry = fixture_geometry(curvature=20.)
    bound = geometry.bind(image([[-1., 0.]]))
    ids = torch.tensor([1])
    for position in (.01, .4, 1.6, 8.):
        coordinate = torch.tensor([[position]], requires_grad=True)
        decoded, jacobian, support = bound._trusted_decode(ids, coordinate)
        numerical = torch.autograd.functional.jacobian(
            lambda z: bound._trusted_decode(ids, z, derivative=False)[0], coordinate)[0, :, 0]
        torch.testing.assert_close(jacobian[0], numerical, atol=2e-5, rtol=2e-5)
        linear = geometry.atlas.centers[ids] + torch.tensor([[position, 0.]])
        limit = geometry.routing_settings["correction_limit"] * geometry.atlas.scales[ids] * support
        assert ((decoded - linear).norm(dim=-1) <= limit + 1e-6).all()
    assert support.item() < 1e-6  # Extrapolated normal polynomial is suppressed.


def test_state_potential_is_smooth_and_ood_gradient_does_not_disappear(monkeypatch):
    geometry = fixture_geometry(curvature=.3)
    bound = geometry.bind(image([[0., 0.]]))  # Both chart priors are active.
    states = image([[.2, .0]]).requires_grad_()
    cost = bound.state_cost(states).sum()
    gradient = torch.autograd.grad(cost, states)[0]
    shifted = states.detach().clone().requires_grad_()
    shifted.data[..., 0] += 1e-4
    other = torch.autograd.grad(bound.state_cost(shifted).sum(), shifted)[0]
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, other, atol=5e-4, rtol=5e-3)
    def disallow_analytic_jacobian(*args, **kwargs):
        raise AssertionError("State-only guidance should not build an analytic Jacobian")
    monkeypatch.setattr(geometry.atlas, "decode_and_jacobian", disallow_analytic_jacobian)
    far = image([[50., 50.]]).requires_grad_()
    far_cost = bound.state_cost(far)
    far_gradient = torch.autograd.grad(far_cost.sum(), far)[0]
    assert far_cost.item() > cost.item()
    assert torch.isfinite(far_gradient).all() and far_gradient.norm() > 0


def test_gram_metric_distinguishes_true_normal_and_tangent_and_has_positive_floor():
    geometry = fixture_geometry()
    bound = geometry.bind(image([[-1., 0.]]))
    local = bound._local_geometry(torch.tensor([[0., 1.]]))
    tangent = bound._direction_energy(local, torch.tensor([[1., 0.]]))
    normal = bound._direction_energy(local, torch.tensor([[0., 1.]]))
    assert normal.item() > 2 * tangent.item()
    assert tangent.item() >= .5 * geometry.routing_settings["metric_floor"]
    far = bound._local_geometry(torch.tensor([[30., 30.]]))
    fallback = bound._direction_energy(far, torch.tensor([[1., 0.]]))
    torch.testing.assert_close(fallback, torch.tensor([.525]), atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="dt"):
        bound.kinetic(image([[0., 1.]]), image([[.1, 1.]]), 0.)


def test_shared_drift_costs_backpropagate_only_error_and_match_individual_calls():
    geometry = fixture_geometry(curvature=.2)
    bound = geometry.bind(image([[-1., 0.], [1., 0.]]))
    state = image([[.3, 1.], [-.4, -1.]]).requires_grad_()
    error1 = image([[.4, .2], [.3, -.1]]).requires_grad_()
    error2 = image([[.2, .7], [-.1, .3]]).requires_grad_()
    first, second = bound.drift_costs(state, error1, error2)
    assert first.shape == second.shape == (2,)
    torch.testing.assert_close(first, bound.drift_cost(state, error1))
    torch.testing.assert_close(second, bound.drift_cost(state, error2))
    (first.mean() + second.mean()).backward()
    assert state.grad is None
    assert error1.grad.norm() > 0 and error2.grad.norm() > 0
    assert torch.isfinite(error1.grad).all() and torch.isfinite(error2.grad).all()
    assert all(parameter.grad is None for parameter in geometry.parameters())


def test_paired_feature_collection_preserves_order_and_limit_across_batches():
    def check_errors(error):
        if error:
            raise RuntimeError(error)
    context = SimpleNamespace(device="cpu", is_distributed=False, check_errors=check_errors)
    loader = [{"hr": image([[1., 2.], [3., 4.]]), "base": image([[11., 12.], [13., 14.]])},
              {"hr": image([[5., 6.], [7., 8.]]), "base": image([[15., 16.], [17., 18.]])}]
    targets, sources = collect_features(FlattenEncoder(), loader, 3, context, paired=True)
    torch.testing.assert_close(targets, torch.tensor([[1., 2.], [3., 4.], [5., 6.]]))
    torch.testing.assert_close(sources, targets + 10)
    torch.testing.assert_close(collect_features(FlattenEncoder(), loader, 3, context), targets)
    empty = collect_features(FlattenEncoder(), loader, 0, context, paired=True)
    assert empty[0].shape == empty[1].shape == (0, 2)


def test_routing_snapshot_is_portable_reproducible_and_validates_corrupt_bank():
    payload = fixture_payload()
    torch.manual_seed(13)
    again = fixture_payload()
    assert payload["routing"]["sha256"] == again["routing"]["sha256"]
    assert payload["routing"]["source_bank"].shape == (2, 2)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    restored = MRIAtlasGeometry(torch.load(buffer, weights_only=True))
    restored.train()
    assert not restored.encoder.training and not restored.atlas.training
    assert all(not parameter.requires_grad for parameter in restored.parameters())
    payload["routing"]["source_scale"][0] = 0
    with pytest.raises(ValueError, match="routing bank"):
        MRIAtlasGeometry(payload)
    with pytest.raises(ValueError, match="positive"):
        _routing_options({"bandwidth": 0.})
    with pytest.raises(ValueError, match="Unknown"):
        _routing_options({"typo": 1})
