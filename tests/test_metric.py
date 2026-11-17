import io

import pytest
import torch
from torch import nn

from vemgsbm.metric import (DataMetric, Representation, fit_metric,
                            metric_from_payload, metric_payload,
                            representation_loss)


@pytest.fixture(autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class FlattenEncoder(nn.Module):
    def forward(self, image):
        return image.flatten(1)


def simple_metric():
    return DataMetric(FlattenEncoder(), torch.tensor([[-1., 0.], [1., 0.]]),
                      torch.full((2, 2), 0.1), torch.full((2,), 0.5),
                      torch.zeros(2), torch.ones(2), floor=0.05)


def image(features):
    return torch.tensor(features, dtype=torch.float32).reshape(-1, 1, 1, 1, 2)


def test_rbf_land_metric_is_positive_bounded_and_data_dependent():
    metric = simple_metric()
    features = torch.tensor([[-1., 0.], [0., 0.], [5., 5.]])
    values = metric.diagonal(features)
    assert values.shape == features.shape
    assert torch.isfinite(values).all() and (values > 0).all()
    assert (values >= 1e-3).all() and (values <= 1e3).all()
    assert not torch.allclose(values[0], values[2])
    assert values[2].mean() > values[0].mean()


def test_off_support_state_cost_and_gradient_do_not_underflow():
    metric = simple_metric()
    near = image([[-1., 0.]])
    far = image([[30., 30.]]).requires_grad_()
    assert metric.state_cost(far).item() > metric.state_cost(near).item()
    metric.state_cost(far).sum().backward()
    assert torch.isfinite(far.grad).all()
    assert far.grad.norm() > 0


def test_metric_kinetic_matches_explicit_feature_velocity_and_backpropagates():
    metric = simple_metric()
    first = image([[-0.8, 0.3], [0.2, 0.4]])
    second = image([[0.2, 0.6], [0.8, 1.3]]).requires_grad_()
    dt = torch.tensor([0.1, 0.2])
    a, b = metric.encode(first), metric.encode(second)
    expected = (((b - a) / dt[:, None]).square() * metric.diagonal((a + b) / 2)).mean(1)
    actual = metric.kinetic(first, second, dt)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    metric_gradient = second.grad.clone()
    second.grad.zero_()
    ((metric.encode(second) - a) / dt[:, None]).square().mean(1).sum().backward()
    assert not torch.allclose(metric_gradient, second.grad)
    with pytest.raises(ValueError, match="dt"):
        metric.kinetic(first, second, 0)


def test_representation_self_supervision_learns_and_preserves_odd_shape():
    torch.manual_seed(1)
    model = Representation(width=2, feature_dim=4)
    hr = torch.rand(4, 1, 5, 13, 15)
    assert model.reconstruct(hr).shape == hr.shape
    features = model(hr)
    assert features.shape == (4, 4) and features.std(0).mean() > 0
    terms = representation_loss(model, hr)
    assert terms["variance"].item() > 0  # anti-collapse regulariser is active
    assert all(torch.isfinite(value) for value in terms.values())
    terms["loss"].backward()
    assert model.feature_head[-1].weight.grad.norm() > 0
    assert model.encoder[0].weight.grad.norm() > 0
    assert model.decoder[-1].weight.grad.norm() > 0


def test_single_item_representation_loss_finite():
    model = Representation(width=2, feature_dim=4)
    terms = representation_loss(model, torch.rand(1, 1, 3, 8, 8))
    assert all(torch.isfinite(value) for value in terms.values())
    terms["loss"].backward()


def test_fit_reproducible_roundtrip_and_frozen_input_gradients():
    torch.manual_seed(7)
    encoder = Representation(width=2, feature_dim=4)
    with torch.no_grad():
        features = encoder(torch.rand(12, 1, 3, 8, 8))
    first = fit_metric(encoder, features, clusters=3, seed=8)
    second = fit_metric(encoder, [features[:5], features[5:]], clusters=3, seed=8)
    torch.testing.assert_close(first.centers, second.centers, rtol=0, atol=0)
    assert all(parameter.requires_grad for parameter in encoder.parameters())
    assert all(not parameter.requires_grad for parameter in first.encoder.parameters())
    volume = torch.rand(2, 1, 3, 8, 8).requires_grad_()
    first.state_cost(volume).sum().backward()
    assert volume.grad.norm() > 0 and torch.isfinite(volume.grad).all()
    assert all(parameter.grad is None for parameter in first.encoder.parameters())
    buffer = io.BytesIO()
    torch.save(metric_payload(first), buffer)
    buffer.seek(0)
    restored = metric_from_payload(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(first.state_cost(volume), restored.state_cost(volume), rtol=0, atol=0)
    first.train()
    assert not first.encoder.training


def test_constant_feature_bank_remains_finite_and_invalid_statistics_rejected():
    metric = fit_metric(Representation(width=2, feature_dim=4), torch.ones(5, 4), clusters=8)
    assert torch.isfinite(metric.diagonal(torch.ones(1, 4))).all()
    assert torch.isfinite(metric.state_cost_features(torch.ones(1, 4))).all()
    with pytest.raises(ValueError, match="positive"):
        DataMetric(FlattenEncoder(), torch.zeros(1, 2), torch.ones(1, 2),
                   torch.ones(1), torch.zeros(2), torch.ones(2), floor=0)
    with pytest.raises(ValueError, match="finite"):
        fit_metric(Representation(width=2, feature_dim=4), torch.full((3, 4), float("nan")))


def test_real_data_metric_bends_conditional_path_toward_observed_support():
    from vemgsbm.paths import linear_path, optimize_path

    # The third observed support centre lies above the straight source-target
    # chord. No named manifold or analytic geodesic is supplied to the optimizer.
    metric = DataMetric(FlattenEncoder(), torch.tensor([[-1., 0.], [0., 1.], [1., 0.]]),
                        torch.full((3, 2), 0.05), torch.full((3,), 0.35),
                        torch.zeros(2), torch.ones(2))
    source, target = image([[-1., 0.]]), image([[1., 0.]])
    mask = torch.zeros_like(source)
    torch.manual_seed(12)
    path, diagnostics = optimize_path(source, target, mask, metric, {
        "iterations": 30, "knots": 5, "quadrature_points": 5,
        "sigma": 0.02, "learning_rate": 0.04,
        "state_weight": 3., "metric_weight": 0.1,
    })
    baseline = linear_path(source, target, mask, knots=5, sigma=0.02)
    assert diagnostics["objective_after"] < diagnostics["objective_before"]
    assert not torch.allclose(path.mean_knots, baseline.mean_knots)
    # A true gradient-induced geometric path, with exact original endpoints.
    assert path.mean_knots.flatten(2)[0, :, 1].max() > 0.01
    torch.testing.assert_close(path.mean(0), source, rtol=0, atol=0)
    torch.testing.assert_close(path.mean(1), target, rtol=0, atol=0)
