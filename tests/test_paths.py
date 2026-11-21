"""Numerical contracts for conditional control and Gaussian drift matching."""
import math

import pytest
import torch
from torch import nn

from vemgsbm.paths import PathBatch, linear_path, optimize_path, sample_path, _objective


def images(batch=2):
    torch.manual_seed(17)
    source = torch.rand(batch, 1, 2, 3, 4, dtype=torch.float64)
    target = torch.rand_like(source)
    return source, target, torch.zeros_like(source)


def test_linear_mean_endpoints_and_velocity_are_exact():
    source, target, mask = images()
    path = linear_path(source, target, mask, knots=5, sigma=.2)
    assert torch.equal(path.mean(0), source)
    assert torch.equal(path.mean(1), target)
    t = torch.tensor([.17, .87], dtype=source.dtype)
    mean, velocity = path.mean_and_velocity(t)
    weight = t[:, None, None, None, None]
    torch.testing.assert_close(mean, (1 - weight) * source + weight * target)
    torch.testing.assert_close(velocity, target - source)


def test_gaussian_forward_and_reversed_controls_equal_brownian_bridge():
    source, target, mask = images()
    path = linear_path(source, target, mask, sigma=.3)
    t = torch.tensor([.21, .73], dtype=source.dtype)
    noise = torch.randn_like(source)
    state, forward = sample_path(path, t, noise, "forward")
    same_state, backward = sample_path(path, t, noise, "backward")
    assert torch.equal(state, same_state)
    weight = t[:, None, None, None, None]
    torch.testing.assert_close(forward, (target - state) / (1 - weight))
    torch.testing.assert_close(backward, (source - state) / weight)
    mean = path.mean(t)
    torch.testing.assert_close(state - mean, .3 * (weight * (1 - weight)).sqrt() * noise)


def test_piecewise_mean_velocity_matches_finite_difference():
    source, target, mask = images()
    path = linear_path(source, target, mask, sigma=.1)
    path.mean_knots[:, 1] += .2
    t, dt = torch.tensor([.31, .83], dtype=source.dtype), 1e-6
    _, velocity = path.mean_and_velocity(t)
    numeric = (path.mean(t + dt) - path.mean(t - dt)) / (2 * dt)
    torch.testing.assert_close(velocity, numeric, rtol=1e-8, atol=1e-8)


def test_masked_coordinates_are_fixed_in_every_mean_state_and_drift():
    source, target, mask = images()
    mask[..., 0] = 1
    target = target * (1 - mask) + source * mask
    path = linear_path(source, target, mask)
    path.mean_knots += 4
    for direction in ("forward", "backward"):
        state, drift = sample_path(path, .4)
        assert torch.equal(state * mask, source * mask)
        assert torch.count_nonzero(drift * mask) == 0
    assert torch.equal(path.mean(.8) * mask, source * mask)


class TargetCost(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(.8))

    def state_cost(self, x):
        return (x - self.offset).square().flatten(1).mean(1)

    def kinetic(self, x0, x1, dt):
        return ((x1 - x0) / dt).square().flatten(1).mean(1)


def test_conditional_optimization_reduces_cost_bends_path_and_not_endpoints():
    torch.manual_seed(4)
    source = torch.zeros(2, 1, 2, 3, 4)
    target = torch.zeros_like(source)
    mask = torch.zeros_like(source)
    metric = TargetCost()
    path, diag = optimize_path(source, target, mask, metric, dict(
        knots=5, iterations=25, learning_rate=.04, sigma=.03,
        state_weight=5, metric_weight=.01, quadrature_points=8))
    assert diag["objective_after"] < diag["objective_before"] - .1
    assert path.mean(.5).mean() > .1
    assert torch.equal(path.mean(0), source)
    assert torch.equal(path.mean(1), target)
    assert not path.mean_knots.requires_grad
    assert metric.offset.grad is None


def test_changing_geometry_changes_the_optimized_path():
    source = torch.zeros(1, 1, 2, 2, 2)
    mask = torch.zeros_like(source)
    cfg = dict(iterations=12, sigma=0., state_weight=5., metric_weight=0.,
               quadrature_points=8, learning_rate=.04)
    positive = TargetCost()
    negative = TargetCost()
    with torch.no_grad():
        negative.offset.fill_(-.8)
    torch.manual_seed(1)
    first, _ = optimize_path(source, source, mask, positive, cfg)
    torch.manual_seed(1)
    second, _ = optimize_path(source, source, mask, negative, cfg)
    assert first.mean(.5).mean() > .1
    assert second.mean(.5).mean() < -.1


def test_backtracking_recovers_atlas_descent_after_an_oversized_adam_step():
    source = torch.zeros(1, 1, 2, 2, 2)
    mask = torch.zeros_like(source)
    mask[..., 0] = 1
    config = dict(iterations=1, sigma=0., state_weight=.03, metric_weight=.01,
                  quadrature_points=5, learning_rate=.2, max_correction=.15)
    atlas = TargetCost()
    original, stalled = optimize_path(source, source, mask, atlas, config)
    path, recovered = optimize_path(source, source, mask, atlas, dict(config, backtrack_steps=8))
    assert stalled["accepted_steps"] == 0
    assert torch.equal(original.mean_knots, torch.zeros_like(original.mean_knots))
    assert recovered["trial_evaluations"] > recovered["accepted_steps"] == 1
    assert recovered["objective_after"] < recovered["objective_before"]
    assert recovered["mean_correction_rms"] > 0
    assert torch.equal(path.mean(0), source)
    assert torch.equal(path.mean(1), source)
    middle = path.mean(.5)
    assert torch.count_nonzero(middle * mask) == 0
    assert (middle * (1 - mask)).mean() > 0
    assert atlas.offset.grad is None


def test_metric_kinetic_energy_is_actually_used_in_path_objective():
    source, target, mask = images()
    metric = TargetCost()
    cfg = dict(iterations=0, state_weight=0., metric_weight=2., sigma=0., quadrature_points=8)
    _, diag = optimize_path(source, target, mask, metric, cfg)
    expected = ((target - source).square().flatten(1).mean(1) * .5).mean().item()
    assert math.isclose(diag["metric_before"], expected, rel_tol=1e-10)
    assert math.isclose(diag["objective_before"], 3 * expected, rel_tol=1e-10)


def test_optimizer_is_usable_inside_no_grad_validation():
    source, target, mask = images()
    with torch.no_grad():
        path, diag = optimize_path(source, target, mask, TargetCost(), dict(iterations=2))
    assert math.isfinite(diag["objective_after"])
    assert not path.mean_knots.requires_grad


def test_invalid_endpoints_times_masks_and_hyperparameters_are_rejected():
    source, target, mask = images()
    with pytest.raises(ValueError, match="observed"):
        linear_path(source, target, torch.ones_like(mask))
    path = linear_path(source, target, mask)
    for t in (0., 1., float("nan")):
        with pytest.raises(ValueError, match="time"):
            sample_path(path, t)
    for cfg in ({"iterations":-1}, {"quadrature_points":0}, {"sigma":-1},
                {"state_weight":-1}, {"learning_rate":0}, {"knots":2}):
        with pytest.raises(ValueError):
            optimize_path(source, target, mask, None, cfg)


@pytest.mark.parametrize("points", [1, 3, 5, 8])
def test_control_integrates_all_segments_and_straight_path_is_global_minimum(points):
    source, target, mask = images()
    cfg = dict(iterations=3, sigma=.1, state_weight=0., metric_weight=0.,
               quadrature_points=points, knots=5)
    path, diag = optimize_path(source, target, mask, None, cfg)
    expected = .5 * (target - source).square().mean().item()
    assert math.isclose(diag["objective_after"], expected, rel_tol=1e-12, abs_tol=1e-12)
    torch.testing.assert_close(path.mean(.34), .66 * source + .34 * target, atol=1e-8, rtol=1e-8)
    # A kink confined to a segment missed by state quadrature still costs action.
    path.mean_knots[:, 0] += 1
    t = torch.tensor([.5], dtype=source.dtype)
    value, _ = _objective(path, None, t, torch.zeros(2, 1, *source.shape[1:]), cfg)
    assert value > expected + .1


def test_learned_variance_derivative_and_forward_backward_covariance_evolution():
    source, target, mask = images()
    linear = linear_path(source, target, mask, sigma=.2)
    path = PathBatch(source, target, mask, linear.mean_knots, sigma=.2,
                     std_knots=torch.tensor([[.2,-.3,.5],[.4,.1,-.2]],dtype=source.dtype))
    t = torch.tensor([.31,.83],dtype=source.dtype)
    std, std_dot = path.std_and_derivative(t)
    dt=1e-6
    numeric=(path.std_and_derivative(t+dt)[0]-path.std_and_derivative(t-dt)[0])/(2*dt)
    torch.testing.assert_close(std_dot,numeric,atol=1e-9,rtol=1e-8)
    a_fwd=(std_dot-path.sigma**2/(2*std))/std
    a_bwd=(-std_dot-path.sigma**2/(2*std))/std
    torch.testing.assert_close(2*a_fwd*std.square()+path.sigma**2,2*std*std_dot)
    torch.testing.assert_close(2*a_bwd*std.square()+path.sigma**2,-2*std*std_dot)
    for direction,sign,a in [('forward',1,a_fwd),('backward',-1,a_bwd)]:
        state,drift=sample_path(path,t,noise=torch.ones_like(source),direction=direction)
        mean,velocity=path.mean_and_velocity(t)
        torch.testing.assert_close(drift,sign*velocity+a[:,None,None,None,None]*(state-mean))


def test_variance_optimization_is_finite_and_keeps_endpoint_conditions():
    source=torch.zeros(2,1,2,2,2)
    cfg=dict(iterations=8,learning_rate=.02,sigma=.2,knots=5,quadrature_points=12,
             state_weight=4.,metric_weight=0.,optimize_std=True)
    path,diag=optimize_path(source,source,source,TargetCost(),cfg)
    assert diag['optimized_std']==1
    assert path.std_knots is not None and not path.std_knots.requires_grad
    assert torch.count_nonzero(path.std_knots)>0
    assert math.isfinite(diag['objective_after'])
    assert diag['objective_after']<=diag['objective_before']
    assert torch.equal(path.mean(0),source) and torch.equal(path.mean(1),source)
    std,_=path.std_and_derivative(.5)
    assert (std>0).all()


def test_brownian_variance_and_linear_mean_minimize_zero_state_cost():
    source,target,mask=images()
    cfg=dict(iterations=5,learning_rate=.02,sigma=.2,knots=5,quadrature_points=3,
             state_weight=0.,metric_weight=0.,optimize_std=True)
    path,diag=optimize_path(source,target,mask,None,cfg)
    assert diag['variance_control_after']>=0
    assert abs(diag['variance_control_after'])<1e-12
    torch.testing.assert_close(path.std_knots,torch.zeros_like(path.std_knots),atol=1e-8,rtol=0)


def test_positive_variance_action_matches_subtracted_gaussian_control_integral():
    source,target,mask=images(batch=1)
    basic=linear_path(source,target,mask,sigma=.2)
    path=PathBatch(source,target,mask,basic.mean_knots,sigma=.2,
                   std_knots=torch.tensor([[.2,-.3,.4]],dtype=source.dtype))
    # Independently integrate the original conditional Gaussian control energy
    # minus the Brownian conditional baseline, and its stable positive form.
    n=16000
    t=(torch.arange(n,dtype=source.dtype)+.5)/n
    original=[];positive=[]
    for time in t:
        std,std_dot=path.std_and_derivative(time)
        coefficient=std_dot-path.sigma**2/(2*std)
        original.append(.5*(coefficient.square()-path.sigma**2*time/(1-time)))
        baseline=path.sigma*(time*(1-time)).sqrt()
        h=(std/baseline).log()
        h_dot=std_dot/std-(1-2*time)/(2*time*(1-time))
        positive.append(.5*std.square()*h_dot.square()+path.sigma**2*h.sinh().square()/(2*time*(1-time)))
    torch.testing.assert_close(torch.stack(original).mean(),torch.stack(positive).mean(),atol=2e-7,rtol=1e-5)

