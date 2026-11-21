"""Numerical and endpoint tests for geometry-conditioned phase-space bridges."""
import pytest
import torch

from vemgsbm.momentum.geometry import assign_chart, fit_atlas
from vemgsbm.momentum.guidance import atlas_anchor_update
from vemgsbm.momentum.model import AccelerationNetwork, HaarXY
from vemgsbm.momentum.phase import PhasePath, _shape_functions, _transition_coefficients, optimize_path, transition


@pytest.fixture(autouse=True)
def small_threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_haar_roundtrip_and_energy():
    x = torch.randn(2, 1, 5, 12, 16)
    codec = HaarXY()
    encoded = codec.encode(x)
    assert encoded.shape == (2, 16, 5, 3, 4)
    torch.testing.assert_close(codec.decode(encoded), x, atol=7e-7, rtol=1e-6)
    torch.testing.assert_close(x.square().sum(), encoded.square().sum())


@pytest.mark.parametrize("elapsed,remaining", [(0.02, 1), (.4, .7), (.05, .0501), (.05, .05)])
def test_transition_moments_equal_independent_gaussian_conditioning(elapsed, remaining):
    h, r = elapsed, remaining
    g_z, g_v, var_z, cov, var_v = _transition_coefficients(h, r)
    unconditional = torch.tensor([[h**3/3, h*h/2], [h*h/2, h]], dtype=torch.float64)
    cross = torch.tensor([h*h*r/2-h**3/6, h*r-h*h/2], dtype=torch.float64)
    variance = r**3 / 3
    torch.testing.assert_close(torch.tensor([g_z,g_v], dtype=torch.float64), cross / variance)
    torch.testing.assert_close(torch.tensor([[var_z,cov],[cov,var_v]], dtype=torch.float64),
                               unconditional - cross[:,None]*cross[None,:]/variance, atol=1e-12, rtol=1e-9)


def test_terminal_transition_pins_position_and_leaves_momentum_free():
    z, v = torch.randn(2, 3), torch.randn(2, 3)
    target = torch.randn_like(z)
    a = transition(z, v, target, .1, .1, .2, torch.Generator().manual_seed(3))
    b = transition(z, v, target, .1, .1, .2, torch.Generator().manual_seed(4))
    assert torch.equal(a[0], target) and torch.equal(b[0], target)
    assert not torch.equal(a[1], b[1])


def test_shifted_acceleration_formula_and_endpoint_constraints():
    time = torch.tensor(.37, dtype=torch.float64, requires_grad=True)
    values, _, controls = _shape_functions(time)
    for value, control in zip(values, controls):
        first, = torch.autograd.grad(value, time, create_graph=True, retain_graph=True)
        second, = torch.autograd.grad(first, time, retain_graph=True)
        torch.testing.assert_close(control, second + 3 * value / (1-time)**2 + 3 * first / (1-time))
    target = torch.randn(2, 4, 1, 2, 2)
    free = torch.ones_like(target[:, :1])
    path = PhasePath(target, torch.randn(2, *target.shape), .15, free)
    z0, v0, _, _ = path.sample(0.)
    z1, _, endpoint, _ = path.sample(1.)
    assert torch.count_nonzero(z0) == torch.count_nonzero(v0) == 0
    assert torch.equal(z1, target) and torch.equal(endpoint, target)


def test_local_pca_changes_path_without_erasing_normal_endpoint():
    torch.manual_seed(4)
    line = torch.linspace(-2, 2, 90)
    bank = torch.stack((line, torch.full_like(line, .6), torch.zeros_like(line)), dim=1)
    atlas = fit_atlas(bank, centers=3, neighbors=18, tangent_rank=1, seed=4)
    torch.testing.assert_close(atlas["basis"].abs()[:,0,0], torch.ones(3))
    endpoint = torch.tensor([.1, .2, -.1]).reshape(1,3,1,1,1)
    free = torch.ones_like(endpoint[:, :1])
    chart = assign_chart(atlas, endpoint * (5/16))
    config = dict(sigma=.05, lr=.03, iterations=15, quadrature=5, max_shift=.5,
                  state_weight=2., normal_weight=.25)
    path, result = optimize_path(endpoint, free, chart, config)
    assert result["path_after"] < result["path_before"]
    assert path.shifts[:, :, 1].abs().sum() > 1e-5
    assert torch.equal(path.sample(1.)[0], endpoint)
    assert not path.shifts.requires_grad


def test_motion_preserves_observed_slices_and_single_step_anchor():
    model = AccelerationNetwork(width=4)
    base = torch.rand(1,1,3,8,8)
    mask = torch.zeros_like(base)
    mask[:,:,::2] = 1
    with torch.no_grad():
        model.output.bias.fill_(.1)
    first = model.restore(base, mask, 1, .2, torch.Generator().manual_seed(11))
    second = model.restore(base, mask, 1, .2, torch.Generator().manual_seed(12))
    assert torch.equal(first, second)
    assert torch.equal(first[:,:,::2], base[:,:,::2])
    assert not torch.equal(first[:,:,1], base[:,:,1])


def test_atlas_force_damps_normal_velocity_and_preserves_observations():
    class FlatChart:
        @staticmethod
        def phase_costs(image, speed):
            # Each plane has tangents along its first half and normals along its second half.
            return image.square().flatten(1).mean(1), speed[..., 4:].square().flatten(1).mean(1)

    codec = HaarXY()
    base = torch.zeros(1, 1, 2, 8, 8)
    mask = torch.zeros_like(base)
    mask[:, :, 0] = 1
    speed = torch.full_like(base, .7)
    time = .6
    options = dict(position_strength=0., velocity_strength=10., start=.25, max_acceleration=.2)
    update = atlas_anchor_update(codec, FlatChart(), base, mask, codec.encode(base),
                                 codec.encode(speed), time, options)
    acceleration = codec.decode(update) * (3 / (1 - time)**2)
    torch.testing.assert_close(acceleration[:, :, 0], torch.zeros_like(acceleration[:, :, 0]))
    torch.testing.assert_close(acceleration[..., :4], torch.zeros_like(acceleration[..., :4]))
    torch.testing.assert_close(acceleration[:, :, 1, :, 4:], torch.full((1, 1, 8, 4), -.2))
