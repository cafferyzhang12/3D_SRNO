import pytest
import torch

from vemgsbm.frequency_bridge import (bridge_sigma, sample_frequency_bridge,
                                     frequency_velocity, integrate_frequency_bridge)


def test_boundary_variance_and_ll_are_deterministic():
    target = torch.randn(4000, 16, 1, 1, 1)
    a, _, _ = sample_frequency_bridge(target, time=.5, sigma_max=.2)
    b, _, _ = sample_frequency_bridge(target, time=.5, sigma_max=.2)
    torch.testing.assert_close(a[:, :1], .5*target[:, :1])
    torch.testing.assert_close(a[:, :1], b[:, :1])
    assert ((a[:, 1:]-.5*target[:, 1:]).var()-.01).abs() < .0004
    for t in [0., 1.]:
        state, _, epsilon = sample_frequency_bridge(target, time=t, sigma_max=.2)
        torch.testing.assert_close(state, t*target)
        assert epsilon.count_nonzero() == 0
        assert bridge_sigma(t, .2) == 0


def test_probability_flow_velocity_matches_path_derivative():
    torch.manual_seed(12)
    y = torch.randn(3, 16, 1, 2, 2).double()
    noise = torch.randn_like(y)
    t, h = .37, 1.e-6
    x, _, eps = sample_frequency_bridge(y, .15, time=t, noise=noise)
    plus = sample_frequency_bridge(y, .15, time=t+h, noise=noise)[0]
    minus = sample_frequency_bridge(y, .15, time=t-h, noise=noise)[0]
    v = frequency_velocity(x, y, eps, t, .15)
    torch.testing.assert_close(v, (plus-minus)/(2*h), rtol=1.e-7, atol=1.e-8)


@pytest.mark.parametrize("steps", [1, 4, 8, 16])
@pytest.mark.parametrize("sampler", ["ode", "sde"])
def test_arbitrary_steps_finite_terminal_and_ll_exact(steps, sampler):
    target = torch.randn(2, 16, 2, 3, 3)
    initial = torch.zeros_like(target)
    visited = []
    def predict(x, t):
        visited.append(t)
        return target, torch.zeros_like(x)
    result = integrate_frequency_bridge(predict, initial, steps, sampler_type=sampler)
    torch.testing.assert_close(result, target)
    assert visited[0] == 0
    assert all(0 <= t < 1 for t in visited)
    assert torch.isfinite(result).all()


def test_continuous_times_and_sde_rng_reproducible():
    target = torch.randn(128, 16, 1, 2, 2)
    _, t, _ = sample_frequency_bridge(target, endpoint_fraction=.2)
    assert len(t.unique()) > 50
    def predict(x, t):
        return .1*x+target, torch.tanh(x)
    outputs = [integrate_frequency_bridge(predict, torch.zeros_like(target), 8,
        sigma_max=.08, sampler_type="sde", temperature=.2, generator=torch.Generator().manual_seed(16))
        for _ in range(2)]
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    torch.testing.assert_close(outputs[0][:, :1], outputs[1][:, :1])


def test_invalid_temperature_rejected():
    x = torch.zeros(1,16,1,2,2)
    with pytest.raises(ValueError, match="temperature"):
        integrate_frequency_bridge(lambda x,t:(x,x), x, temperature=.31)


def test_sde_and_ode_have_identical_ll_with_noise_sensitive_predictor():
    x = torch.zeros(2,16,1,2,2)
    def predict(state,t):
        return torch.cat((torch.ones_like(state[:,:1]), state[:,1:]),1), torch.zeros_like(state)
    ode = integrate_frequency_bridge(predict,x,8,sigma_max=.08)
    sde = integrate_frequency_bridge(predict,x,8,sigma_max=.08,sampler_type="sde",temperature=.3)
    torch.testing.assert_close(ode[:,:1],sde[:,:1])
    assert (sde[:,1:]-ode[:,1:]).square().sum() > 0


def test_multimodal_posterior_preserves_variance_and_more_steps_do_not_collapse():
    """Exact two-mode posterior separates sampler math from network fitting."""
    torch.manual_seed(22)
    sigma,amplitude,t0 = .15,.25,.001
    initial=torch.zeros(2048,16,1,1,1)
    # An interior sample is necessary for generative deterministic PF from a
    # point source. A zero PF initial state would remain at the symmetric mean.
    initial[:,1:]=sigma*(t0*(1-t0))**.5*torch.randn_like(initial[:,1:])
    def posterior(x,t):
        clean=amplitude*torch.tanh(amplitude*x/(sigma*sigma*(1-t)))
        eps=(x-t*clean)/(sigma*(t*(1-t))**.5)
        return torch.cat((torch.zeros_like(x[:,:1]),clean[:,1:]),1),eps
    for mode in ["ode","sde"]:
        ratios=[]
        for steps in [1,4,8,16]:
            result=integrate_frequency_bridge(posterior,initial,steps,sigma_max=sigma,
                sampler_type=mode,temperature=.2,generator=torch.Generator().manual_seed(22))
            ratios.append(float(result[:,1:].var()/amplitude**2))
        assert ratios[0] < .01
        assert min(ratios[1:]) > .98
        assert all(abs(1-b) <= abs(1-a)+2.e-3 for a,b in zip(ratios,ratios[1:]))
    deterministic=integrate_frequency_bridge(posterior,torch.zeros_like(initial),16,sigma_max=sigma)
    stochastic=integrate_frequency_bridge(posterior,torch.zeros_like(initial),16,sigma_max=sigma,
        sampler_type="sde",temperature=.3,generator=torch.Generator().manual_seed(22))
    assert deterministic[:,1:].var() == 0
    assert stochastic[:,1:].var()/amplitude**2 > .98


def test_deterministic_interpolation_and_rollout_follow_exact_linear_path(monkeypatch):
    target = torch.randn(2, 16, 1, 2, 2, dtype=torch.float64)
    initial = .25*torch.randn_like(target)

    def forbidden(*args, **kwargs):
        raise AssertionError('The zero-sigma bridge does not sample noise')

    monkeypatch.setattr(torch, 'randn_like', forbidden)
    monkeypatch.setattr(torch, 'randn', forbidden)
    sampled, times, epsilon = sample_frequency_bridge(target, time=.375, initial=initial)
    assert not epsilon.count_nonzero()
    torch.testing.assert_close(sampled, .625*initial+.375*target)
    torch.testing.assert_close(frequency_velocity(sampled, target, None, times), target-initial)
    with pytest.raises(ValueError, match='direct endpoint readout'):
        frequency_velocity(target, target, None, 1.)

    for solver in ('euler', 'heun'):
        def predict(state, time):
            torch.testing.assert_close(state, (1-time)*initial+time*target)
            return target, None

        result = integrate_frequency_bridge(predict, initial, 8, solver=solver, curvature=forbidden)
        torch.testing.assert_close(result, target, rtol=0, atol=0)
