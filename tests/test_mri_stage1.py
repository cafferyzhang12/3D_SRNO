import numpy as np
import pytest
import torch

from vemgsbm.momentum.model import HaarXY
from vemgsbm.mri_stage1 import correction_regions, correction_report
from vemgsbm.network import diffusion_adjusted_model, diffusion_drift_weights
from vemgsbm.paths import linear_path, sample_path


@pytest.mark.parametrize("eta", [0., .05, .1])
def test_changed_diffusion_preserves_gaussian_marginal_evolution(eta):
    source = torch.zeros((1, 1, 3, 4, 4), dtype=torch.float64)
    path = linear_path(source, source + .8, source, sigma=.10)
    path.mean_knots[:, 0] += .1
    path.std_knots = torch.tensor([[.2, -.3, .4]], dtype=source.dtype)
    t = .37
    mean, velocity = path.mean_and_velocity(t)
    std, derivative = path.std_and_derivative(t)
    states = []
    calls = []
    def model(state, base, mask, time, direction):
        calls.append(direction)
        _, drift = sample_path(path, time, noise=(state - mean) / std, direction=direction)
        return drift
    adjusted = diffusion_adjusted_model(model, path.sigma, eta)
    for noise in (-1., 1.):
        state, _ = sample_path(path, t, torch.full_like(source, noise))
        states.append(adjusted(state, source, source, t, "forward"))
    # Gaussian mean derivative = E[b], variance derivative = 2 Cov[X,b]+eta^2.
    torch.testing.assert_close((states[0] + states[1]) / 2, velocity)
    variance_dot = (states[1] - states[0]) * std + eta**2
    torch.testing.assert_close(variance_dot, torch.ones_like(source) * (2 * std * derivative))
    assert len(calls) == (2 if eta == .10 else 4)
    assert diffusion_drift_weights(.10, .05) == (.625, -.375)


def test_correction_identity_and_haar_parseval():
    rng = np.random.default_rng(43)
    hr, plain, guided = rng.random((3, 8, 12))
    codec = HaarXY().double()
    stats = correction_regions(hr, plain, guided, codec)
    delta = np.mean((hr - guided)**2 - (hr - plain)**2)
    assert correction_report(stats["whole"])["delta_mse"] == pytest.approx(delta)
    assert stats["haar_high"]["count"] == 15 * 6
    # DC is the normalized sum within each aligned 4x4 block.
    dc_error = (hr - plain).reshape(2, 4, 3, 4).sum((1, 3)) / 4
    dc_change = (guided - plain).reshape(2, 4, 3, 4).sum((1, 3)) / 4
    assert stats["haar_high"]["dot"] + (dc_error * dc_change).sum() == pytest.approx(stats["whole"]["dot"])
    assert correction_report(correction_regions(hr, plain, plain, codec)["whole"])["optimal_change_multiplier"] is None


def test_replay_runtime_weights_exact_trajectory_rng_and_hr_oracle():
    from collections import Counter, defaultdict
    from test_atlas_geometry import fixture_geometry, image
    from vemgsbm.mri_guidance_replay import _collect_chart, trace_tile
    from vemgsbm.mri_stage1 import oracle_fit_scores

    geometry = fixture_geometry(curvature=.5)
    base = image([[.15, -.25]])
    def model(state, condition, mask, time, direction):
        return state.sin() + .12
    generator = torch.Generator().manual_seed(42)
    result, chart, calls = trace_tile(
        model, base, torch.zeros_like(base), "gsbm", geometry=geometry, generator=generator,
        sigma=.05, steps=20, terminal_noise=False,
        geometry_guidance={"strength":.02,"normal_strength":.01,"start":.3,"max_update":.002})
    assert torch.isfinite(result).all() and calls == 20
    assert len(chart.observations) == 26  # .35 through .95, both cost branches
    assert {s for s, _, _, _ in chart.observations} == set(range(7,20))
    assert all(not w.requires_grad for _, _, w, _ in chart.observations)
    counts = defaultdict(Counter)
    _collect_chart(chart, oracle_fit_scores(geometry, image([[0,1]])),counts)
    assert len(counts) == 26
    assert all(v["updates"] == 1 for v in counts.values())
    assert all(v["trust_weight_mass"] <= 1 for v in counts.values())
    assert any(abs(v["lr_prior_gap"] - v["actual_weights_gap"]) > 1.e-6 for v in counts.values())
