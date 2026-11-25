"""Deployment-state SR supervision, physical units, and detached sampler prefixes."""
import pytest
import torch
from torch import nn

from vemgsbm.bridge import integrate_endpoint
from vemgsbm.haar_residual import HaarObjective, X1HaarResidualNetwork
from vemgsbm.momentum.model import HaarXY


class ToyEndpoint(nn.Module):
    predict_target = "x1"
    sigma_0 = 0.
    sigma_bridge = 0.
    sampler_type = "ode"
    solver = "euler"
    use_band_whitening = True

    def __init__(self, scale=1., state_weight=.3):
        super().__init__()
        self.codec = HaarXY()
        self.gain = nn.Parameter(torch.linspace(.1, .4, 16).reshape(1, 16, 1, 1, 1))
        self.condition_gain = nn.Parameter(torch.tensor(.2))
        self.register_buffer("band_std", scale * torch.linspace(.1, .3, 16) - 1.e-6)
        self.register_buffer("band_mean", scale * torch.linspace(-.02, .03, 16))
        self.state_weight = state_weight
        self.calls = []

    def lr_encoder(self, lr):
        return [lr.expand(-1, 16, -1, -1, -1) * self.condition_gain]

    def initial_state(self, lr, base):
        return lr.new_zeros(len(lr), 16, *lr.shape[2:])

    def normalize_haar(self, coefficients):
        return ((coefficients - self.band_mean.reshape(1, 16, 1, 1, 1))
                / (self.band_std + 1.e-6).reshape(1, 16, 1, 1, 1))

    def denormalize_haar(self, coefficients):
        return (coefficients * (self.band_std + 1.e-6).reshape(1, 16, 1, 1, 1)
                + self.band_mean.reshape(1, 16, 1, 1, 1))

    def forward(self, state, condition, time):
        self.calls.append(dict(state=state.detach().clone(), time=float(time),
                               grad_enabled=torch.is_grad_enabled(),
                               state_grad=state.requires_grad,
                               condition_grad=condition[0].requires_grad))
        return self.gain + self.state_weight * state + condition[0] * (1 + time), None


def config(steps=5, **train):
    return {"train": dict(objective="sr_reconstruction", rollout_steps=2, **train),
            "sampling": dict(steps=steps, sampler_type="ode", solver="euler", inference_sigma=0.),
            "mls_reg_weight": 0.}


def batch(n=16):
    generator = torch.Generator().manual_seed(19)
    lr = torch.rand(n, 1, 1, 2, 3, generator=generator)
    base = torch.rand(n, 1, 1, 8, 12, generator=generator) * .1
    hr = torch.rand(n, 1, 1, 8, 12, generator=generator) * .3
    return lr, base, hr


@pytest.mark.parametrize("solver", ["euler", "heun"])
def test_sr_prefix_matches_real_deployment_grid_and_has_no_gradient(solver):
    model, (lr, base, hr) = ToyEndpoint(), batch()
    cfg = config()
    cfg["sampling"]["solver"] = solver
    terms = HaarObjective(model, cfg)(lr, base, hr)
    calls = list(model.calls)
    final_call, intermediate_call = calls[-2:]
    assert len(calls[0]["state"]) == 16 and calls[0]["time"] == 0.
    assert not calls[0]["state"].any()
    assert len(final_call["state"]) == 2 and final_call["time"] == 4 / 5
    assert intermediate_call["time"] in {1 / 5, 2 / 5, 3 / 5}
    assert all(not call["grad_enabled"] and not call["condition_grad"] for call in calls[1:-2])
    assert all(call["grad_enabled"] and not call["state_grad"] for call in calls[-2:])
    assert all(call["condition_grad"] for call in calls[-2:])
    condition = model.lr_encoder(lr[:2])
    accepted = {}

    def deployment_predict(state, time):
        accepted[time] = state.clone()
        return model(state, condition, time)[0]

    with torch.no_grad():
        result = integrate_endpoint(deployment_predict, model.initial_state(lr[:2], base[:2]),
                                    5, solver=solver, sigma_bridge=0.)
    torch.testing.assert_close(final_call["state"], accepted[4 / 5])
    torch.testing.assert_close(intermediate_call["state"], accepted[intermediate_call["time"]])
    actual_mse = (base[:2] + model.codec.decode(model.denormalize_haar(result)) - hr[:2]).square().mean()
    torch.testing.assert_close(terms["rollout_pixel_mse"], actual_mse)
    terms["loss"].backward()
    assert model.gain.grad.abs().sum() > 0
    assert model.condition_gain.grad.abs() > 0


def test_sr_inputs_do_not_leak_paired_hr_but_loss_and_gradients_use_it():
    model, (lr, base, hr) = ToyEndpoint(), batch(8)
    objective = HaarObjective(model, config())
    torch.manual_seed(8)
    first = objective(lr, base, hr)
    inputs = list(model.calls)
    first["loss"].backward()
    gradient = model.gain.grad.clone()
    model.zero_grad()
    model.calls.clear()
    torch.manual_seed(8)
    second = objective(lr, base, hr + .4)
    second["loss"].backward()
    for original, changed in zip(inputs, model.calls):
        assert original["time"] == changed["time"]
        torch.testing.assert_close(original["state"], changed["state"])
    assert not torch.isclose(first["loss"], second["loss"])
    assert not torch.allclose(gradient, model.gain.grad)
    assert not any(key.startswith("bridge_aux") for key in second)


def test_sr_common_normalization_preserves_physical_scale_invariance():
    lr, base, hr = batch(8)
    models = [ToyEndpoint(scale=scale) for scale in (1., 7.)]
    outcomes = []
    for model, scale in zip(models, (1., 7.)):
        torch.manual_seed(3)
        terms = HaarObjective(model, config())(lr, base * scale, hr * scale)
        terms["loss"].backward()
        outcomes.append(terms)
    torch.testing.assert_close(outcomes[0]["loss"], outcomes[1]["loss"])
    torch.testing.assert_close(models[0].gain.grad, models[1].gain.grad)
    for phase in ("start", "rollout", "intermediate"):
        for name in ("pixel_mse", "hf_mse", "edge_mse"):
            torch.testing.assert_close(outcomes[1][f"{phase}_{name}"],
                                       outcomes[0][f"{phase}_{name}"] * 49)
    expected = (models[0].band_std + 1.e-6).square().mean()
    torch.testing.assert_close(outcomes[0]["loss_normalization"], expected)


def test_sr_high_frequency_keeps_sixteen_band_denominator_and_edges_are_normalized():
    model = ToyEndpoint(state_weight=0.)
    model.band_std.fill_(1. - 1.e-6)
    model.band_mean.zero_()
    model.condition_gain.data.zero_()
    model.gain.data.zero_()
    model.gain.data[:, 0] = 4.  # Unit-valued constant image, entirely LL.
    lr, base, hr = batch(1)
    base.zero_()
    hr.zero_()
    objective = HaarObjective(model, config(steps=1))
    constant = objective(lr, base, hr)
    assert constant["start_pixel_mse"].detach() == pytest.approx(1.)
    assert constant["start_hf_mse"] == 0
    assert constant["start_edge_mse"].detach() == pytest.approx(0., abs=1.e-12)
    model.gain.data.zero_()
    model.gain.data[:, 1] = 4.
    striped = objective(lr, base, hr)
    assert striped["start_hf_mse"].detach() == pytest.approx(1.)
    image = model.codec.decode(model.gain.expand(1, 16, 1, 2, 3))
    expected_edge = .5 * (image.diff(dim=-1).square().mean() + image.diff(dim=-2).square().mean()) / 4
    torch.testing.assert_close(striped["start_edge_mse"], expected_edge)


@pytest.mark.parametrize("steps", [1, 2])
def test_sr_short_rollouts_do_not_double_weight_a_missing_intermediate(steps):
    model, (lr, base, hr) = ToyEndpoint(), batch(1)
    terms = HaarObjective(model, config(steps=steps, hf_weight=0., edge_weight=0.))(lr, base, hr)
    assert not any(key.startswith("intermediate_") for key in terms)
    expected = (terms["start_pixel_mse"] + terms["rollout_pixel_mse"]) / terms["loss_normalization"]
    torch.testing.assert_close(terms["loss"], expected)
    assert terms["rollout_items"] == 1


def test_sr_mls_receives_physical_scale_paired_target_and_base_coefficients():
    class Atlas:
        def __init__(self):
            self.calls = []

        def reconstruction_loss(self, prediction, truth, ids, band_scale, base_coefficients):
            self.calls.append((truth, ids, band_scale, base_coefficients))
            return ((prediction - truth) * band_scale.reshape(1, 16, 1, 1, 1)).square().mean()

    model, atlas, (lr, base, hr) = ToyEndpoint(), Atlas(), batch(8)
    cfg = config()
    cfg["mls_reg_weight"] = .05
    terms = HaarObjective(model, cfg, atlas=atlas)(lr, base, hr, list("abcdefgh"))
    assert len(atlas.calls) == 3
    truth, ids, scale, base_coefficients = atlas.calls[0]
    torch.testing.assert_close(truth, model.normalize_haar(model.codec.encode(hr - base)))
    torch.testing.assert_close(base_coefficients, model.codec.encode(base))
    torch.testing.assert_close(scale, model.band_std + 1.e-6)
    assert ids == list("abcdefgh") and not truth.requires_grad
    assert atlas.calls[1][1] == atlas.calls[2][1] == ["a"]
    assert terms["start_mls_reconstruction_mse"] > 0
    terms["loss"].backward()
    assert model.gain.grad.abs().sum() > 0


def test_sr_initial_endpoint_head_receives_direct_gradient():
    model = X1HaarResidualNetwork([4, 6, 8], sigma_0=0., sigma_bridge=0., use_band_whitening=False)
    lr, base, hr = batch(1)
    terms = HaarObjective(model, config(steps=1, rollout_weight=0.))(lr, base, hr)
    terms["loss"].backward()
    assert model.endpoint.weight.grad.abs().sum() > 0
    assert model.endpoint.bias.grad.abs().sum() > 0
    assert all(torch.isfinite(parameter.grad).all()
               for parameter in model.parameters() if parameter.grad is not None)


def test_sr_rejects_teacher_coupling_and_missing_train_atlas():
    model, (lr, base, hr) = ToyEndpoint(), batch(1)
    with pytest.raises(ValueError, match="does not use a teacher"):
        HaarObjective(model, config(), teacher=nn.Identity())(lr, base, hr)
    cfg = config()
    cfg["mls_reg_weight"] = .05
    with pytest.raises(ValueError, match="TRAIN atlas and case IDs"):
        HaarObjective(model, cfg)(lr, base, hr)


def test_sr_ssim_uses_deployed_reconstruction_and_adds_dimensionless_loss():
    from vemgsbm.mri_metrics import mean_xy_ssim
    model, (lr, base, hr) = ToyEndpoint(), batch(16)
    torch.manual_seed(21)
    original = HaarObjective(model, config(steps=4))(lr, base, hr)
    torch.manual_seed(21)
    updated = HaarObjective(model, config(steps=4, ssim_weight=.3))(lr, base, hr)
    extra = .3 * (updated["start_ssim_loss"] + .5 * (updated["rollout_ssim_loss"] + updated["intermediate_ssim_loss"]))
    torch.testing.assert_close(updated["loss"], original["loss"] + extra)
    assert all(updated[f"{phase}_ssim_items"] == 2 for phase in ("start", "rollout", "intermediate"))
    cond = model.lr_encoder(lr[:2])
    with torch.no_grad():
        endpoint = integrate_endpoint(lambda x,t: model(x,cond,t)[0], model.initial_state(lr[:2],base[:2]),4,sigma_bridge=0.)
        image = (base[:2] + model.codec.decode(model.denormalize_haar(endpoint))).clamp(0,1)
    torch.testing.assert_close(updated["rollout_ssim_loss"], 1-mean_xy_ssim(image,hr[:2]))
    updated["rollout_ssim_loss"].backward()
    assert model.gain.grad.abs().sum() > 0 and torch.isfinite(model.gain.grad).all()
