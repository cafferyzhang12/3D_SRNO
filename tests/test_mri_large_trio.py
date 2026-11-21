import torch
import pytest

from vemgsbm.haar_residual import HaarObjective, HaarResidualNetwork
from vemgsbm.paths import linear_path, sample_path


def test_scaled_drift_errors_are_endpoint_discrepancies():
    zero = torch.zeros(3, 16, 2, 3, 4)
    target = torch.randn_like(zero)
    path = linear_path(zero, target, zero)
    t = torch.tensor([.03, .45, .97])
    noise = torch.randn_like(zero)
    state, f = sample_path(path, t, noise, "forward")
    _, r = sample_path(path, t, noise, "backward")
    tau = t[:,None,None,None,None]
    torch.testing.assert_close(state + (1 - tau)*f, target)
    torch.testing.assert_close(state + tau*r, zero)


@pytest.mark.parametrize("weight", [0., 1.])
def test_endpoint_unit_objective_and_control(weight):
    net = HaarResidualNetwork("haar_bridge", [4, 8, 12], "endpoint" if weight else "integral")
    cfg = {"train": {"matching_units":"endpoint","endpoint_weight":weight},
           "path":{"sigma":.1,"knots":3}, "sampling":{"steps":4,"inference_sigma":.05}}
    lr = torch.rand(2, 1, 3, 4, 4)
    base = torch.zeros(2, 1, 3, 16, 16)
    hr = torch.rand_like(base)
    terms = HaarObjective(net, cfg)(lr, base, hr)
    expected = (terms["forward_endpoint_equivalent_mse"] + terms["reverse_endpoint_equivalent_mse"]) / 2
    expected += weight * (terms["endpoint_pixel_mse"] + terms["rollout_pixel_mse"]) / 2
    torch.testing.assert_close(terms["loss"].detach(), expected.detach())
    terms["loss"].backward()
    assert bool(net.endpoint.weight.grad.abs().sum() > 0) == bool(weight)
    assert net.drift.weight.grad.abs().sum() > 0
