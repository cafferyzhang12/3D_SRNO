import torch
import pytest

from vemgsbm.mri_metrics import mean_xy_ssim
from vemgsbm.mri_v2_losses import (ReconstructionLoss, fast_train_ssim, full_batch_ssim,
                                  gradient_diagnostics, local_moments, multiscale_ssim)
from vemgsbm.mri_v2_diagnostics import (local_variance_ratio, textured_region_ssim,
                                       noise_ceiling_estimate, upsample_xy, ssim_map)


def texture():
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing="ij")
    return (.5+.12*torch.sin(17*xx)*torch.cos(13*yy)+.15*torch.exp(-3*(xx*xx+yy*yy)))[None, None, None].repeat(2, 1, 3, 1, 1)


def test_chunked_ssim_exact_value_and_gradient_on_full_batch():
    torch.manual_seed(4)
    y = texture()
    p = (.8*y+.03*torch.randn_like(y)).requires_grad_()
    exact = mean_xy_ssim(p, y)
    actual = full_batch_ssim(p, y, chunk_slices=2)
    assert torch.allclose(actual, exact, atol=1.e-12, rtol=1.e-12)
    first, = torch.autograd.grad(actual, p, retain_graph=True)
    second, = torch.autograd.grad(exact, p)
    torch.testing.assert_close(first, second, atol=1.e-8, rtol=1.e-5)
    assert (first.flatten(2).norm(dim=-1) > 0).all()


def test_float32_training_ssim_matches_exact_metric_and_gradient():
    rng = torch.Generator().manual_seed(4)
    y = texture()
    p = (.8*y+.03*torch.randn(y.shape, generator=rng)).requires_grad_()
    actual = fast_train_ssim(p, y)
    exact = mean_xy_ssim(p, y)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), exact, atol=2.e-5, rtol=0)
    gradient, = torch.autograd.grad(actual, p)
    expected, = torch.autograd.grad(exact, p)
    torch.testing.assert_close(gradient, expected, atol=1.e-7, rtol=1.e-3)
    assert (gradient.flatten(2).norm(dim=-1) > 0).all()


def test_training_ssim_and_local_std_share_full_resolution_pool(monkeypatch):
    import vemgsbm.mri_v2_losses as losses

    y = texture()
    p = (.9*y).requires_grad_()
    _, vx = local_moments(p.double())
    _, vy = local_moments(y.double())
    expected = ((vx+1.e-8).sqrt()-(vy+1.e-8).sqrt()).abs().mean()
    original = losses.F.avg_pool2d
    windows = []

    def track_pool(x, kernel_size, *args, **kwargs):
        if kernel_size == 7 and x.shape[-2:] == p.shape[-2:]:
            windows.append(x.shape)
        return original(x, kernel_size, *args, **kwargs)

    monkeypatch.setattr(losses.F, "avg_pool2d", track_pool)
    terms = ReconstructionLoss()(p, y)
    assert windows == [(6, 5, 32, 32)]
    torch.testing.assert_close(terms["local_std"].double(), expected, atol=3.e-6, rtol=0)
    gradient, = torch.autograd.grad(terms["local_std"], p)
    assert torch.isfinite(gradient).all()
    assert (gradient.flatten(2).norm(dim=-1) > 0).all()


def test_all_loss_terms_have_finite_balanced_gradients():
    y = texture()
    p = (.5+(y-.5)*.8).requires_grad_()
    loss = ReconstructionLoss()
    terms = loss(p, y)
    norms = gradient_diagnostics(terms, (p,), loss.weights)
    assert all(torch.isfinite(value) for value in terms.values())
    assert norms["ssim"]["norm_share"] > .05
    assert norms["local_std"]["norm_share"] > .005
    terms["loss"].backward()
    assert torch.isfinite(p.grad).all()
    assert (p.grad.flatten(2).norm(dim=-1) > 0).all()


def test_identical_and_constant_losses_are_finite():
    for value in (torch.zeros(2, 1, 2, 16, 16), texture()):
        p = value.clone().requires_grad_()
        terms = ReconstructionLoss()(p, value)
        assert abs(float(terms["ssim"].detach())) < 1.e-10
        assert abs(float(terms["msssim"].detach())) < 2.e-5
        terms["loss"].backward()
        assert torch.isfinite(p.grad).all()
    assert multiscale_ssim(texture(), texture()).item() == pytest.approx(1, abs=2.e-5)


@pytest.mark.parametrize("mode", ["bilinear", "bicubic", "lanczos"])
def test_interpolation_preserves_z_constants_and_gradients(mode):
    x = torch.ones(2, 1, 3, 5, 6, requires_grad=True)
    actual = upsample_xy(x, mode=mode)
    assert actual.shape == (2, 1, 3, 20, 24)
    torch.testing.assert_close(actual, torch.ones_like(actual), atol=1.e-6, rtol=1.e-6)
    actual.mean().backward()
    assert torch.isfinite(x.grad).all()


def test_diagnostics_recover_variance_and_exact_map():
    y = texture()
    p = .5+.8*(y-.5)
    assert local_variance_ratio(p, y) == pytest.approx(.64, abs=3.e-5)
    assert textured_region_ssim(y, y) == pytest.approx(1, abs=1.e-10)
    assert ssim_map(p, y).mean().item() == pytest.approx(mean_xy_ssim(p, y).item(), abs=1.e-12)
    info = noise_ceiling_estimate(y)
    assert info["physical_upper_bound_identified"] is False
    assert 0 < info["additive_noise_oracle_ssim_estimate"] <= 1


def test_lpips_is_explicit_opt_in():
    with pytest.raises(ValueError, match="pretrained"):
        ReconstructionLoss({"lpips": .02})
    class FrozenDistance(torch.nn.Module):
        def forward(self, p, r):
            return (p-r).square().mean((1, 2, 3))
    p = (.8*texture()).requires_grad_()
    terms = ReconstructionLoss({"lpips": .02}, FrozenDistance())(p, texture())
    assert terms["lpips"] > 0
    terms["loss"].backward()
    assert p.grad is not None


def test_twenty_step_texture_reconstruction_overfit():
    # Direct intensity parameters isolate the composite loss and its gradients
    # from backbone expressivity; end-to-end network smoke is a separate check.
    y = texture()[:1, :, :1]
    prediction = torch.nn.Parameter(.5+.8*(y-.5))
    optimizer = torch.optim.Adam([prediction], lr=.012)
    loss = ReconstructionLoss()
    before = float(mean_xy_ssim(prediction, y).detach())
    for _ in range(20):
        optimizer.zero_grad()
        loss(prediction, y)["loss"].backward()
        optimizer.step()
    assert float(mean_xy_ssim(prediction, y).detach()) > .98
    assert float(mean_xy_ssim(prediction, y).detach()) > before
