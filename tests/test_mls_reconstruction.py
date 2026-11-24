"""Paired TRAIN charts supervise curved residual errors in physical Haar units."""
import copy

import pytest
import torch
import yaml

from vemgsbm.spatial_atlas import TrainingMLSAtlas, prepare_training_mls


def _bank(scale=None):
    generator = torch.Generator().manual_seed(241)
    coordinates = torch.randn(400, 2, generator=generator) * .3
    physical = torch.zeros(400, 16)
    physical[:, :2] = coordinates
    physical[:, 2] = .6 * coordinates[:, 0].square() + .3 * coordinates[:, 1].square()
    scale = torch.ones(16) if scale is None else scale
    return dict(residuals=physical / scale, base_coefficients=torch.randn(400, 16, generator=generator),
                band_scale=scale, case_indices=torch.arange(4).repeat_interleave(100),
                train_case_ids=list("abcd"), provenance={"split": "train"})


def _atlas(payload=None, **settings):
    options = dict(geometry_mode="paired_curved", dim=2, folds=2, centers=2, neighbors=100,
                   loss_sites_per_patch=32, query_chunk=8, max_fit_sites_per_fold=180)
    options.update(settings)
    return TrainingMLSAtlas(_bank() if payload is None else payload, options)


def _analytic_atlas(scale=None):
    atlas = _atlas(_bank(scale))
    atlas.mean.zero_()
    atlas.mean[:, 3] = 1.7  # Deliberate chart bias must cancel at the target.
    atlas.basis.zero_()
    atlas.basis[:, :2] = torch.eye(2)
    atlas.bandwidth.fill_(1.)
    atlas.offset.zero_()
    atlas.linear.zero_()
    atlas.form.zero_()
    atlas.form[:, 0, 0, 2] = .6
    atlas.form[:, 1, 1, 2] = .3
    atlas.pca_supported.fill_(1.)
    atlas.quadratic_supported.fill_(1.)
    atlas.domain_lo.fill_(-3.)
    atlas.domain_hi.fill_(3.)
    atlas.radius.fill_(float("inf"))
    return atlas


def _volume(rows):
    return rows.T.reshape(1, 16, 1, 1, -1)


def test_actual_quadratic_fit_has_curvature_and_train_exclusion():
    payload = _bank()
    atlas = _atlas(payload)
    assert atlas.quadratic_supported.sum() > 0
    assert atlas.form.square().sum() > 1.e-7
    assert atlas.fit_sites_per_fold == [180, 180]
    altered = copy.deepcopy(payload)
    held_out = payload["case_indices"] == 0
    altered["residuals"][held_out] += 100.
    altered["base_coefficients"][held_out] -= 70.
    other = _atlas(altered)
    fold = atlas.folds["a"]
    lo, hi = atlas.offsets[fold:fold+2]
    # Both geometry and descriptor normalization exclude the querying fold.
    for key in ("mean", "basis", "form", "centers", "domain_lo", "domain_hi"):
        torch.testing.assert_close(getattr(atlas, key)[lo:hi], getattr(other, key)[lo:hi], rtol=0, atol=0)
    torch.testing.assert_close(atlas.query_mean[fold], other.query_mean[fold], rtol=0, atol=0)
    torch.testing.assert_close(atlas.query_std[fold], other.query_std[fold], rtol=0, atol=0)


def test_curved_loss_and_tangent_gradient_use_predicted_coordinates():
    atlas = _analytic_atlas()
    target = torch.zeros(1, 16, 1, 1, 1, requires_grad=True)
    base = torch.zeros_like(target, requires_grad=True)
    prediction = torch.zeros_like(target)
    prediction[:, 0] = .5
    prediction.requires_grad_()
    loss = atlas.reconstruction_loss(prediction, target, ["a"], torch.ones(16), base)
    assert loss.item() == pytest.approx((.6 * .5**2)**2 / 16)
    loss.backward()
    assert prediction.grad[0, 0, 0, 0, 0].item() == pytest.approx(4*.6**2*.5**3/16)
    assert target.grad is None and base.grad is None
    assert torch.isfinite(prediction.grad).all()
    # Traveling along the actual parabola has zero curved normal discrepancy.
    prediction = prediction.detach()
    prediction[:, 2] = .6 * .5**2
    assert atlas.reconstruction_loss(prediction, target, ["a"], torch.ones(16), base).item() < 1.e-12


def test_exact_truth_has_zero_value_and_gradient_even_with_chart_bias():
    atlas = _analytic_atlas()
    target = torch.randn(1, 16, 1, 2, 3)
    prediction = target.clone().requires_grad_()
    loss = atlas.reconstruction_loss(prediction, target, ["a"], torch.ones(16), torch.randn_like(target))
    assert loss.item() == 0.
    loss.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_anisotropic_whitening_preserves_physical_geometry_and_units():
    scale = torch.linspace(.03, .9, 16)
    ordinary, whitened = _analytic_atlas(), _analytic_atlas(scale)
    physical = torch.zeros(1, 16, 1, 1, 1)
    physical[:, 0], physical[:, 2] = .5, .4
    raw = physical.clone().requires_grad_()
    normalized = (physical / scale.view(1, 16, 1, 1, 1)).requires_grad_()
    zero = torch.zeros_like(raw)
    first = ordinary.reconstruction_loss(raw, zero, ["a"], torch.ones(16), zero)
    second = whitened.reconstruction_loss(normalized, zero, ["a"], scale, zero)
    assert first.item() == pytest.approx((.4-.6*.5**2)**2/16)
    torch.testing.assert_close(first, second)
    first.backward()
    second.backward()
    torch.testing.assert_close(raw.grad, normalized.grad/scale.view(1,16,1,1,1))
    # Fitting the same physical rows with different whitening gives the same
    # physical surfaces as well, not just a rescaled output loss.
    fitted_raw, fitted_normalized = _atlas(_bank()), _atlas(_bank(scale))
    for key in ("mean", "covariance", "bandwidth"):
        torch.testing.assert_close(getattr(fitted_raw, key), getattr(fitted_normalized, key), atol=1.e-6, rtol=1.e-5)


def test_paired_base_changes_training_chart_routing():
    atlas = _analytic_atlas()
    atlas.query_mean.zero_()
    atlas.query_std.fill_(1.)
    atlas.centers.zero_()
    atlas.centers[::2, 16] = -1.
    atlas.centers[1::2, 16] = 1.
    atlas.form[::2, 0, 0, 2] = .25
    atlas.form[1::2, 0, 0, 2] = 1.
    prediction = torch.zeros(1, 16, 1, 1, 1)
    prediction[:, 0] = .5
    truth, base = torch.zeros_like(prediction), torch.zeros_like(prediction)
    base[:, 0] = -1.
    first = atlas.reconstruction_loss(prediction, truth, ["a"], torch.ones(16), base)
    base[:, 0] = 1.
    second = atlas.reconstruction_loss(prediction, truth, ["a"], torch.ones(16), base)
    assert first.item() == pytest.approx((.25*.5**2)**2/16)
    assert second.item() == pytest.approx((1.*.5**2)**2/16)


def test_trusted_domains_and_rank_deficient_charts_keep_finite_gradients():
    atlas = _analytic_atlas()
    prediction = torch.zeros(1, 16, 1, 1, 1)
    prediction[:, 0] = 100.
    prediction.requires_grad_()
    truth = torch.zeros_like(prediction)
    loss = atlas.reconstruction_loss(prediction, truth, ["a"], torch.ones(16), truth)
    assert loss.item() == pytest.approx(((100-3)**2 + (.6*3**2)**2)/16)
    loss.backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad[:, 0].abs().sum() > 0
    constant = _bank()
    constant["residuals"].zero_()
    unsupported = _atlas(constant)
    assert not unsupported.pca_supported.any()
    zero_loss = unsupported.reconstruction_loss(prediction, truth, ["a"], torch.ones(16), truth)
    assert zero_loss.item() == 0.
    zero_loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_bank_rejects_nontrain_and_unaligned_pairs():
    class ValidationSet:
        split = "val"
    with pytest.raises(ValueError, match="TRAIN"):
        prepare_training_mls(ValidationSet(), None, {})
    payload = _bank()
    payload["provenance"]["split"] = "val"
    with pytest.raises(ValueError, match="TRAIN provenance"):
        _atlas(payload)
    payload = _bank()
    payload["base_coefficients"] = payload["base_coefficients"][:-1]
    with pytest.raises(ValueError, match="aligned"):
        _atlas(payload)


def test_parallel_bank_workers_preserve_paired_rows_and_provenance(tmp_path):
    from test_training import _write_case
    from vemgsbm.haar_residual import X1HaarResidualNetwork
    from vemgsbm.mri_data import PairedMRIPatchDataset

    root = tmp_path/"train"
    for index in range(2):
        _write_case(root, f"train_{index}", 13+index*17, size=16, textured=True)
    dataset = PairedMRIPatchDataset(root/"LQ/X4_png", root/"HQ_png", "train",
                                    scale=4, patch_size=(3, 8, 8), samples_per_volume=3)
    model = X1HaarResidualNetwork([4, 6, 8], use_band_whitening=False)
    options = dict(geometry_mode="paired_curved", seed=72, sites_per_case=20, patches_per_case=3)
    rng_before = torch.random.get_rng_state().clone()
    serial = prepare_training_mls(dataset, model, dict(options, bank=str(tmp_path/"serial.pt"), bank_workers=0))
    parallel = prepare_training_mls(dataset, model, dict(options, bank=str(tmp_path/"parallel.pt"), bank_workers=2))
    assert torch.equal(rng_before, torch.random.get_rng_state())
    for key in ("residuals", "base_coefficients", "band_scale", "case_indices"):
        torch.testing.assert_close(serial[key], parallel[key], rtol=0, atol=0)
    for key in ("identity", "provenance", "train_case_ids"):
        assert serial[key] == parallel[key]
    # Performance settings do not invalidate a previously built identical bank.
    cached = prepare_training_mls(dataset, model, dict(options, bank=str(tmp_path/"serial.pt"), bank_workers=2))
    torch.testing.assert_close(cached["residuals"], serial["residuals"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="bank_workers"):
        prepare_training_mls(dataset, model, dict(options, bank=str(tmp_path/"invalid.pt"), bank_workers=9))


def test_paired_curved_training_bank_checkpoint_and_deployment(tmp_path):
    from test_x1_bridge import x1_config
    from vemgsbm.runtime import build_models, load_checkpoint
    from vemgsbm.train import run_training

    cfg = x1_config(tmp_path)
    cfg.update(sigma_0=0., sigma_bridge=0., mls_reg_weight=.1)
    cfg["path"]["sigma"] = 0.
    cfg["sampling"].update(sigma=0., solver="euler")
    cfg["train"].update(objective="sr_reconstruction", bridge_aux_weight=0.)
    cfg["mls_regularizer"] = dict(bank=str(tmp_path/"paired-bank.pt"), geometry_mode="paired_curved",
                                  dim=2, folds=2, centers=2, neighbors=8, sites_per_case=16,
                                  patches_per_case=2, loss_sites_per_patch=8, query_chunk=4,
                                  max_fit_sites_per_fold=16)
    path = tmp_path/"paired.yaml"
    path.write_text(yaml.safe_dump(cfg))
    saved = load_checkpoint(run_training(path, device_name="cpu")/"last.pt")
    bank = saved["training_mls"]
    assert bank["identity"]["version"] == 2
    assert bank["provenance"]["split"] == "train"
    assert bank["train_case_ids"] == ["train_00", "train_01"]
    assert bank["base_coefficients"].shape == bank["residuals"].shape
    model, geometry = build_models(saved, torch.device("cpu"))
    assert geometry is None
    torch.testing.assert_close(bank["band_scale"], model.band_std+1.e-6)
    history = saved["epoch_history"][0]
    assert history["start_pixel_mse"] >= 0
    assert history["rollout_pixel_mse"] >= 0
    assert history["start_mls_reconstruction_mse"] >= 0
    assert history["rollout_mls_reconstruction_mse"] >= 0
