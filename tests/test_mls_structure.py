"""Structural sampling and target-anchored XY chart discrepancy tests."""
import pytest
import torch

from test_mls_reconstruction import _analytic_atlas, _atlas
from vemgsbm.spatial_atlas import _stratified_sites, _structure_labels, _xy_neighbor_pairs


def test_stratification_covers_texture_weak_edge_and_background_without_duplicates():
    labels = torch.arange(4).repeat_interleave(100)
    torch.manual_seed(27)
    indices = _stratified_sites(labels, 100, (.1, .2, .5, .2))
    assert len(indices.unique()) == 100
    assert torch.bincount(labels[indices], minlength=4).tolist() == [10, 20, 50, 20]
    # Absent strata reassign quotas without duplicates or failure on black HR.
    black = torch.zeros(79, dtype=torch.long)
    indices = _stratified_sites(black, 64, (.1, .2, .5, .2))
    assert len(indices) == len(indices.unique()) == 64
    assert torch.equal(_stratified_sites(black, 100, (.1, .2, .5, .2)), torch.arange(79))


def test_hr_structure_labels_are_detached_and_never_mix_z():
    first = torch.full((1, 1, 1, 16, 16), .1)
    second = torch.zeros_like(first)
    reference = torch.cat((first, second), 2).requires_grad_()
    labels = _structure_labels(reference, (2, 4, 4)).reshape(2, 4, 4)
    assert not labels.requires_grad
    assert (labels[0] == 1).all()  # constant foreground: weak, no slice-border edge
    assert (labels[1] == 0).all()
    with pytest.raises(ValueError, match="aligned"):
        _structure_labels(reference, (2, 3, 4))


def test_xy_pairs_cannot_cross_rows_or_slices():
    a, b = _xy_neighbor_pairs(torch.arange(12), (2, 2, 3))
    assert len(a) == 14  # each 2x3 slice has 7 undirected XY neighbor pairs
    assert torch.equal(a//6, b//6)
    xy = b-a == 1
    assert torch.equal(a[xy]//3, b[xy]//3)
    assert torch.all((b-a == 1) | (b-a == 3))


def test_neighbor_defect_uses_curvature_and_backpropagates_through_coordinates():
    atlas = _analytic_atlas()
    atlas.neighbor_weight = 1.
    target = torch.zeros(1, 16, 1, 1, 2, requires_grad=True)
    prediction = target.detach().clone()
    prediction[0, 0, 0, 0] = torch.tensor([.2, .5])
    prediction.requires_grad_()
    terms = atlas.reconstruction_terms(prediction, target, ["a"], torch.ones(16), target)
    expected = ((.6*.5**2-.6*.2**2)/2)**2/16
    assert terms["neighbor_mse"].item() == pytest.approx(expected)
    terms["neighbor_mse"].backward()
    assert prediction.grad[:, 0].abs().sum() > 0
    assert torch.isfinite(prediction.grad).all() and target.grad is None
    # A varying prediction that follows the chart exactly has no curved normal
    # discrepancy, despite nonzero ordinary adjacent image/coefficient errors.
    curved = prediction.detach().clone()
    curved[:, 2] = .6*curved[:, 0].square()
    terms = atlas.reconstruction_terms(curved, target, ["a"], torch.ones(16), target)
    assert terms["local_mse"] < 1.e-12 and terms["neighbor_mse"] < 1.e-12
    assert (curved[..., 1:]-curved[..., :-1]).square().sum() > 0


def test_neighbor_loss_preserves_truth_and_does_not_penalize_z_changes():
    atlas = _analytic_atlas()
    atlas.neighbor_weight = .1
    target = torch.zeros(1, 16, 2, 2, 3)
    prediction = target.clone().requires_grad_()
    loss = atlas.reconstruction_loss(prediction, target, ["a"], torch.ones(16), target)
    assert loss.item() == 0.
    loss.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))
    prediction = target.clone()
    prediction[:, 3, 1] = .4  # constant XY normal offset only on the second slice
    terms = atlas.reconstruction_terms(prediction, target, ["a"], torch.ones(16), target)
    assert terms["local_mse"] > 0 and terms["neighbor_mse"] == 0
    assert terms["neighbor_pairs"] == 14


def test_stratified_reconstruction_requires_truth_image_and_reports_coverage():
    atlas = _atlas(loss_sampling="structure_stratified")
    target = torch.zeros(1, 16, 1, 2, 3)
    prediction = target.clone().requires_grad_()
    with pytest.raises(ValueError, match="reference_image"):
        atlas.reconstruction_loss(prediction, target, ["a"], torch.ones(16), target)
    reference = torch.full((1, 1, 1, 8, 12), .2, requires_grad=True)
    terms = atlas.reconstruction_terms(prediction, target, ["a"], torch.ones(16), target, reference)
    assert terms["sample_fraction_weak"] == 1
    assert terms["sampled_sites"] == 6 and terms["neighbor_pairs"] == 0
    (terms["local_mse"]+terms["neighbor_mse"]).backward()
    assert reference.grad is None


@pytest.mark.parametrize("settings", [{"loss_sampling": "mystery"}, {"neighbor_weight": -1},
                                      {"neighbor_weight": float("nan")},
                                      {"structure_fractions": [.1, .2, .3, .3]}])
def test_rejects_invalid_structure_settings(settings):
    with pytest.raises(ValueError):
        _atlas(**settings)
