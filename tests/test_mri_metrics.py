import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import gaussian_laplace
from skimage.metrics import structural_similarity
import torch

from vemgsbm.mri_metrics import mean_xy_ssim
from vemgsbm.mri_compare import paired_comparison
from vemgsbm.mri_data import index_cases, read_plane
from vemgsbm.mri_infer import case_metrics


def test_validation_xy_ssim_matches_full_case_slice_metric():
    generator = torch.Generator().manual_seed(17)
    reference = torch.rand((2, 1, 3, 12, 10), dtype=torch.float64, generator=generator)
    prediction = (reference + torch.rand(reference.shape, generator=generator) * .2).clamp(0, 1)
    slices = zip(reference[:, 0].flatten(0, 1).numpy(), prediction[:, 0].flatten(0, 1).numpy())
    expected = np.mean([structural_similarity(x, y, data_range=1., win_size=7) for x, y in slices])
    np.testing.assert_allclose(float(mean_xy_ssim(prediction, reference)), expected, atol=1e-12)


def test_case_regions_hfen_and_z_skip_quarantined_gaps(tmp_path):
    for root in ("lr", "hr"):
        (tmp_path / root / "one").mkdir(parents=True)
    plane = np.tile([0, 2, 3, 17, 54, 63, 80, 91], (8, 1)).astype("uint8")
    for z in (0, 1, 3):
        Image.fromarray(plane[::4, ::4]).save(tmp_path / "lr" / "one" / f"{z}.png")
        Image.fromarray(plane).save(tmp_path / "hr" / "one" / f"{z}.png")
    case, = index_cases(tmp_path / "lr", tmp_path / "hr", 4)
    ref = read_plane(case.hq[0], 255).astype(np.float64)
    pred = np.stack([ref, ref * .5, ref * .25])
    row = case_metrics(pred, case)
    assert row["foreground_voxels"] == 3 * 8 * 6
    assert row["z_difference_voxels"] == 64
    assert row["z_difference_mse"] == pytest.approx(np.square(ref * .5).mean())
    filt = gaussian_laplace(ref, 1.5)
    expected = np.sqrt(sum(np.square(gaussian_laplace(s - ref, 1.5)).sum() for s in pred)
                       / (3 * np.square(filt).sum()))
    assert row["hfen"] == pytest.approx(expected)


def test_bootstrap_resamples_paired_case_clusters_and_recomputes_pooled_metrics():
    def arm(factor, offset):
        cases = []
        for i in (1, 4):
            metric = {"squared_error_sum": .1 * i**2 * factor, "pixel_count": 16 * i,
                      "mean_slice_ssim": .1 * i + offset, "ssim_slice_count": i, "evaluated_slices": i}
            cases.append({"case_id": str(i), "input_fingerprint": str(i), "metrics": metric})
        return {"cases": cases, "subset": {"case_ids": ["1", "4"]}}

    left, right = arm(1, 0), arm(.5, .02)
    right["cases"].reverse()
    diff = paired_comparison(left, right, resamples=100)
    np.testing.assert_allclose(diff["pooled_psnr_db"]["ci95"], 10 * np.log10(2))
    np.testing.assert_allclose(diff["mean_slice_ssim"]["ci95"], .02)
    right["cases"][1]["input_fingerprint"] = "different"
    with pytest.raises(ValueError, match="fingerprints differ"):
        paired_comparison(left, right)
