import numpy as np
from skimage.metrics import structural_similarity
import torch

from vemgsbm.mri_metrics import mean_xy_ssim


def test_validation_xy_ssim_matches_full_case_slice_metric():
    generator = torch.Generator().manual_seed(17)
    reference = torch.rand((2, 1, 3, 12, 10), dtype=torch.float64, generator=generator)
    prediction = (reference + torch.rand(reference.shape, generator=generator) * .2).clamp(0, 1)
    slices = zip(reference[:, 0].flatten(0, 1).numpy(), prediction[:, 0].flatten(0, 1).numpy())
    expected = np.mean([structural_similarity(x, y, data_range=1., win_size=7) for x, y in slices])
    np.testing.assert_allclose(float(mean_xy_ssim(prediction, reference)), expected, atol=1e-12)
