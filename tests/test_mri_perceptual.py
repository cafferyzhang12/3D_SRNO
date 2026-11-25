from types import SimpleNamespace

import numpy as np
import pytest
from scipy import linalg
import torch

from vemgsbm.mri_perceptual import FeatureMoments, consume_case, frechet_distance, rgb_inputs


def moments(x):
    result = FeatureMoments(x.shape[1])
    result.update(x)
    return result


def test_fid_moments_are_global_not_average_shard_covariances():
    x = np.random.default_rng(4).normal(size=(21, 6))
    shards = [moments(part) for part in np.array_split(x, 4)]
    merged = FeatureMoments(6)
    merged.n = sum(s.n for s in shards)
    merged.total = sum(s.total for s in shards)
    merged.outer = sum(s.outer for s in shards)
    mean, covariance = merged.statistics()
    np.testing.assert_allclose(mean, x.mean(0), atol=1.e-14)
    np.testing.assert_allclose(covariance, np.cov(x, rowvar=False), atol=1.e-14)
    assert frechet_distance(merged, moments(x)) < 1.e-12


def test_fid_matches_standard_gaussian_formula():
    rng = np.random.default_rng(3)
    x, y = moments(rng.normal(size=(60, 7))), moments(rng.normal(1, 2, size=(53, 7)))
    mx, cx = x.statistics()
    my, cy = y.statistics()
    expected = np.square(mx-my).sum() + np.trace(cx+cy-2*linalg.sqrtm(cx @ cy))
    assert frechet_distance(x, y) == pytest.approx(expected, abs=1.e-10)
    assert frechet_distance(y, x) == pytest.approx(expected, abs=1.e-10)


def test_fid_handles_rank_deficient_val_covariances():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(4, 16))
    y = x + 0.3
    assert frechet_distance(moments(x), moments(x)) < 2.e-6
    assert frechet_distance(moments(x), moments(y)) == pytest.approx(16*.3**2, abs=2.e-6)
    with pytest.raises(ValueError, match="two slices"):
        frechet_distance(moments(x[:1]), moments(y))


def test_gray_rgb_conversion_and_distinct_metric_ranges():
    plane = np.array([[[0., .5, 1.]]], np.float32)
    lpips, fid = rgb_inputs(plane, "cpu")
    assert lpips.dtype == torch.float32 and fid.dtype == torch.uint8
    assert lpips.shape == fid.shape == (1, 3, 1, 3)
    torch.testing.assert_close(lpips[0, 0, 0], torch.tensor([-1., 0., 1.]))
    assert fid[0, 0, 0].tolist() == [0, 128, 255]
    for channel in (1, 2):
        torch.testing.assert_close(lpips[:, 0], lpips[:, channel])
        torch.testing.assert_close(fid[:, 0], fid[:, channel])


@pytest.mark.parametrize("value", [float("nan"), -0.01, 1.01])
def test_metric_inputs_reject_bad_ranges(value):
    with pytest.raises(ValueError, match="finite"):
        rgb_inputs(np.full((2, 8, 8), value), "cpu")


def test_complete_paired_slices_consumed_once_with_uneven_batches(monkeypatch):
    import vemgsbm.mri_perceptual as module
    planes = np.random.default_rng(8).random((5, 8, 8)).astype(np.float32)
    case = SimpleNamespace(hq=list(range(5)), scale=4, lr_shape=(2, 2),
                           normalization_scale=1., normalization_offset=0., channel_policy="gray", bits=8)
    reads = []
    def read(path, *_):
        reads.append(path)
        return planes[path]
    monkeypatch.setattr(module, "read_plane", read)
    sizes = []
    def paired(x, y):
        sizes.append(len(x))
        return (x-y).square().mean((1, 2, 3), keepdim=True)
    def features(x):
        assert x.dtype == torch.uint8
        return (x.double().mean((2, 3)),)
    p, y = FeatureMoments(3), FeatureMoments(3)
    row = consume_case(planes, case, paired, features, p, y, "cpu", 2)
    assert reads == list(range(5)) and sizes == [2, 2, 1]
    assert row == {"slice_count": 5, "lpips_sum": 0., "lpips": 0.}
    assert p.n == y.n == 5
    assert frechet_distance(p, y) < 1.e-5


def test_offline_weight_verification_rejects_missing_assets(monkeypatch, tmp_path):
    import vemgsbm.mri_perceptual as module
    monkeypatch.setattr(module, "_packages", lambda: {})
    def missing(_):
        raise FileNotFoundError("missing pretrained assets")
    monkeypatch.setattr(module, "_asset_identity", missing)
    with pytest.raises(FileNotFoundError, match="pretrained"):
        module.load_extractors(tmp_path, "cpu")
