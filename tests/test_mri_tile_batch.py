"""Changing inference tile batch size preserves Haar phase and stochastic streams."""
import numpy as np
import torch

from test_training import _config, _write_case
from vemgsbm.haar_residual import HaarResidualNetwork
from vemgsbm.mri_data import index_cases
from vemgsbm.mri_infer import _infer_case


def test_haar_tile_batch_matches_serial_sampling(tmp_path):
    config = _config(tmp_path)
    source = tmp_path / "paired"
    _write_case(source, "one", 30, size=32, textured=True)
    case, = index_cases(source / "LQ/X4_png", source / "HQ_png", scale=4)
    config.update(mode="haar_bridge", inference_variant="endpoint_safe")
    config["data"]["patch_size"] = [3, 16, 16]
    torch.manual_seed(74)
    model = HaarResidualNetwork("haar_bridge", [4, 8, 12], "integral").eval()
    torch.nn.init.normal_(model.drift.weight, std=.002)
    outputs = []
    for size in (1, 4):
        config["sampling"]["tile_batch"] = size
        dest = tmp_path / f"batch_{size}"
        _infer_case(case, dest, model, None, config, torch.device("cpu"), 17, 3, 1)
        outputs.append(np.load(dest / "mean.npy"))
    np.testing.assert_allclose(*outputs, rtol=1.e-5, atol=1.e-6)


def test_volume_predictor_receives_native_3d_tiles_and_clips_after_fusion(tmp_path):
    config = _config(tmp_path)
    source = tmp_path / "paired"
    _write_case(source, "one", 30, size=32, textured=True)
    case, = index_cases(source / "LQ/X4_png", source / "HQ_png", scale=4)
    config["data"]["patch_size"] = [5, 16, 16]
    config["sampling"]["tile_batch"] = 4
    calls = []

    def predictor(native_lr, generator):
        assert native_lr.shape[1:] == (1, 5, 4, 4)
        assert generator.initial_seed() == 42
        calls.append(len(native_lr))
        return torch.full((len(native_lr), 1, 5, 16, 16), 1.5)

    dest = tmp_path / "volume"
    shape, metrics = _infer_case(case, dest, None, None, config, torch.device("cpu"), 42, 1, 1,
                                 volume_predictor=predictor)
    assert any(size == 4 for size in calls)
    assert shape == (len(case.lq), 32, 32) and metrics["evaluated_slices"] == len(case.lq)
    np.testing.assert_array_equal(np.load(dest / "mean.npy"), np.ones(shape))
