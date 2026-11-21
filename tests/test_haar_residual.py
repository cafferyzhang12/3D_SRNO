"""XY phase alignment, native LR deployment and complete A/B entry points."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from vemgsbm.haar_residual import HaarResidualNetwork
from vemgsbm.mri_infer import infer_mri, make_mri_tile
from vemgsbm.runtime import build_models, load_checkpoint
from vemgsbm.train import run_training
from test_training import _config


def test_tile_native_lr_halo_phase_and_short_run():
    planes = np.random.default_rng(133).random((4, 9, 7), dtype=np.float32)
    case = SimpleNamespace(lq=range(4), lr_shape=(9, 7))
    base, mask, lr = make_mri_tile(case, (1, 3), (0, 12, 4), (5, 32, 24), 4,
                                   reader=planes.__getitem__, with_native_lr=True)
    native = torch.from_numpy(planes[1:3, 3:, 1:])[None, None]
    torch.testing.assert_close(lr, F.pad(native, (0, 0, 0, 2, 0, 3), mode="replicate"))
    full = F.interpolate(torch.from_numpy(planes[1:3])[:, None], scale_factor=4, mode="bilinear",
                         align_corners=False)
    torch.testing.assert_close(base[0, 0, :2, :24], full[:, 0, 12:, 4:])
    assert not mask.any()
    model = HaarResidualNetwork("haar_direct", [4, 8, 12])
    torch.testing.assert_close(model.codec.encode(base)[:, :, :2, :6],
                               model.codec.encode(full.transpose(0, 1).unsqueeze(0))[:, :, :, 3:, 1:])
    with pytest.raises(ValueError, match="align"):
        make_mri_tile(case, (1, 3), (0, 12, 2), (5, 32, 24), 4,
                      reader=planes.__getitem__, with_native_lr=True)


@pytest.mark.parametrize("mode,weight", [("haar_direct", 1.), ("haar_bridge", 1.), ("haar_bridge", 0.)])
def test_residual_checkpoint_and_full_case_val(tmp_path, mode, weight):
    cfg = _config(tmp_path)
    cfg.pop("representation")
    cfg.update(mode=mode, model={"widths": [4, 6, 8]})
    cfg["train"].update(selection_metric="pooled_psnr", endpoint_weight=weight, bridge_epochs=1)
    cfg["sampling"]["readout"] = "endpoint" if weight else "integral"
    cfg["path"].update(optimize_std=True, state_weight=0., metric_weight=0.)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf8")
    run = run_training(path, stage="bridge", device_name="cpu")
    checkpoint = load_checkpoint(run / "best.pt")
    model, _ = build_models(checkpoint, torch.device("cpu"))
    assert model.mode == mode and checkpoint["format_version"] == 3
    whole = json.loads((run / "whole_val" / "epoch_0001.json").read_text())
    assert whole["val_pixel_count"] == 2 * 3 * 8 * 8
    for readout in (["integral", "endpoint"] if weight and mode == "haar_bridge" else [None]):
        roots = cfg["data"]["val"]
        dest = tmp_path / f"val_{readout}"
        infer_mri(run / "best.pt", roots["root_lq"], dest, root_hq=roots["root_hq"],
                  device_name="cpu", split="val", readout=readout)
        summary = json.loads((dest / "summary.json").read_text())
        assert summary["total_slices"] == 6
        if readout in (None, cfg["sampling"]["readout"]):
            assert summary["all_psnr_db"] == pytest.approx(whole["val_pooled_psnr_db"])
            assert summary["mean_slice_ssim"] == pytest.approx(whole["val_mean_slice_ssim"])

def _four_rank_train(rank, rendezvous, paths):
    from datetime import timedelta
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=120))
    try:
        for path in paths:
            run_training(path, stage="bridge", device_name="cpu")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo unavailable")
def test_four_rank_haar_training(tmp_path):
    cfg = _config(tmp_path)
    cfg.pop("representation")
    cfg["train"].update(samples_per_epoch=8, batch_size=1, selection_metric="pooled_psnr", bridge_epochs=1)
    cfg["model"] = {"widths": [4, 6, 8]}
    cfg["path"].update(iterations=0, optimize_std=True)
    paths = []
    for mode in ["haar_direct", "haar_bridge"]:
        cfg.update(mode=mode, run_dir=str(tmp_path / mode))
        path = tmp_path / (mode + ".yaml")
        path.write_text(yaml.safe_dump(cfg), encoding="utf8")
        paths.append(str(path))
    torch.multiprocessing.spawn(_four_rank_train, args=((tmp_path / "store").resolve().as_uri(), paths), nprocs=4)
    for mode in ["haar_direct", "haar_bridge"]:
        metrics = json.loads((tmp_path / mode / "bridge" / "metrics.jsonl").read_text())
        assert metrics["train_items"] == 8 and metrics["world_size"] == 4
        assert metrics["val_case_count"] == 2 and metrics["val_pixel_count"] == 384
