import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vemgsbm import mri_perceptual, mri_v2_train
from vemgsbm.runtime import fingerprint, load_checkpoint


class TinyDataset:
    all_cases = ()

    def __init__(self):
        self.loaded = []

    def __len__(self):
        return 5

    def __getitem__(self, key):
        epoch, index = key
        self.loaded.append((epoch, index))
        return {"lr": torch.tensor(0.1 * index), "hr": torch.tensor(0.2 * (index + epoch)), "id": "train"}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def set_band_statistics(self, stats):
        pass


class TinyObjective(torch.nn.Module):
    has_unused_parameters = False

    def __init__(self, model, cfg, atlas, paired):
        super().__init__()
        self.model = model

    def forward(self, lr, hr, case_ids):
        perturbation = 0.1 * (torch.rand(()) + random.random() + np.random.rand())
        loss = (lr * self.model.weight + perturbation - hr).square().mean()
        return {"loss": loss, "error": loss.detach().sqrt()}


def test_mid_epoch_resume_restores_used_patches_rng_optimizer_and_ema(tmp_path, monkeypatch):
    setup = dict(epochs=2, batch_size=2, workers=0, accumulation_steps=2,
                 seed=17, threads=torch.get_num_threads(), learning_rate=0.1, warmup_epochs=0,
                 clip_grad_norm=1., baseline_psnr=32., save_every_updates=1,
                 val_every_epochs=2, val_final_epochs=0)
    run_dir = tmp_path / "uninterrupted"
    cfg = dict(run_dir=str(run_dir), train=setup, loss={}, perceptual={"weights_dir": str(tmp_path)},
               base_interpolation="bicubic", data={"patch_size": [1, 16, 16]})
    dataset = TinyDataset()
    val_epochs = []
    monkeypatch.setattr(mri_v2_train, "read_config", lambda _: cfg)
    monkeypatch.setattr(mri_v2_train, "make_model", lambda _: TinyModel())
    monkeypatch.setattr(mri_v2_train, "V2Objective", TinyObjective)
    monkeypatch.setattr(mri_v2_train, "training_data",
                        lambda *a, **kw: (dataset, dataset, "digest", [{"split": "train"}]))
    monkeypatch.setattr(mri_v2_train, "prepare_train_geometry", lambda *a: {
        "statistics": {}, "jetatlas": None, "contract": mri_v2_train.geometry_contract(cfg, "digest"),
    })
    monkeypatch.setattr(mri_perceptual, "load_extractors", lambda *a: (torch.nn.Identity(), None))

    def validate(model, *args):
        val_epochs.append(float(model.weight))
        return dict(psnr=32., ssim=.95, lpips=.01, fid=1.), []

    monkeypatch.setattr(mri_v2_train, "validate", validate)
    expected = mri_v2_train.run("ignored", device="cpu")
    full_order = dataset.loaded[:]
    complete = load_checkpoint(run_dir / "last.pt")
    assert len(val_epochs) == 1
    assert not (run_dir / "val_epoch_0001.json").exists()
    assert (run_dir / "val_epoch_0002.json").exists()

    run_dir = tmp_path / "resumed"
    cfg["run_dir"] = str(run_dir)
    dataset.loaded.clear()
    atomic_save = mri_v2_train.atomic_save

    def interrupt_after_save(value, path, **kwargs):
        atomic_save(value, path, **kwargs)
        if path.name == "last.pt" and value["next_batch"] == 2:
            raise InterruptedError("simulated shutdown after durable checkpoint")

    with monkeypatch.context() as patch:
        patch.setattr(mri_v2_train, "atomic_save", interrupt_after_save)
        with pytest.raises(InterruptedError, match="simulated shutdown"):
            mri_v2_train.run("ignored", device="cpu")
    before_resume = load_checkpoint(run_dir / "last.pt")
    assert (before_resume["epoch"], before_resume["next_batch"], before_resume["ema_updates"]) == (0, 2, 1)
    assert len(dataset.loaded) == 4

    assert mri_v2_train.run("ignored", device="cpu", resume=run_dir / "last.pt") == expected
    assert dataset.loaded == full_order
    after_resume = load_checkpoint(run_dir / "last.pt")
    assert after_resume["ema_updates"] == complete["ema_updates"] == 4
    assert after_resume["history"] == complete["history"]
    for key in ("model", "ema_model", "optimizer", "scheduler"):
        assert after_resume[key] == complete[key]

    best = run_dir / "best.pt"
    mri_v2_train.run("ignored", device="cpu", resume=best, evaluate=True)
    assert json.loads((run_dir / "final_val.json").read_text())["checkpoint_sha256"] == fingerprint(best)


def test_perceptual_observer_reuses_decoded_hq_without_changing_metrics(monkeypatch):
    paths = ("a", "b", "c")
    volume = np.stack([np.full((8, 8), 0.1 * i, dtype=np.float32) for i in range(3)])
    reads = []
    observations = []
    case = SimpleNamespace(hq=paths, scale=2, lr_shape=(4, 4), normalization_scale=255,
                           normalization_offset=0, channel_policy="native", bits=8)

    def read_plane(path, *args):
        reads.append(path)
        return volume[paths.index(path)]

    monkeypatch.setattr(mri_perceptual, "read_plane", read_plane)
    paired = lambda pred, truth: (pred - truth).square().mean((1, 2, 3))
    features = lambda values: (values.float().mean((2, 3)),)
    pred = mri_perceptual.FeatureMoments(3)
    ref = mri_perceptual.FeatureMoments(3)
    metrics = mri_perceptual.consume_case(
        volume, case, paired, features, pred, ref, "cpu", 2,
        observe=lambda p, y: observations.append((p.copy(), y.copy())),
    )
    assert reads == list(paths)
    for p, y in observations:
        np.testing.assert_array_equal(p, y)
    assert metrics == {"lpips_sum": 0., "slice_count": 3, "lpips": 0.}
    np.testing.assert_array_equal(pred.outer, ref.outer)
