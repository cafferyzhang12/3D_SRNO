"""Small real-PNG acceptance tests for the new geometry-to-GSBM pipeline."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import torch
import yaml

from vemgsbm import train
from vemgsbm.datasets import training_data
from vemgsbm.mri_infer import infer_mri
from vemgsbm.mri_inventory import prepare_inventory
from vemgsbm.runtime import load_checkpoint
from vemgsbm.atlas_geometry import MRIAtlasGeometry
from vemgsbm.momentum.train import run_training as run_momentum


def _write_case(root, cid, value, size=8, textured=False):
    hq, lq = root / "HQ_png" / cid, root / "LQ" / "X4_png" / cid
    hq.mkdir(parents=True)
    lq.mkdir(parents=True)
    for z in range(3):
        # Different local structure across cases, with known uint8 PNG data.
        yy, xx = np.mgrid[:size, :size]
        high = value + yy * 7 + xx * 3 + z * 11
        if textured:
            high = high + (value % 17) * xx**2 + yy**2 * (z + 3) + xx * yy
        high = (high % 240).astype("uint8")
        low = np.asarray(Image.fromarray(high).resize((size // 4, size // 4), Image.Resampling.BILINEAR))
        Image.fromarray(high).save(hq / f"{z:03}.png")
        Image.fromarray(low).save(lq / f"{z:03}.png")


def _config(tmp_path):
    roots = {}
    for split, count in (("train", 2), ("val", 2), ("test", 20)):
        root = tmp_path / "data" / split
        for index in range(count):
            _write_case(root, f"{split}_{index:02}", index * 17 + 13)
        roots[split] = {"root_lq": str(root / "LQ" / "X4_png"), "root_hq": str(root / "HQ_png")}
    return {
        "mode": "gsbm", "run_dir": str(tmp_path / "run"),
        "data": {"kind": "paired_mri_png", "degradation": "paired_xy", "scale": 4,
                 "patch_size": [3, 8, 8], "scan_workers": 1,
                 "channel_policy": "identical_rgb_to_gray", **roots,
                 "scan_cache": {s: str(tmp_path / f"{s}_index.json") for s in roots}},
        "representation": {"width": 2, "feature_dim": 4, "clusters": 2,
                           "bank_samples": 4, "checkpoint": str(tmp_path / "run" / "representation" / "best.pt")},
        "model": {"width": 2},
        "train": {"epochs": 2, "representation_epochs": 2, "bridge_epochs": 1,
                  "batch_size": 2, "samples_per_epoch": 4, "val_samples_per_epoch": 2,
                  "train_samples": 1, "val_samples": 1, "seed": 42,
                  "workers": 0, "threads": 1, "learning_rate": 0.001,
                  "matching_time_min": 0.02},
        "path": {"knots": 3, "iterations": 1, "quadrature_points": 2,
                 "sigma": 0.05, "learning_rate": 0.01,
                 "control_weight": 1., "state_weight": 0.1, "metric_weight": 0.05},
        "sampling": {"steps": 2, "sigma": 0.05},
    }


def _save_config(cfg, path):
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


@pytest.mark.parametrize(("mode", "guided"), [("gsbm", False), ("gsbm", True), ("momentum", False)])
def test_mls_atlas_optimizes_paired_png_bridge(tmp_path, mode, guided):
    cfg = _config(tmp_path)
    for index in range(6):
        _write_case(tmp_path / "data" / "train", f"extra_{index}", index * 29, size=16)
    cfg["mode"] = mode
    cfg["train"].update(epochs=1, representation_epochs=1, samples_per_epoch=24)
    cfg["representation"].update(bank_samples=24)
    cfg["atlas"] = {
        "charts": 2, "neighborhoods": [8, 16, 24], "dim_candidates": [2],
        "mls": {"degree": 1, "neighbors": 16, "anchor_limit": 4},
    }
    if guided:
        cfg["atlas"]["source_routing"] = True
        cfg["path"]["geometry_ramp_power"] = 2.0
        cfg["train"]["metric_matching_weight"] = 0.05
        cfg["sampling"].update(steps=3, geometry_guidance={"strength": 0.02, "start": 0.3, "max_update": 0.002})
    if mode == "momentum":
        cfg["path"].update(quadrature=2, lr=0.02, state_weight=0.01,
                           normal_weight=0.01, max_shift=0.5)
    config_path = tmp_path / "config.yaml"
    _save_config(cfg, config_path)
    inventory = None
    if mode == "gsbm":
        inventory = tmp_path / "atlas_cases.json.gz"
        prepare_inventory(config_path, inventory)
    representation = train.run_training(config_path, "representation", "cpu", case_inventory=inventory)
    payload = load_checkpoint(representation / "best.pt")["metric"]
    assert payload["kind"] == "mls"
    assert payload["diagnostics"]["train_features"] == 24
    geometry = MRIAtlasGeometry(payload)
    state = torch.rand(2, 1, 3, 8, 8, requires_grad=True)
    chart = geometry.bind(state.detach())
    if guided:
        position = chart.state_cost(state).mean()
        velocity = chart.drift_cost(state, torch.randn_like(state)).mean()
    else:
        position, velocity = chart.phase_cost(state, torch.randn_like(state))
    (position + velocity).backward()
    assert torch.isfinite(state.grad).all() and state.grad.abs().sum() > 0
    assert all(p.grad is None for p in geometry.parameters())
    if mode == "gsbm":
        output = train.run_training(config_path, "bridge", "cpu", case_inventory=inventory)
        diagnostics = ("path_state_after", "path_metric_after")
        field = "metric"
    else:
        output = run_momentum(config_path, "cpu")
        diagnostics = ("path_normal_position", "path_normal_velocity")
        field = "atlas"
    report = json.loads((output / "metrics.jsonl").read_text(encoding="utf-8"))
    assert all(report[key] > 0 for key in diagnostics)
    checkpoint = load_checkpoint(output / "best.pt")
    assert checkpoint[field]["diagnostics"]["atlas_version"] == payload["diagnostics"]["atlas_version"]
    if guided:
        assert report["metric_matching"] > 0
        assert "validation_without_online_guidance" in report
        roots = cfg["data"]["test"]
        out = tmp_path / "guided_test"
        infer_mri(output / "best.pt", roots["root_lq"], out, root_hq=roots["root_hq"],
                  device_name="cpu", case_inventory=inventory, save_volume_limit=0,
                  inference_variant="endpoint_safe")
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        assert summary["subset"]["selected_cases"] == 3
        assert summary["geometry_guidance"] == cfg["sampling"]["geometry_guidance"]


@pytest.fixture
def trained_pipeline(tmp_path, monkeypatch):
    cfg, config_path = _config(tmp_path), tmp_path / "config.yaml"
    _save_config(cfg, config_path)
    fits = []
    original = train.fit_metric

    def checked_fit(encoder, features, **kwargs):
        assert all(p.requires_grad for p in encoder.parameters())
        fitted = original(encoder, features, **kwargs)
        # Catch the historical pitfall where fitting freezes the live learner
        # and silently disables representation training in the next epoch.
        assert all(p.requires_grad for p in encoder.parameters())
        assert all(not p.requires_grad for p in fitted.parameters())
        fits.append(features.detach().cpu().clone())
        return fitted

    monkeypatch.setattr(train, "fit_metric", checked_fit)
    representation_dir = train.run_training(config_path, "representation", "cpu")
    assert len(fits) == 2
    assert not torch.equal(fits[0], fits[1])
    assert load_checkpoint(representation_dir / "best.pt")["metric"] is not None
    assert load_checkpoint(representation_dir / "last.pt")["epoch"] == 2
    assert not (representation_dir / ".training.lock").exists()
    bridge_dir = train.run_training(config_path, "bridge", "cpu")
    return cfg, config_path, bridge_dir


def test_representation_bridge_resume_and_manifest_accounting(trained_pipeline):
    cfg, config_path, bridge_dir = trained_pipeline
    first = load_checkpoint(bridge_dir / "last.pt")
    assert first["epoch"] == 1 and first["training_stage"] == "bridge"
    assert first["stage"] == "reconstruction" and first["metric"] is not None
    cfg["train"]["bridge_epochs"] = 2
    _save_config(cfg, config_path)
    train.run_training(config_path, "bridge", "cpu", str(bridge_dir / "last.pt"))
    resumed = load_checkpoint(bridge_dir / "last.pt")
    assert resumed["epoch"] == 2
    assert resumed["manifest_sha256"] == first["manifest_sha256"]
    assert resumed["data_identity"] == first["data_identity"]
    assert any(not torch.equal(first["model"][key], resumed["model"][key]) for key in first["model"])
    metrics = [json.loads(line) for line in (bridge_dir / "metrics.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in metrics] == [1, 2]
    assert all(row["train_items"] == 4 and row["validation_items"] == 2 for row in metrics)
    assert all(row["path_objective_after"] <= row["path_objective_before"] for row in metrics)
    assert not (bridge_dir / ".training.lock").exists()


def test_inference_fixed_15_percent_and_prediction_independent_of_hq(trained_pipeline, tmp_path):
    cfg, _, bridge_dir = trained_pipeline
    checkpoint = str(bridge_dir / "best.pt")
    roots = cfg["data"]["test"]
    with_hq, without_hq = tmp_path / "with_hq", tmp_path / "without_hq"
    common = {"device_name": "cpu", "test_fraction": 0.15, "subset_seed": 20261110,
              "seed": 17, "steps": 2}
    infer_mri(checkpoint, roots["root_lq"], with_hq, root_hq=roots["root_hq"], **common)
    infer_mri(checkpoint, roots["root_lq"], without_hq, root_hq=None, **common)
    evaluated = json.loads((with_hq / "summary.json").read_text())
    unlabelled = json.loads((without_hq / "summary.json").read_text())
    assert evaluated["subset"]["available_cases"] == 20
    assert evaluated["subset"]["selected_cases"] == 3
    assert evaluated["total_slices"] == 9
    assert evaluated["subset"] == unlabelled["subset"]
    assert len(evaluated["cases"]) == len(unlabelled["cases"]) == 3
    assert "all_mse" in evaluated and "all_mse" not in unlabelled
    for case in evaluated["cases"]:
        cid = case["case_id"]
        np.testing.assert_array_equal(np.load(with_hq / cid / "mean.npy"),
                                      np.load(without_hq / cid / "mean.npy"))
        metadata = json.loads((with_hq / cid / "metadata.json").read_text())
        assert metadata["shape_zyx"] == [3, 8, 8]
        assert metadata["scale_z"] == 1 and metadata["scale_xy"] == 4
        assert metadata["stems"] == ["000", "001", "002"]
        assert case["metrics"]["evaluated_slices"] == 3
    assert not with_hq.with_name(with_hq.name + ".inference.lock").exists()


def test_prepared_inventory_reuses_pairing_and_test_subset(trained_pipeline, tmp_path, monkeypatch):
    cfg, config, bridge = trained_pipeline
    direct = training_data(cfg)
    inventory = tmp_path / "cases.json.gz"
    prepare_inventory(config, inventory)

    def unexpected_scan(*args, **kwargs):
        pytest.fail("A prepared inventory should avoid rescanning MRI PNGs")

    monkeypatch.setattr("vemgsbm.mri_data.index_cases", unexpected_scan)
    monkeypatch.setattr("vemgsbm.mri_infer.index_cases", unexpected_scan)
    prepared = training_data(cfg, inventory)
    assert prepared[2:] == direct[2:]
    for split in (0, 1):
        assert prepared[split].all_cases == direct[split].all_cases
        torch.testing.assert_close(prepared[split][0]["hr"], direct[split][0]["hr"])
    roots = cfg["data"]["test"]
    out = tmp_path / "inventory_test"
    infer_mri(bridge / "best.pt", roots["root_lq"], out, root_hq=roots["root_hq"],
              device_name="cpu", steps=2, save_volume_limit=0, case_inventory=inventory)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["subset"]["selected_cases"] == 3
    assert summary["total_slices"] == 9
    cfg["data"]["debug_case_limit"] = 1
    with pytest.raises(ValueError, match="data contract differ"):
        training_data(cfg, inventory)


def test_chart_momentum_trains_and_tests_ten_percent_of_paired_png_cases(tmp_path):
    cfg = _config(tmp_path)
    for index in range(6):
        _write_case(tmp_path / "data" / "train", f"texture_{index}", index * 29 + 5, size=16, textured=True)
    cfg.update(mode="momentum", chart_momentum={},
               atlas={"charts": 2, "bank_patches": 8, "neighborhoods": [8, 16, 48],
                      "dim_candidates": [2], "mls": {"degree": 1, "neighbors": 24, "anchor_limit": 4},
                      "source_routing": {"candidates": 2}},
               path={"sigma": .05, "lr": .02, "iterations": 1, "quadrature": 2,
                     "state_weight": .005, "normal_weight": .002, "max_shift": .5})
    cfg["train"].update(epochs=1, samples_per_epoch=8)
    cfg["test"] = {"fraction": .1, "subset_seed": 20261110}
    path = tmp_path / "momentum.yaml"
    _save_config(cfg, path)
    inventory = tmp_path / "momentum_cases.json.gz"
    prepare_inventory(path, inventory)
    output = run_momentum(path, "cpu", case_inventory=inventory)
    checkpoint = output / "best.pt"
    saved = load_checkpoint(checkpoint)
    assert saved["algorithm"] == "chart_trivialized_momentum_bridge"
    assert saved["atlas"]["kind"] == "haar_mls"
    assert saved["atlas"]["diagnostics"]["source_routing_train_pairs"] == 64
    training = json.loads((output / "metrics.jsonl").read_text(encoding="utf-8"))
    assert training["path_normal_velocity"] > 0
    assert training["path_after"] <= training["path_before"]
    roots = cfg["data"]["test"]
    result = infer_mri(checkpoint, roots["root_lq"], tmp_path / "test10",
                       root_hq=roots["root_hq"], device_name="cpu", steps=2,
                       case_inventory=inventory, test_fraction=cfg["test"]["fraction"])
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    assert summary["subset"]["selected_cases"] == 2
    assert summary["total_slices"] == 6
    assert summary["inference_variant"] == "chart_trivialized_momentum_bridge"
