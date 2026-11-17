"""Real two-rank Gloo check of representation -> CondSOC/bridge matching.

The rendezvous uses a FileStore, so Windows builds without libuv are supported.
The existing process group stays alive across both train.run_training calls;
this exercises DDP rather than replacing collectives with mocks.
"""
from datetime import timedelta
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
import yaml


pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="CPU distributed Gloo is unavailable",
)


def _config(tmp_path, textured=False):
    data = {
        "kind": "paired_mri_png", "degradation": "paired_xy", "scale": 4,
        "patch_size": [3, 8, 8], "scan_workers": 1,
        "channel_policy": "identical_rgb_to_gray",
    }
    for split in ("train", "val"):
        root = tmp_path / "data" / split
        roots = {"root_hq": root / "HQ_png", "root_lq": root / "LQ" / "X4_png"}
        for index in range(2):
            cid = f"{split}_{index}"
            for folder in roots.values():
                (folder / cid).mkdir(parents=True)
            for z in range(4):
                yy, xx = np.mgrid[:12, :12]
                high = yy * 9 + xx * 4 + z * 13 + index * 29 + 7
                if textured:
                    high = high + (z + 5) * yy**2 + (index + 3) * xx**2 + xx * yy
                high = (high % 240).astype(np.uint8)
                low = Image.fromarray(high).resize((3, 3), Image.Resampling.BILINEAR)
                Image.fromarray(high).save(roots["root_hq"] / cid / f"{z:03}.png")
                low.save(roots["root_lq"] / cid / f"{z:03}.png")
        data[split] = {kind: str(path) for kind, path in roots.items()}
    return {
        "mode": "gsbm", "run_dir": str(tmp_path / "run"), "data": data,
        "model": {"width": 2},
        "representation": {
            "width": 2, "feature_dim": 4, "clusters": 2, "bank_samples": 4,
            "checkpoint": str(tmp_path / "run" / "representation" / "best.pt"),
        },
        "train": {
            "epochs": 1, "representation_epochs": 1, "bridge_epochs": 1,
            "batch_size": 1, "samples_per_epoch": 8, "val_samples_per_epoch": 4,
            "train_samples": 1, "val_samples": 1, "learning_rate": 0.001,
            "workers": 0, "threads": 1, "seed": 42, "matching_time_min": 0.02,
        },
        "path": {
            "knots": 3, "iterations": 1, "quadrature_points": 5,
            "sigma": 0.05, "learning_rate": 0.01, "max_correction": 0.25,
            "control_weight": 1.0, "state_weight": 0.01, "metric_weight": 0.01,
            "optimize_std": True, "std_bound": 0.75,
        },
        "sampling": {"steps": 2, "sigma": 0.05},
    }


def _worker(rank, rendezvous_uri, config_path, output, inventory=None):
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), USE_LIBUV="0")
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous_uri, rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    mode = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["mode"]
    if mode == "momentum":
        from vemgsbm.momentum import train
    else:
        from vemgsbm import train

    writes, scans, updates = [], [], []
    original_save, original_data = train.atomic_save, train.training_data
    original_step = torch.optim.AdamW.step

    def save(value, path, **kwargs):
        writes.append(str(path))
        return original_save(value, path, **kwargs)

    def scan(config, case_inventory=None):
        scans.append(config["run_dir"])
        return original_data(config, case_inventory)

    def step(optimizer, *args, **kwargs):
        before = torch.cat([p.detach().cpu().flatten().clone()
                            for group in optimizer.param_groups for p in group["params"]])
        result = original_step(optimizer, *args, **kwargs)
        after = torch.cat([p.detach().cpu().flatten().clone()
                           for group in optimizer.param_groups for p in group["params"]])
        moments = torch.cat([optimizer.state[p]["exp_avg"].detach().cpu().flatten().clone()
                             for group in optimizer.param_groups for p in group["params"]
                             if "exp_avg" in optimizer.state[p]])
        updates.append({"parameters": after, "moments": moments,
                        "changed": not torch.equal(before, after)})
        return result

    train.atomic_save, train.training_data = save, scan
    torch.optim.AdamW.step = step
    try:
        if mode == "momentum":
            train.run_training(config_path, "cpu", case_inventory=inventory)
        else:
            train.run_training(config_path, "representation", "cpu", case_inventory=inventory)
            train.run_training(config_path, "bridge", "cpu", case_inventory=inventory)
        # Artifact writes here are worker reports, separate from training's
        # patched rank-zero atomic_save contract.
        torch.save({"writes": writes, "scans": scans, "updates": updates},
                   Path(output) / f"worker_{rank}.pt")
    finally:
        train.atomic_save, train.training_data = original_save, original_data
        torch.optim.AdamW.step = original_step
        dist.destroy_process_group()


@pytest.mark.parametrize("prepared", [False, True])
def test_two_rank_representation_bridge_optimizer_sync_and_single_writer(tmp_path, prepared):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config(tmp_path)), encoding="utf-8")
    inventory = None
    if prepared:
        from vemgsbm.mri_inventory import prepare_inventory
        inventory = tmp_path / "cases.json.gz"
        prepare_inventory(config_path, inventory)
    # The FileStore rendezvous file must not exist before init_process_group.
    rendezvous_uri = (tmp_path / "gloo_rendezvous").resolve().as_uri()
    multiprocessing.spawn(_worker, args=(rendezvous_uri, str(config_path), str(tmp_path), inventory),
                          nprocs=2, join=True)
    main = torch.load(tmp_path / "worker_0.pt", weights_only=True)
    other = torch.load(tmp_path / "worker_1.pt", weights_only=True)
    assert main["writes"] and not other["writes"]
    assert len(main["scans"]) == 2 and len(other["scans"]) == (2 if prepared else 0)
    assert len(main["updates"]) == len(other["updates"]) == 8
    assert all(update["changed"] for update in main["updates"])
    for first, second in zip(main["updates"], other["updates"]):
        torch.testing.assert_close(first["parameters"], second["parameters"], rtol=0, atol=0)
        torch.testing.assert_close(first["moments"], second["moments"], rtol=0, atol=0)
    for stage in ("representation", "bridge"):
        folder = tmp_path / "run" / stage
        rows = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["world_size"] == 2
        assert rows[0]["train_items"] == 8 and rows[0]["validation_items"] == 4
        saved = torch.load(folder / "best.pt", weights_only=True)
        assert saved["world_size"] == 2 and saved["epoch"] == 1
        assert all(not key.startswith("module.") for key in saved["model"])
        assert saved["metric"] is not None
    assert rows[0]["path_objective_after"] <= rows[0]["path_objective_before"]
    assert rows[0]["path_optimized_std"] == 1.0
    assert not list((tmp_path / "run").rglob(".training.lock"))


@pytest.mark.parametrize("chart_momentum", [False, True])
def test_two_rank_momentum_trains_and_publishes_png_metrics(tmp_path, chart_momentum):
    from vemgsbm.mri_infer import infer_mri

    cfg = _config(tmp_path, textured=chart_momentum)
    cfg.update(mode="momentum", pca={"bank_patches":4, "tokens_per_patch":4, "centers":2,
                                      "neighbors":6, "tangent_rank":2},
               path={"sigma":.05, "lr":.01, "iterations":2, "quadrature":3, "max_shift":.5,
                     "state_weight":.25, "normal_weight":.25})
    inventory = None
    if chart_momentum:
        from vemgsbm.mri_inventory import prepare_inventory
        cfg.update(chart_momentum={},
                   atlas={"charts": 2, "bank_patches": 8, "neighborhoods": [8, 16, 64],
                          "dim_candidates": [2], "source_routing": {"candidates": 2},
                          "mls": {"degree": 1, "neighbors": 48, "anchor_limit": 4}})
        cfg["path"].update(state_weight=.005, normal_weight=.002, backtracking_steps=2)
        inventory = tmp_path / "cases.json.gz"
    config_path = tmp_path / "momentum.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    if chart_momentum:
        prepare_inventory(config_path, inventory)
    uri = (tmp_path / "momentum_gloo").resolve().as_uri()
    multiprocessing.spawn(_worker, args=(uri, str(config_path), str(tmp_path), inventory),
                          nprocs=2, join=True)
    first, second = [torch.load(tmp_path / f"worker_{i}.pt", weights_only=True) for i in range(2)]
    assert first["writes"] and not second["writes"] and len(first["scans"]) == 1
    assert len(second["scans"]) == int(chart_momentum)
    assert len(first["updates"]) == len(second["updates"]) == 4
    for a, b in zip(first["updates"], second["updates"]):
        assert a["changed"] and b["changed"]
        torch.testing.assert_close(a["parameters"], b["parameters"], rtol=0, atol=0)
        torch.testing.assert_close(a["moments"], b["moments"], rtol=0, atol=0)
    folder = tmp_path / "run" / "momentum"
    row = json.loads((folder / "metrics.jsonl").read_text())
    assert (row["train_items"], row["validation_items"], row["world_size"]) == (8, 4, 2)
    assert row["path_after"] <= row["path_before"]
    checkpoint = folder / "best.pt"
    atlas = torch.load(checkpoint, weights_only=True)["atlas"]
    if chart_momentum:
        assert atlas["diagnostics"]["source_routing_train_pairs"] == 64
    else:
        assert atlas["training_tokens"] == 16
    roots = cfg["data"]["val"]
    output = tmp_path / "reconstructed"
    infer_mri(checkpoint, roots["root_lq"], output, root_hq=roots["root_hq"], device_name="cpu",
              test_fraction=1, max_cases=1, save_volume_limit=0)
    summary = json.loads((output / "summary.json").read_text())
    expected = "chart_trivialized_momentum_bridge" if chart_momentum else "momentum_bridge"
    assert summary["total_slices"] == 4 and summary["inference_variant"] == expected
    assert np.isfinite(summary["all_mse"])
