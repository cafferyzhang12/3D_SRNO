import json
from pathlib import Path

import pytest
import torch
import yaml

from vemgsbm import mri_x1_pipeline as pipeline
from vemgsbm.runtime import read_config


ROOT = Path(__file__).resolve().parents[1]


def test_late_stability_accepts_early_learning_but_rejects_recent_collapse():
    rows = [{"val_pooled_psnr_db": value} for value in [23.,27.,30.,30.1,30.2,30.3,30.4]]
    stable,drop = pipeline.late_stability(rows)
    assert stable and drop == 0
    rows[-1]["val_pooled_psnr_db"] = 29.8
    stable,drop = pipeline.late_stability(rows)
    assert not stable and drop == pytest.approx(.5)
    assert not pipeline.late_stability(rows[:3])[0]


@pytest.mark.parametrize("values", [[1.,1.1],[1.,float("nan")],[1.,-1.]])
def test_fixed_batch_rejects_deployment_failure_even_when_total_loss_falls(values):
    with pytest.raises(ValueError,match="rollout_pixel_mse"):
        pipeline.check_fixed_batch({"loss":[2.,1.],"rollout_pixel_mse":values,
                                    "rollout_steps":4,"inference_steps":4})


def test_fixed_batch_requires_same_budget_as_inference():
    record = {"loss":[2.,1.],"rollout_pixel_mse":[.01,.008],"rollout_steps":4,"inference_steps":4}
    pipeline.check_fixed_batch(record)
    record["inference_steps"] = 8
    with pytest.raises(ValueError,match="budgets"):
        pipeline.check_fixed_batch(record)


def test_sr_template_has_aligned_deterministic_and_bounded_mls_objective():
    cfg = read_config(ROOT/"configs/mri_haar_sr_mls_x4.yaml")
    assert cfg["train"]["objective"] == "sr_reconstruction"
    assert cfg["sigma_0"] == cfg["sigma_bridge"] == cfg["sampling"]["inference_sigma"] == 0
    assert cfg["infer_steps"] == cfg["sampling"]["steps"] == cfg["train"]["rollout_steps"] == 4
    assert cfg["use_band_whitening"] and cfg["mls_reg_weight"] == .05
    assert cfg["mls_regularizer"]["geometry_mode"] == "paired_curved"
    assert cfg["mls_regularizer"]["max_fit_sites_per_fold"] == 65536
    assert cfg["train"]["batch_size"] == 64 and cfg["train"]["epochs"] == 40


def smoke_record():
    return dict(epoch=1,world_size=4,global_batch_size=256,val_case_count=10,rollout_items=8,
                loss=.9,grad_norm=.5,start_pixel_mse=.002,rollout_pixel_mse=.001,
                start_hf_mse=.001,start_edge_mse=.0005,start_mls_reconstruction_mse=.001,
                rollout_mls_reconstruction_mse=.001,val_pooled_psnr_db=30.,val_mean_slice_ssim=.88)


def write_smoke(tmp_path,record):
    cfg = yaml.safe_load((ROOT/"configs/mri_haar_sr_mls_x4.yaml").read_text())
    cfg["run_dir"] = str(tmp_path/"run")
    path = tmp_path/"smoke.yaml"
    path.write_text(yaml.safe_dump(cfg))
    folder = tmp_path/"run/bridge"
    folder.mkdir(parents=True)
    (folder/"metrics.jsonl").write_text(json.dumps(record)+"\n")
    return path,folder


def test_sr_smoke_requires_full_four_rank_epoch_without_metric_threshold(tmp_path):
    path,folder = write_smoke(tmp_path,smoke_record())
    pipeline.check_sr_smoke(path)
    assert json.loads((folder/"smoke_gate.json").read_text())["passed"]


@pytest.mark.parametrize("key,value",[("world_size",1),("val_case_count",2),("rollout_items",1),
                                     ("grad_norm",0),("start_mls_reconstruction_mse",float("nan"))])
def test_sr_smoke_rejects_invalid_execution(tmp_path,key,value):
    row = smoke_record()
    row[key] = value
    path,folder = write_smoke(tmp_path,row)
    with pytest.raises(ValueError):
        pipeline.check_sr_smoke(path)
    assert not (folder/"smoke_gate.json").exists()


def test_sr_report_does_not_reinstate_old_321_test_threshold(tmp_path):
    path,folder = write_smoke(tmp_path,smoke_record())
    rows = [dict(epoch=i,val_pooled_psnr_db=29+i/100) for i in range(1,41)]
    (folder/"metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    val = tmp_path/"val"
    val.mkdir()
    (val/"summary.json").write_text(json.dumps(dict(all_psnr_db=29.4,mean_slice_ssim=.87)))
    pipeline.report_sr(path,val)
    result = json.loads((folder.parent/"sr_report.json").read_text())
    assert result["completed_40_epochs"] and result["final_test_authorized"]
    assert result["stability_gate"] and not result["metric_threshold_required"]


def test_sr_prepare_reuses_only_identity_matching_full_train_statistics(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from vemgsbm import haar_stats, mri_inventory

    cfg = yaml.safe_load((ROOT/"configs/mri_haar_sr_mls_x4.yaml").read_text())
    cfg["data"]["train"]["root_lq"] = "/resolved/legacy/TRAIN/LQ"
    identity = {"full_train":True}
    # The real dataset has variable case lengths; use one accounting record for the sum.
    train = [SimpleNamespace(lq=[]) for _ in range(3918)]+[SimpleNamespace(lq=range(292657))]
    val = [SimpleNamespace(lq=[]) for _ in range(9)]+[SimpleNamespace(lq=range(595))]
    monkeypatch.setattr(mri_inventory,"load_inventory",lambda *a: ({"train":train,"val":val},None))
    monkeypatch.setattr(haar_stats,"statistics_identity",lambda cases: identity)
    monkeypatch.setattr(pipeline,"project_path",lambda value: Path(value) if Path(value).is_absolute() else tmp_path/value)
    checkpoint = tmp_path/"old.pt"
    torch.save({"config":cfg},checkpoint)
    cache = tmp_path/"stats.pt"
    torch.save({"identity":identity,"mean":torch.zeros(16),"std":torch.ones(16)},cache)
    pipeline.prepare_sr(ROOT/"configs/mri_haar_sr_mls_x4.yaml",checkpoint,"inventory","sr_run",str(cache))
    output = yaml.safe_load((tmp_path/"runs/mri/sr_run/experiment.yaml").read_text())
    assert output["band_stats_cache"] == str(cache)
    assert output["data"] == cfg["data"]
    assert output["mls_reg_weight"] == .05
    smoke = yaml.safe_load((tmp_path/"runs/mri/sr_run/smoke.yaml").read_text())
    assert smoke["train"]["epochs"] == 1 and smoke["train"]["batch_size"] == 64
    torch.save({"identity":{"different":True}},cache)
    with pytest.raises(ValueError,match="TRAIN identity"):
        pipeline.prepare_sr(ROOT/"configs/mri_haar_sr_mls_x4.yaml",checkpoint,"inventory","other_run",str(cache))
