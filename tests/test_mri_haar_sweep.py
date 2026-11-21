"""Cross-stage controls and whole-validation selection for the study launcher."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import yaml
import pytest

from test_training import _config
from vemgsbm.mri_haar_sweep import Arm, plan, replicate_roles, select
from vemgsbm.mri_infer import infer_mri
from vemgsbm.train import run_training


def test_sweep_selects_complete_real_entrypoint_val(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _config(tmp_path)
    cfg.pop("representation")
    cfg["model"] = {"widths": [4, 8, 8]}
    cfg["train"].update(selection_metric="pooled_psnr", bridge_epochs=1)
    cfg["path"].update(iterations=0)
    study = "test"
    for arm in plan("backbone", {}):
        rid = arm.run_id(study)
        run = tmp_path / "runs" / "mri" / rid
        cfg["run_dir"] = str(run)
        cfg["mode"] = "haar_direct" if arm.variant == "direct" else "haar_bridge"
        cfg["train"]["endpoint_weight"] = float(arm.variant != "drift_only")
        cfg["sampling"]["readout"] = "integral" if arm.variant == "drift_only" else "endpoint"
        if not run.is_dir():
            path = tmp_path / f"{rid}.yaml"
            path.write_text(yaml.safe_dump(cfg))
            run_training(path, device_name="cpu")
        dest = tmp_path / "outputs" / "mri" / (rid + (
            "_integral_val" if arm.variant == "bridge" and arm.readout == "integral" else "_val"))
        infer_mri(run / "bridge" / "best.pt", cfg["data"]["val"]["root_lq"], dest,
                  root_hq=cfg["data"]["val"]["root_hq"], device_name="cpu", split="val",
                  readout=arm.readout if cfg["mode"] == "haar_bridge" else None)
    previous = {}
    select("backbone", tmp_path, study, previous)
    report = json.loads((tmp_path / "backbone.json").read_text())
    assert len(report["rows"]) == 4
    bridges = [r for r in report["rows"] if r["arm"]["variant"] != "direct"]
    assert report["winners"]["bridge"]["pooled_psnr"] == max(r["pooled_psnr"] for r in bridges)
    assert len((tmp_path / "dimensions.tsv").read_text().splitlines()) == 9
    direct = report["winners"]["direct"]
    assert direct["validation_replay_delta"]["psnr_db"] == pytest.approx(0., abs=1.e-5)
    summary_path = Path(direct["result"]) / "summary.json"
    changed = json.loads(summary_path.read_text())
    changed["seed"] += 1
    summary_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Invalid VAL record"):
        select("backbone", tmp_path, study, {})


def test_frozen_mls_and_three_seed_controls():
    curve = Arm("mls", dim=4, path=1)
    previous = {
        "backbone": {"winners": {k: {"arm": asdict(Arm(k))} for k in ["direct", "bridge"]}},
        "dimensions": {"winners": {k: {"arm": asdict(Arm(k, dim=4))} for k in ["mean", "pca", "mls"]}},
        "controls": {"winners": {"physical_mls": {"arm": asdict(curve)}}},
        "reference": {"winners": {k: {"arm": asdict(replace(curve, reference=k))}
                                  for k in ["geometry", "shuffled"]}},
    }
    arms = plan("reference", previous)
    assert arms == [replace(curve, reference=r) for r in ["identity", "geometry", "shuffled"]]
    repeat = plan("replicate", previous)
    assert len(repeat) == 24
    assert all(set(replace(a, seed=i) for i in (42, 43, 44)).issubset(repeat) for a in repeat)
    assert replicate_roles(previous)[Arm("pca", dim=4)] == ["finalist", "matched_pca"]
    assert replicate_roles(previous)[Arm("mls", dim=4)] == ["matched_mls"]
    previous["controls"]["winners"]["physical_mls"] = {"arm": asdict(Arm("mls", dim=4))}
    assert len(plan("replicate", previous)) == 21
    assert replicate_roles(previous)[Arm("mls", dim=4)] == ["finalist", "matched_mls"]


def test_matched_mls_pca_three_seeds_do_not_enlarge_finalist_ranking(tmp_path, monkeypatch):
    previous = {stage: {"winners": {}, "rows": []} for stage in ["backbone", "dimensions", "controls", "reference"]}
    for stage, key, arm in [
            ("backbone", "direct", Arm("direct")), ("backbone", "bridge", Arm("bridge")),
            ("dimensions", "mean", Arm("mean")), ("dimensions", "pca", Arm("pca", dim=2)),
            ("dimensions", "mls", Arm("mls", dim=8)), ("controls", "physical_mls", Arm("mls", dim=8, query="state", path=1)),
            ("reference", "geometry", Arm("mls", dim=8, query="state", path=1, reference="geometry")),
            ("reference", "shuffled", Arm("mls", dim=8, query="state", path=1, reference="shuffled"))]:
        previous[stage]["winners"][key] = {"arm": asdict(arm)}

    def evaluate(arm, study):
        return {"arm": asdict(arm), "val_identity": "same", "pixels": 100, "run_id": arm.run_id(study),
                "pooled_psnr": 45. if arm == replace(Arm("pca", dim=8), seed=arm.seed) else 40.,
                "slice_ssim": .95}
    monkeypatch.setattr("vemgsbm.mri_haar_sweep.evaluate", evaluate)
    select("replicate", tmp_path, "test", previous)
    report = previous["replicate"]
    assert len(report["rows"]) == 27
    assert len(report["ranking"]) == 7
    assert max(r["mean_pooled_psnr"] for r in report["ranking"]) == 40.
    assert set(report["matched_geometry"]) == {"mls", "pca"}
    for entry in report["matched_models"]:
        cfg = entry["config"]
        assert entry["roles"] == ["matched_" + cfg["variant"]]
        assert cfg["dim"] == 8 and cfg["reference"] == "identity" and cfg["query"] == "endpoint"
        assert cfg["path"] == cfg["shuffle"] == 0
        assert sorted(row["arm"]["seed"] for row in entry["runs"]) == [42, 43, 44]
