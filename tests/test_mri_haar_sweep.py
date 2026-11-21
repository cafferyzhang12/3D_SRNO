"""Cross-stage controls and whole-validation selection for the study launcher."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import yaml

from test_training import _config
from vemgsbm.mri_haar_sweep import Arm, plan, select
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
    assert len(repeat) == 21
    assert all(set(replace(a, seed=i) for i in (42, 43, 44)).issubset(repeat) for a in repeat)
