import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from vemgsbm.mri_inventory import _case_record
from vemgsbm.mri_task_inventory import SPLITS, discover, index_cases, main
from vemgsbm.runtime import PROJECT_ROOT


def test_checkout_groups_eight_methods_and_reports_different_tdm_fraction(tmp_path):
    result = discover(PROJECT_ROOT / "configs", tmp_path)
    group, = result["protocols"]
    original = [c for c in group["configs"] if c["origin"] == "original"]
    assert len(original) == 6
    assert sum(c["origin"] == "research_template" for c in group["configs"]) == 2
    assert [c["config"] for c in original if c["default_test_fraction"] == 0.10] == ["mri_momentum_tdm_x4.yaml"]
    assert len(group["missing_roots"]) == 6
    assert group["status"] == "missing_roots"


def test_recursive_real_png_audit_compares_names_without_publishing_ids(tmp_path, monkeypatch):
    configs, data = tmp_path / "configs", {"scale": 2, "kind": "paired_mri_png", "degradation": "paired_xy"}
    (configs / "nested").mkdir(parents=True)
    patient = "private_patient"
    splits = {}
    for split in SPLITS:
        roots = {}
        for role, res in [("lq", 3), ("hq", 6)]:
            root = tmp_path / split / role
            case_dir = root / (patient + split)
            case_dir.mkdir(parents=True)
            for k in range(2):
                Image.fromarray(np.full((res, res), k, dtype=np.uint8)).save(case_dir / f"frame{k}.png")
            roots["root_" + role] = str(root)
        data[split] = roots
        splits[split] = [_case_record(case) for case in index_cases(**roots, scale=2)]
    archive = tmp_path / "archive.pt"
    torch.save({"config": {"data": data}}, archive)
    inventory = tmp_path / "inventory.json.gz"
    with gzip.open(inventory, "wt", encoding="utf8") as stream:
        json.dump({"version": 1, "data": data, "splits": splits}, stream)
    (configs / "nested/extra.yaml").write_text(yaml.safe_dump({"mode": "gsbm", "data": data}))
    (configs / "alternate.yaml").write_text(yaml.safe_dump({"mode": "gsbm", "data": dict(data, normalization_scale=127)}))
    (configs / "em.yml").write_text(yaml.safe_dump({"data": {"kind": "unpaired_npy"}}))
    (configs / "invalid.yml").write_text(yaml.safe_dump({"data": dict(data, val={})}))
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["inventory", "--configs-dir", str(configs),
                        "--protocol-checkpoint", str(archive), "--case-inventory", str(inventory),
                        "--output-dir", str(output)])
    main()
    result = json.loads((output / "inventory.json").read_text())
    assert result["verified_reference"] and result["protocol_count"] == 2
    assert [i["config"] for i in result["skipped"]] == ["em.yml"]
    assert [i["reason"] for i in result["unsupported"]] == ["missing val.root_lq"]
    assert patient not in json.dumps(result) and patient not in (output / "inventory.csv").read_text()
    live, archived = result["protocols"]
    assert live["status"] == "verified_independent_png_index"
    assert archived["fixed_test_15pct"]["slices"] == 2
    assert archived["splits"]["train"] == live["splits"]["train"] == {"cases": 1, "slices": 2, "hr_voxels": 72}
    assert archived["configs"][0]["origin"] == "additional" and live["id"] != archived["id"]
    for role in ("lq", "hq"):
        case_dir = Path(data["val"]["root_" + role]) / (patient + "val")
        (case_dir / "frame1.png").rename(case_dir / "frame2.png")
    changed = discover(configs, checkpoint=archive, inventory=inventory)
    assert not changed["verified_reference"]
    assert changed["protocols"][1]["status"] == "verification_failed"
