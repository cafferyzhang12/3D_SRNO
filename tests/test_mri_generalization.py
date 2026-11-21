import gzip
import json
import csv

import numpy as np
from PIL import Image
import pytest
import torch

from vemgsbm.mri_data import index_cases
from vemgsbm.mri_generalization import confirm, prepare
from vemgsbm.mri_inventory import _case_record
from vemgsbm.runtime import fingerprint


def archive_fixture(path, count):
    archive, small = path/"archive", path/"small"
    roots = {}
    for sub, name, side in (("LQ/X4", "LQ", 3), ("HQ", "HQ", 12)):
        for cid in ["dev", "trained", "testcase", *(f"extra{i:03}" for i in range(count))]:
            folder = archive/sub/cid
            folder.mkdir(parents=True)
            p = folder/"00000000.jpg"
            Image.fromarray(np.arange(side*side,dtype=np.uint8).reshape(side,side)).save(p)
            if cid == "dev":
                folder = small/name/cid
                folder.mkdir(parents=True)
                with Image.open(p) as image:
                    image.save(folder/"00000000.png")
        roots["root_" + name.lower()] = str(small/name)
    c = index_cases(roots["root_lq"],roots["root_hq"],scale=4)[0]
    data = dict(kind="paired_mri_png",degradation="paired_xy",scale=4,patch_size=[3,8,8],val=roots)
    checkpoint = path/"model.pt"
    torch.save(dict(config={"mode":"gsbm","data":data,"sampling":{"sigma":.1,"steps":20},
                            "train":{"threads":1}},stage="reconstruction"),checkpoint)
    inv = path/"inventory.json.gz"
    with gzip.open(inv,"wt") as stream:
        json.dump({"version":1,"data":data,"splits":{"val":[_case_record(c)],
                   "train":[dict(id="trained",fingerprint="x")],"test":[dict(id="testcase",fingerprint="y")]}},stream)
    return archive, checkpoint, inv


def test_refuse_non_equivalent_archive(tmp_path):
    archive, checkpoint, inv = archive_fixture(tmp_path,1)
    Image.fromarray(np.zeros((12,12),dtype=np.uint8)).save(tmp_path/"small/HQ/dev/00000000.png")
    with pytest.raises(ValueError,match="does not exactly reproduce"):
        prepare(checkpoint,inv,archive,tmp_path/"cohort",count=1)
    assert not (tmp_path/"cohort").exists()


def test_frozen_independent_confirmation(tmp_path,monkeypatch):
    from vemgsbm import mri_infer
    archive, checkpoint, inv = archive_fixture(tmp_path,84)
    cohort = tmp_path/"cohort"
    locked = prepare(checkpoint,inv,archive,cohort)
    assert locked["development_pairs_pixel_verified"] == 1
    assert all(r["case_id"].startswith("extra") for r in locked["cases"])
    manifest = tmp_path/"manifest.json"
    manifest.write_text("{}")
    record = dict(inventory_sha256=fingerprint(inv),manifest_sha256=fingerprint(manifest),
                  selected=dict(checkpoint_sha256=fingerprint(checkpoint),inference={
                      "steps":0,"inference_variant":"endpoint_safe","predictor":"bilinear","seed":42,"samples":1}))
    (tmp_path/"freeze.json").write_text(json.dumps(record))
    output = tmp_path/"confirmed"
    calls = []
    original = mri_infer._write_case
    def interrupted(case, *args, **kwargs):
        calls.append(case.id)
        if len(calls) == 2:
            raise InterruptedError("archive preemption")
        return original(case, *args, **kwargs)
    monkeypatch.setattr(mri_infer, "_write_case", interrupted)
    with pytest.raises(InterruptedError, match="archive preemption"):
        confirm(tmp_path,cohort,output,"cpu")
    assert not output.exists()
    assert (tmp_path/".confirmed.resume"/"cases"/calls[0]/"case_commit.json").is_file()
    for manifest_path in (cohort/"cohort.json",tmp_path/"freeze.json"):
        original_bytes = manifest_path.read_bytes()
        manifest_path.write_bytes(original_bytes + b"\n")
        with pytest.raises(ValueError,match="resume identity changed"):
            confirm(tmp_path,cohort,output,"cpu")
        manifest_path.write_bytes(original_bytes)
    assert len(calls)==2
    confirm(tmp_path,cohort,output,"cpu")
    assert len(calls)==85 and calls.count(calls[0]) == 1
    summary = json.loads((output/"summary.json").read_text())
    assert summary["split"] == "independent_archive" and len(summary["cases"]) == 84
    identity = json.loads((output/"evaluation_identity.json").read_text())
    receipt = json.loads((output/"publication.json").read_text())
    csvrow = next(csv.DictReader((output/"summary.csv").read_text().splitlines()))
    for key, expected in (("cohort_sha256",fingerprint(cohort/"cohort.json")),
                          ("freeze_sha256",fingerprint(tmp_path/"freeze.json")),("split","independent_archive")):
        assert summary[key] == identity[key] == csvrow[key] == expected
    assert all(receipt[name] == fingerprint(output/name) for name in receipt)
    confirm(tmp_path,cohort,output,"cpu")
    assert len(calls)==85
    original_plane = cohort/"HQ"/calls[0]/"00000000.png"
    Image.fromarray(np.full((12,12), 13, dtype=np.uint8)).save(original_plane)
    with pytest.raises(ValueError,match="pixel files changed"):
        confirm(tmp_path,cohort,output,"cpu")
