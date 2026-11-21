import json
from multiprocessing import get_context

import pytest
import torch

from test_training import _config
from vemgsbm import mri_infer
from vemgsbm.mri_case_cache import acquire_lock


def _setup(tmp_path):
    cfg = _config(tmp_path)
    model = tmp_path/"checkpoint.pt"
    torch.save(dict(config=cfg,stage="reconstruction"),model)
    roots = cfg["data"]["val"]
    args = dict(checkpoint_path=model, root_lq=roots["root_lq"],root_hq=roots["root_hq"],
                output_root=tmp_path/"val", device_name="cpu", split="val", predictor="bilinear",
                resume_cases=True, save_volume_limit=1)
    return cfg,args


def test_resume_after_interrupted_case(tmp_path,monkeypatch):
    _, args = _setup(tmp_path)
    original = mri_infer._write_case
    calls = []
    def interrupted(case, directory, *other, **kwargs):
        calls.append(case.id)
        if case.id == "val_01":
            partial = directory/case.id
            partial.mkdir()
            (partial/"mean.npy").write_bytes(b"incomplete")
            raise InterruptedError("preemption")
        original(case,directory,*other,**kwargs)
    monkeypatch.setattr(mri_infer,"_write_case",interrupted)
    with pytest.raises(InterruptedError):
        mri_infer.infer_mri(**args)
    cache = tmp_path/".val.resume"
    assert not args["output_root"].exists()
    assert list(p.name for p in (cache/"cases").iterdir()) == ["val_00"]
    assert list((cache/"attempts").rglob("mean.npy"))
    with pytest.raises(ValueError,match="resume identity"):
        mri_infer.infer_mri(**args,seed=7)
    def counted(case,*rest,**kwargs):
        calls.append(case.id)
        return original(case,*rest,**kwargs)
    monkeypatch.setattr(mri_infer,"_write_case",counted)
    mri_infer.infer_mri(**args)
    assert calls == ["val_00","val_01","val_01"]
    summary = json.loads((args["output_root"]/"summary.json").read_text())
    assert summary["total_slices"] == 6
    assert not list((cache/"attempts").iterdir())
    mri_infer.infer_mri(**args)
    assert len(calls) == 3
    # A metadata change inside a published case must be caught on idempotent restart.
    (args["output_root"]/"val_00"/"metrics.json").write_text("{}")
    with pytest.raises(ValueError,match="checksum"):
        mri_infer.infer_mri(**args)


def test_cached_contract_detects_changed_source_with_preserved_inventory(tmp_path):
    from PIL import Image
    import numpy as np
    from test_training import _save_config
    from vemgsbm.mri_inventory import prepare_inventory

    cfg, args = _setup(tmp_path)
    config_path = tmp_path/"cfg.yaml"
    _save_config(cfg,config_path)
    inventory = tmp_path/"inventory.json.gz"
    prepare_inventory(config_path, inventory)
    args["case_inventory"] = inventory
    mri_infer.infer_mri(**args)
    hr = next((tmp_path/"data/val/HQ_png/val_00").glob("*.png"))
    Image.fromarray(np.full((8,8),42,dtype=np.uint8)).save(hr)
    with pytest.raises(ValueError,match="identity differs"):
        mri_infer.infer_mri(**args)


def test_source_change_during_case_prevents_commit(tmp_path,monkeypatch):
    _, args = _setup(tmp_path)
    original = mri_infer._write_case
    def write_and_mutate(case, *rest, **kwargs):
        original(case, *rest, **kwargs)
        with case.hq[0].open("ab") as stream:
            stream.write(b"changed")
    monkeypatch.setattr(mri_infer, "_write_case", write_and_mutate)
    with pytest.raises(ValueError, match="source images changed"):
        mri_infer.infer_mri(**args)
    assert not list((tmp_path/".val.resume"/"cases").iterdir())


def _hold_lock(path, ready):
    from time import sleep
    with acquire_lock(path):
        ready.set()
        sleep(120)


def test_killed_worker_releases_lock(tmp_path):
    ctx = get_context("spawn")
    ready = ctx.Event()
    path = tmp_path/"stable-lock"
    worker = ctx.Process(target=_hold_lock,args=(path,ready))
    worker.start()
    try:
        assert ready.wait(30)
        with pytest.raises(FileExistsError):
            acquire_lock(path)
    finally:
        worker.kill()
        worker.join(10)
    with acquire_lock(path):
        assert path.is_file()


def _distributed_run(rank, uri, args, interrupt):
    from datetime import timedelta
    import torch.distributed as dist
    dist.init_process_group("gloo",init_method=uri,rank=rank,world_size=2,timeout=timedelta(seconds=60))
    original = mri_infer._write_case
    try:
        def injected(case,*other,**kwargs):
            if rank == 1 and interrupt:
                raise InterruptedError("simulated rank preemption")
            if rank == 0 and not interrupt:
                raise AssertionError("Already committed rank-zero case was evaluated again")
            original(case,*other,**kwargs)
        mri_infer._write_case = injected
        if interrupt:
            with pytest.raises(RuntimeError,match="preemption"):
                mri_infer.infer_mri(**args)
        else:
            mri_infer.infer_mri(**args)
    finally:
        mri_infer._write_case = original
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(),reason="Gloo unavailable")
def test_two_rank_resume(tmp_path):
    _,args = _setup(tmp_path)
    for attempt, interrupt in enumerate((True,False)):
        uri = (tmp_path/f"store_{attempt}").resolve().as_uri()
        torch.multiprocessing.spawn(_distributed_run,args=(uri,args,interrupt),nprocs=2)
    assert sorted(p.name for p in args["output_root"].iterdir() if p.is_dir()) == ["val_00","val_01"]
