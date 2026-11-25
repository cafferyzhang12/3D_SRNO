"""Real PNG -> TRAIN whitening/atlas -> checkpoint -> Hann VAL four-metric smoke.

All images are generated locally and explicitly labeled synthetic. This invokes
the production training/evaluation entry point, without mocks or TEST access.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))


def make_cases(root):
    """Six distinct cases; five TRAIN cases support true 5-fold exclusion."""
    from mri_ceiling_audit import synthetic_pair
    records = {}
    for split, count in (("train", 5), ("val", 1)):
        records[split] = dict(root_lq=str(root/split/"LQ"), root_hq=str(root/split/"HQ"))
        for index in range(count):
            lr, hr = synthetic_pair(32, depth=3, seed=1000*int(split == "val")+index+40)
            # Vary anatomy as well as noise so source fingerprints stay distinct.
            shift = index-2 if split == "train" else 1
            hr = hr.roll(shift, dims=-1)
            lr = F.avg_pool2d(hr.reshape(3, 1, 32, 32), 4, 4).reshape(1, 1, 3, 8, 8)
            for level, value in (("LQ", lr), ("HQ", hr)):
                directory = root/split/level/f"{split}_{index:02}"
                directory.mkdir(parents=True)
                for z in range(3):
                    raw = (value[0, 0, z].numpy().clip(0, 1)*255).round().astype(np.uint8)
                    Image.fromarray(raw).save(directory/f"slice_{z:03}.png")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", type=int, choices=[1,2,3,4], default=[1,2,3,4])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weights-dir", default="cache/perceptual")
    parser.add_argument("--output", default="reports/mri/v2_e2e.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    project = Path(__file__).resolve().parents[1]
    (project/"outputs").mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="v2-e2e-", dir=project/"outputs"))
    roots = make_cases(work/"data")
    from vemgsbm.mri_v2_train import run
    from vemgsbm.runtime import load_checkpoint
    result = dict(source="generated synthetic PNG sequences", test_used=False, device=args.device,
                  working_directory=str(work), train_cases=5, val_cases=1,
                  hr_shape=[3,32,32], lr_shape=[3,8,8], variants={})
    for variant in args.variants:
        template = next((project/"configs").glob(f"mri_v2_exp{variant}_*.yaml"))
        cfg = yaml.safe_load(template.read_text(encoding="utf-8"))
        cfg["data"].update(train=roots["train"], val=roots["val"], patch_size=[3,32,32], scan_workers=1)
        cfg["data"].pop("scan_cache", None); cfg["data"].pop("legacy_scan_cache", None)
        cfg["model"].update(widths=[4,8,12], blocks=[1,1,1], heads=1, window_size=2,
                            refine_width=4, gradient_checkpointing=True)
        cfg["train"].update(epochs=1, batch_size=1, accumulation_steps=1, samples_per_epoch=2,
            val_samples_per_epoch=1, workers=0, threads=1, warmup_epochs=0, schedule_epochs=1,
            train_samples=1, val_samples=1, report_every_updates=1)
        cfg["sampling"].update(steps=4, tile_batch=1)
        cfg["perceptual"].update(weights_dir=str((project/args.weights_dir).resolve()), batch_size=3)
        if cfg.get("jetatlas"):
            cfg["jetatlas"].update(folds=5,dim=2,pca_dim=4,centers=4,neighbors=32,
                normal_rank=2,patches_per_case=1,sites_per_case=16,loss_sites_per_patch=8,
                router_hidden=8,top_m=2,query_chunk=64)
        cfg["run_dir"] = str(work/f"exp{variant}")
        path = work/f"exp{variant}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        logfile = work/f"exp{variant}.log"
        started = time.perf_counter()
        print(f"Running real PNG E2E exp{variant}: {path}", flush=True)
        record = dict(config=str(path), log=str(logfile))
        try:
            with logfile.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                trained = run(str(path), device=args.device)
                checkpoint = Path(cfg["run_dir"])/"last.pt"
                loaded = load_checkpoint(checkpoint)
                evaluated = run(str(path), device=args.device, resume=str(checkpoint), evaluate=True)
            record.update(status="passed", training=trained, reloaded_evaluation=evaluated,
                checkpoint=str(checkpoint), checkpoint_format=loaded["format_version"],
                has_atlas=loaded["jetatlas"] is not None,
                hann_complete_case=True, elapsed_seconds=time.perf_counter()-started)
            for key in ("psnr", "ssim", "lpips", "fid", "local_variance_ratio"):
                if not np.isfinite(evaluated[key]):
                    raise FloatingPointError(f"Nonfinite real {key} evaluation")
            if evaluated["slice_count"] != 3 or evaluated["pixel_count"] != 3*32*32:
                raise RuntimeError("Hann validation missed or duplicated slice coverage")
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                          elapsed_seconds=time.perf_counter()-started)
        result["variants"][str(variant)] = record
        output = project/args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps({"variant":variant,**record}), flush=True)
        if record["status"] != "passed":
            print(logfile.read_text(encoding="utf-8")[-6000:], flush=True)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
