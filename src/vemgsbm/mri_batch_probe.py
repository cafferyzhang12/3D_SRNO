"""Measure training peak memory at native MRI tile size before selecting a shared DDP batch."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from torch.nn.parallel import DistributedDataParallel

from .distributed import initialize_distributed
from .haar_residual import HaarObjective, make_haar_model
from .runtime import atomic_save, read_config, setup


def probe(config, destination):
    ctx = initialize_distributed("cuda")
    try:
        if ctx.world_size != 4:
            raise ValueError("Four GPU probes required")
        cfg = read_config(config)
        device = setup(42, str(ctx.device), 4)
        model = make_haar_model(cfg).to(device)
        objective = HaarObjective(model, cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"])
        def measure(net, size):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            lr = torch.rand(size, 1, 5, 32, 32, device=device)
            base = torch.nn.functional.interpolate(lr, scale_factor=(1,4,4), mode="trilinear", align_corners=False)
            hr = (base + .03 * torch.randn_like(base)).clamp(0, 1)
            begin = time.perf_counter()
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss = net(lr, base, hr)["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            torch.cuda.synchronize()
            _, total = torch.cuda.mem_get_info()
            return dict(batch=size, peak_reserved=torch.cuda.max_memory_reserved(),
                        total=total, seconds_per_step=(time.perf_counter()-begin)/2)
        measurements = []
        for size in (4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 320, 384):
            if measurements:
                last = measurements[-1]
                if last["peak_reserved"] * size / last["batch"] > .74 * last["total"]:
                    break
            try:
                m = measure(objective, size)
            except torch.cuda.OutOfMemoryError:
                break
            if m["peak_reserved"] > .77 * m["total"]:
                break
            measurements.append(m)
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        ctx.check_errors(None if measurements else "No safe batch for this GPU")
        results = [None] * ctx.world_size
        torch.distributed.all_gather_object(results, measurements)
        # Prefer a useful occupancy with headroom, without sacrificing >5% throughput.
        common = {r["batch"] for r in results[0]}
        for rows in results[1:]:
            common.intersection_update(r["batch"] for r in rows)
        rates = {b: min(b / next(r["seconds_per_step"] for r in rows if r["batch"] == b)
                        for rows in results) for b in common}
        efficient = sorted(b for b in rates if rates[b] >= .95 * max(rates.values()))
        roomy = [b for b in efficient if all(next(r["peak_reserved"]/r["total"] for r in rows
                                                  if r["batch"] == b) >= .5 for rows in results)]
        chosen = min(roomy or efficient)
        ddp = DistributedDataParallel(objective, device_ids=[ctx.local_rank], broadcast_buffers=False)
        # Validate synchronization, bucket storage, clipping and optimizer state together.
        tested = measure(ddp, chosen)
        verified = [None] * ctx.world_size
        torch.distributed.all_gather_object(verified, tested)
        if any(row["peak_reserved"] > .85 * row["total"] for row in verified):
            raise RuntimeError("Insufficient memory headroom after DDP verification")
        if ctx.is_main:
            destination.mkdir(exist_ok=True, parents=True)
            atomic_save(dict(per_gpu=chosen, global_batch=4*chosen, per_rank=results, ddp=verified,
                             parameters=sum(p.numel() for p in model.parameters())),
                        destination/"memory.json", json_file=True)
            (destination/"batch.txt").write_text(str(chosen)+"\n", encoding="ascii")
            print(json.dumps(verified), flush=True)
    finally:
        ctx.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    probe(args.config, args.output)
