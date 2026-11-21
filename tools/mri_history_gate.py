"""Check the historical checkpoint, cohort and fixed real TRAIN batch under torchrun."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from vemgsbm.datasets import training_data
from vemgsbm.distributed import initialize_distributed
from vemgsbm.paths import linear_path, sample_path
from vemgsbm.runtime import atomic_save, build_models, fingerprint, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "inventory", "historical-summary", "manifest", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    ctx = initialize_distributed("cuda")
    try:
        saved = load_checkpoint(args.checkpoint)
        if saved["epoch"] != 38 or saved["metric"]["kind"] != "mls":
            raise ValueError("Expected the epoch-38 reconstruction with its MLS atlas")
        summary = json.loads(args.historical_summary.read_text())
        manifest = json.loads(args.manifest.read_text())
        pixels = sum(row["metrics"]["pixel_count"] for row in summary["cases"])
        if (summary["subset"] != manifest or len(summary["cases"]) != 84
                or summary["total_slices"] != 14991 or pixels != 3929800704):
            raise ValueError("Historical cohort mismatch")
        train, val, digest, _ = training_data(saved["config"], args.inventory)
        if saved["manifest_sha256"] != digest:
            raise ValueError("Historical checkpoint and inventory identity mismatch")
        batch = train[ctx.rank]
        base, hr, mask = [batch[k][None].to(ctx.device) for k in ("base", "hr", "mask")]
        model, _ = build_models(saved, ctx.device)
        model.train()
        if ctx.is_distributed:
            model = DistributedDataParallel(model, device_ids=[ctx.local_rank])
        torch.manual_seed(42 + ctx.rank)
        state, target = sample_path(linear_path(base, hr, mask), 0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.e-4)
        losses = []
        for i in range(33):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(state, base, mask, 0.5, "forward") - target).square().mean()
            if i in (0, 32):
                losses.append(ctx.sum([loss.detach().item()])[0] / ctx.world_size)
            if i < 32:
                loss.backward()
                optimizer.step()
        if not losses[1] < losses[0]:
            raise RuntimeError(f"Fixed real TRAIN batch loss did not improve: {losses}")
        if ctx.is_main:
            record = dict(epoch=saved["epoch"], atlas=saved["metric"]["kind"], pixels=pixels,
                          checkpoint_sha256=fingerprint(args.checkpoint), world_size=ctx.world_size,
                          train_cases=len(train.cases), val_cases=len(val.cases),
                          train_loss_initial=losses[0], train_loss_final=losses[1],
                          all_psnr_db=summary["all_psnr_db"], mean_slice_ssim=summary["mean_slice_ssim"])
            atomic_save(record, args.output, json_file=True)
            print(json.dumps(record), flush=True)
    finally:
        ctx.cleanup()


if __name__ == "__main__":
    main()
