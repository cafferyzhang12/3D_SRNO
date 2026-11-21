"""Configure or smoke-test one spatial MRI experiment within its compute allocation."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from torch.nn.parallel import DistributedDataParallel
import yaml

from .datasets import training_data
from .distributed import initialize_distributed
from .haar_residual import HaarObjective, make_haar_model
from .runtime import atomic_save, load_checkpoint, project_path, read_config, setup
from .train import _prepare_spatial_bank, _whole_case_validation


def create_variant(args):
    cfg = read_config(args.template)
    cfg["train"]["seed"] = args.seed
    if args.variant == "direct":
        if cfg["mode"] != "haar_direct":
            raise ValueError("Direct regression requires the direct config")
    elif cfg["mode"] != "haar_bridge":
        raise ValueError("Bridge variants require the Haar bridge config")
    if args.variant == "drift_only":
        cfg["train"]["endpoint_weight"] = 0.
        cfg["sampling"]["readout"] = "integral"
    if args.variant in {"mean", "pca", "mls"}:
        if not args.direct_checkpoint:
            raise ValueError("Select the frozen A checkpoint using whole-case validation")
        cfg["spatial_atlas"] = {
            "method": args.variant, "dim": args.dim, "reference": args.reference, "query": args.query,
            "path_geometry": args.path_geometry, "shuffle_routing": args.shuffle_routing,
            "direct_checkpoint": str(project_path(args.direct_checkpoint)),
            "seed": 42, "neighbors": 64, "centers": 128, "candidates": 3, "query_chunk": args.query_chunk,
            "ridge": .0001, "condition_limit": 1.e6, "folds": 5, "sites_per_case": 256, "patches_per_case": 16,
        }
        if args.bank:
            cfg["spatial_atlas"]["bank"] = str(project_path(args.bank))
        if args.path_geometry:
            cfg["path"].update(iterations=5, state_weight=.01, metric_weight=.01, backtrack_steps=3, lr=.0002)
    elif any([args.path_geometry, args.shuffle_routing, args.reference != "identity", args.query != "endpoint"]):
        raise ValueError("Requested a geometry control without a spatial atlas")
    rendered = yaml.safe_dump(cfg, sort_keys=False)
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != rendered:
            raise ValueError("Existing experiment configuration differs")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    return cfg


def overfit(config_path, inventory, directory, steps, device_name):
    """Use identical patch, t, and Gaussian noise for all optimizer updates."""
    cfg = read_config(config_path)
    ctx = initialize_distributed(device_name)
    try:
        device = setup(cfg["train"]["seed"], str(ctx.device), cfg["train"].get("threads", 2))
        directory = Path(directory)
        ctx.main_call(lambda: directory.mkdir(parents=True, exist_ok=True))
        train, val, _, ids = training_data(cfg, inventory)
        model = make_haar_model(cfg).to(device)
        atlas = None
        if cfg.get("spatial_atlas"):
            from .spatial_atlas import load_spatial_atlas
            bank_path = ctx.main_call(lambda: _prepare_spatial_bank(cfg, train, ids, device, directory),
                                      broadcast_result=True)
            payload = load_checkpoint(bank_path)["payload"]
            atlas = load_spatial_atlas(payload, cfg["spatial_atlas"]).requires_grad_(False).to(device).eval()
        raw = HaarObjective(model, cfg, atlas)
        objective = (DistributedDataParallel(raw, device_ids=[ctx.local_rank] if device.type == "cuda" else None,
                                             broadcast_buffers=False) if ctx.is_distributed else raw)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"])
        # Prefer a nontrivial training patch (paired residual energy above machine precision).
        for idx in range(ctx.rank, len(train), ctx.world_size):
            row = train[idx]
            if float((row["hr"] - row["base"]).square().mean()) > 1.e-8:
                break
        else:
            raise ValueError("No nontrivial paired training patch found on this rank")
        lr, base, hr = (row[key][None].to(device) for key in ("lr", "base", "hr"))
        benchmark = {}
        if atlas is not None:
            bind_ms, feature_ms = [], []
            for repeat in range(4):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                begin = time.perf_counter()
                bound = atlas.bind(lr, [row["id"]])
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                middle = time.perf_counter()
                bound.features(model.codec.encode(hr - base))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                end = time.perf_counter()
                if repeat:
                    bind_ms.append(1000 * (middle - begin))
                    feature_ms.append(1000 * (end - middle))
            benchmark = {"native_lr_shape": list(lr.shape), "bind_ms": statistics.median(bind_ms),
                         "features_ms": statistics.median(feature_ms),
                         "query_chunk": cfg["spatial_atlas"]["query_chunk"]}
        losses = []
        for iteration in range(steps + 1):
            torch.manual_seed(20261110 + ctx.rank)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(lr, base, hr, [row["id"]])["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite overfit loss")
            losses.append(ctx.sum([float(loss.detach())])[0] / ctx.world_size)
            if iteration < steps:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
        if losses[-1] >= losses[0]:
            raise RuntimeError(f"Fixed-minibatch loss did not decrease: {losses[0]} to {losses[-1]}")
        model.eval()
        scores = _whole_case_validation(model, val.all_cases[:1], cfg, ctx, directory, 0, atlas)
        record = {"fixed_minibatch_losses": losses, "world_size": ctx.world_size,
                  "validation_case": val.all_cases[0].id, "spatial_timing": benchmark, **scores}
        ctx.main_call(lambda: atomic_save(record, directory / "smoke.json", json_file=True))
        if ctx.is_main:
            print(json.dumps(record), flush=True)
    finally:
        ctx.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    variant = sub.add_parser("config")
    variant.add_argument("--template", required=True)
    variant.add_argument("--output", type=Path, required=True)
    variant.add_argument("--variant", choices=["direct", "bridge", "drift_only", "mean", "pca", "mls"], required=True)
    variant.add_argument("--seed", type=int, default=42)
    variant.add_argument("--dim", choices=[2, 4, 6, 8], type=int, default=2)
    variant.add_argument("--query-chunk", type=int, default=2048)
    variant.add_argument("--reference", choices=["identity", "geometry", "shuffled"], default="identity")
    variant.add_argument("--query", choices=["endpoint", "state"], default="endpoint")
    variant.add_argument("--shuffle-routing", action="store_true")
    variant.add_argument("--path-geometry", action="store_true")
    variant.add_argument("--direct-checkpoint")
    variant.add_argument("--bank")
    smoke = sub.add_parser("overfit")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--inventory", required=True)
    smoke.add_argument("--directory", required=True)
    smoke.add_argument("--steps", type=int, default=24)
    smoke.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.action == "config":
        create_variant(args)
    else:
        overfit(args.config, args.inventory, args.directory, args.steps, args.device)


if __name__ == "__main__":
    main()
