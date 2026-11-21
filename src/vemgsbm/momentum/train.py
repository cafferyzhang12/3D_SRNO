"""Train scaled acceleration matching with a frozen MRI atlas or legacy PCA."""
import argparse
import copy
import json
import math

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from ..datasets import training_data
from ..distributed import initialize_distributed, ValidationShardSampler
from ..mri_metrics import mean_xy_ssim
from ..runtime import atomic_save, load_checkpoint, project_path, read_config, sampling_noise, setup
from .geometry import assign_chart, fit_atlas
from .guidance import momentum_geometry_options
from .model import AccelerationNetwork
from .phase import optimize_path
from .trivialized import ChartMomentumAtlas, chart_momentum_options, fit_chart_momentum_atlas


@torch.no_grad()
def _atlas_from_training(codec, loader, options, context, seed):
    per_rank = (options["bank_patches"] + context.world_size - 1) // context.world_size
    local, count, error = [], 0, None
    try:
        for batch in loader:
            hr, base = (batch[key].to(context.device)[:per_rank - count] for key in ("hr", "base"))
            blocks = codec.encode(hr - base).flatten(2).transpose(1, 2)
            # Retain flat background while giving non-flat detail more coverage.
            weights = (blocks.square().mean(-1) + 1e-6).sqrt()
            indices = torch.multinomial(weights, min(options["tokens_per_patch"], blocks.shape[1]), replacement=False)
            local.append(blocks.gather(1, indices[..., None].expand(-1, -1, blocks.shape[-1])).flatten(0, 1).cpu())
            count += len(hr)
            if count >= per_rank:
                break
        local = torch.cat(local)
    except Exception as exc:
        error = f"Training atlas features: {exc}"
    context.check_errors(error)
    if context.is_distributed:
        banks = [None] * context.world_size
        torch.distributed.all_gather_object(banks, local)
    else:
        banks = [local]
    return context.main_call(lambda: fit_atlas(torch.cat(banks), options["centers"], options["neighbors"],
                                               options["tangent_rank"], seed), broadcast_result=True)


def run_training(config_path, device_name="auto", resume=None, case_inventory=None):
    cfg = read_config(config_path)
    if cfg["mode"] != "momentum":
        raise ValueError("Expected a momentum config")
    use_mls = bool(cfg.get("atlas"))
    chart_options = chart_momentum_options(cfg)
    guidance = momentum_geometry_options(cfg)
    train_cfg = cfg["train"]
    metric_weight = train_cfg.get("metric_matching_weight", 0.)
    if (isinstance(metric_weight, bool) or not isinstance(metric_weight, (int, float))
            or not math.isfinite(metric_weight) or metric_weight < 0 or (metric_weight and guidance is None)):
        raise ValueError("Momentum metric matching needs finite nonnegative weight and the LR-routed atlas")
    selection = train_cfg.get("selection_metric", "mse")
    if selection not in ("mse", "ssim"):
        raise ValueError("Momentum selection_metric must be mse or ssim")
    sample_sigma = sampling_noise(cfg)
    reference_sigma = cfg["sampling"]["sigma"]
    validation_settings = {"inference": (sample_sigma, guidance)}
    if guidance is not None:
        validation_settings["without_online_guidance"] = (sample_sigma, None)
        if sample_sigma != reference_sigma:
            validation_settings["reference_noise"] = (reference_sigma, guidance)
    report_ssim = guidance is not None or selection == "ssim"
    context = initialize_distributed(device_name)
    run_dir = project_path(cfg["run_dir"]) / "momentum"
    lock = None

    def prepare():
        nonlocal lock
        run_dir.mkdir(parents=True, exist_ok=True)
        if (run_dir / "last.pt").exists() and not resume:
            raise FileExistsError("Existing momentum run requires resume or a new run directory")
        lock = (run_dir / ".training.lock").open("x")
        atomic_save(cfg, run_dir / "config.json", json_file=True)

    try:
        context.assert_same((cfg, resume), "momentum configuration")
        context.main_call(prepare)
        seed = train_cfg["seed"]
        device = setup(seed, str(context.device), train_cfg.get("threads", 4))
        if case_inventory is None:
            data = context.main_call(lambda: training_data(cfg), broadcast_result=True)
        else:
            data, error = None, None
            try:
                data = training_data(cfg, case_inventory)
            except Exception as exc:
                error = f"Prepared MRI inventory: {type(exc).__name__}: {exc}"
            context.check_errors(error)
        train_set, val_set, digest, identity = data
        if case_inventory is not None:
            context.assert_same(digest, "MRI inventory identity")
        if len(train_set) % context.world_size:
            raise ValueError("The global patch budget must be divisible by WORLD_SIZE")
        sampler = DistributedSampler(train_set, context.world_size, context.rank, seed=seed) if context.is_distributed else None
        loader = DataLoader(train_set, batch_size=train_cfg["batch_size"], sampler=sampler,
                            shuffle=sampler is None, num_workers=train_cfg.get("workers", 0),
                            generator=torch.Generator().manual_seed(seed), pin_memory=device.type == "cuda")
        val_loader = DataLoader(val_set, batch_size=train_cfg["batch_size"], num_workers=0,
                               sampler=ValidationShardSampler(val_set, context.rank, context.world_size))
        model = AccelerationNetwork(**cfg["model"], chart_features=chart_options is not None).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=1e-5)
        start, best = 0, float("inf")
        if resume:
            prior = load_checkpoint(project_path(resume))
            previous, current = copy.deepcopy(prior["config"]), copy.deepcopy(cfg)
            previous["train"].pop("epochs")
            current["train"].pop("epochs")
            if (previous != current or prior["data_identity"] != identity or prior["world_size"] != context.world_size):
                raise ValueError("Momentum resume requires matching config, data and world size")
            model.load_state_dict(prior["model"])
            optimizer.load_state_dict(prior["optimizer"])
            atlas = prior["atlas"]
            start, best = prior["epoch"], prior["best"]
        elif not use_mls:
            atlas = _atlas_from_training(model.codec, loader, cfg["pca"], context, seed)
        if chart_options is not None:
            if not resume:
                atlas = fit_chart_momentum_atlas(model.codec, loader, val_loader, cfg, context, chart_options)
            geometry = ChartMomentumAtlas(atlas, chart_options).to(device)
            geometry_report = {"geometry": "chart_trivialized_mls", "atlas": atlas["diagnostics"]}
        elif use_mls:
            from ..atlas_geometry import MRIAtlasGeometry, load_mri_atlas
            if resume:
                if atlas.get("kind") != "mls":
                    raise ValueError("Momentum resume requires its fitted MLS atlas")
                geometry = MRIAtlasGeometry(atlas).to(device)
            else:
                atlas, geometry = load_mri_atlas(cfg, digest, identity, device)
            if guidance is not None and not geometry.source_routing:
                raise ValueError("Momentum geometry requires the fitted LR source routing")
            geometry_report = {"geometry": "source_conditioned_mls" if guidance is not None else "mls",
                               "atlas": atlas["diagnostics"], "sampling_sigma": sample_sigma,
                               "reference_sigma": reference_sigma, "selection_metric": selection,
                               "validation_conditions": list(validation_settings)}
        else:
            device_atlas = {k: v.to(device) for k, v in atlas.items() if isinstance(v, torch.Tensor)}
            geometry_report = {"geometry": "pca", "training_tokens": atlas["training_tokens"],
                               "pca_centers": len(atlas["centers"]), "tangent_rank": atlas["basis"].shape[-1],
                               "mean_pca_confidence": float(atlas["confidence"].mean())}
        network = (DistributedDataParallel(model, device_ids=[context.local_rank] if device.type == "cuda" else None,
                                           broadcast_buffers=False) if context.is_distributed else model)
        if context.is_main:
            print(json.dumps({"world_size": context.world_size, **geometry_report}), flush=True)
        for epoch in range(start, train_cfg["epochs"]):
            torch.manual_seed(seed + epoch + 100003 * context.rank)
            loader.generator.manual_seed(seed + epoch)
            train_set.set_epoch(epoch)
            if sampler is not None:
                sampler.set_epoch(epoch)
            network.train()
            totals, count = {}, 0
            iterator = iter(loader)
            for _ in range(len(loader)):
                error = None
                try:
                    batch = next(iterator)
                    hr, base, mask = (batch[k].to(device) for k in ("hr", "base", "mask"))
                    bound = None
                    with torch.no_grad():
                        free = model.codec.free(mask)
                        endpoint = model.codec.encode(hr - base) * free
                        condition = model.codec.encode(base)
                        if chart_options is not None:
                            frame = geometry.bind(condition, free)
                            endpoint = frame.to_body(endpoint)
                        else:
                            frame = None
                    if guidance is not None:
                        bound = geometry.bind(base)

                        def phase_cost(position, velocity, free):
                            terms = bound.phase_costs(base + model.codec.decode(position * free),
                                                      model.codec.decode(velocity * free))
                            return tuple(term.mean() for term in terms)

                        path, details = optimize_path(endpoint, free, phase_cost, cfg["path"])
                    elif frame is not None:
                        path, details = frame.optimize(endpoint, cfg["path"])
                    else:
                        cost = (geometry.bind(hr).momentum_cost(base, model.codec) if use_mls
                                else assign_chart(device_atlas, endpoint * (5/16)).cost)
                        path, details = optimize_path(endpoint, free, cost, cfg["path"])
                    time = torch.rand(len(hr), device=device)
                    time = torch.where(torch.rand_like(time) < 0.1, 0.0, time)
                    position, velocity, target, _ = path.sample(time)
                    physical = frame.features(position, velocity) if frame is not None else None
                except Exception as exc:
                    error = f"Momentum path batch: {exc}"
                context.check_errors(error)
                optimizer.zero_grad(set_to_none=True)
                predicted = network(position, velocity, condition, time, physical)
                anchor_error = (predicted - target) * free
                euclidean = anchor_error.square().sum() / (free.sum() * predicted.shape[1]).clamp_min(1)
                terms = {"scaled_acceleration_mse": euclidean}
                loss = euclidean
                if metric_weight:
                    image = base + model.codec.decode(position)
                    metric_error = bound.drift_cost(image, model.codec.decode(anchor_error))
                    ramp = time.pow(cfg["path"].get("geometry_ramp_power", 0.))
                    weighted_metric = (metric_error * ramp).mean()
                    terms["mls_acceleration_matching"] = weighted_metric
                    loss = euclidean + metric_weight * weighted_metric
                context.check_errors(None if torch.isfinite(loss) else "Nonfinite acceleration matching loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                for key, value in {**{name: float(term.detach()) for name, term in terms.items()},
                                   "loss": float(loss.detach()), **details}.items():
                    totals[key] = totals.get(key, 0.0) + value * len(hr)
                count += len(hr)
            keys = sorted(totals)
            global_count, *sums = context.sum([count, *[totals[k] for k in keys]])
            if int(global_count) != len(train_set):
                raise RuntimeError("Unexpected momentum DDP patch count")
            model.eval()
            scores = {name: [0., 0., 0.] for name in validation_settings}
            guidance_delta, items, error = 0., 0, None
            try:
                generators = {name: torch.Generator(device=device).manual_seed(seed + 12345 + context.rank)
                              for name in validation_settings}
                with torch.no_grad():
                    for batch in val_loader:
                        hr, base, mask = (batch[k].to(device) for k in ("hr", "base", "mask"))
                        validation_chart = geometry.bind(base) if guidance is not None else None
                        guided_pred = None
                        for name, (sigma, options) in validation_settings.items():
                            use_geometry = chart_options is not None or options is not None
                            pred = model.restore(base, mask, generator=generators[name],
                                                 steps=cfg["sampling"]["steps"], sigma=sigma,
                                                 geometry=geometry if use_geometry else None,
                                                 geometry_guidance=options).clamp(0, 1)
                            mse = ((pred - hr).square() * (1 - mask)).sum() / (1 - mask).sum().clamp_min(1)
                            scores[name][0] += float(mse) * len(hr)
                            if report_ssim:
                                scores[name][1] += float(mean_xy_ssim(pred, hr)) * len(hr)
                            if validation_chart is not None:
                                scores[name][2] += float(validation_chart.state_cost(pred).sum())
                                if name == "inference":
                                    guided_pred = pred
                                elif name == "without_online_guidance":
                                    difference = (guided_pred - pred).square().flatten(1).mean(1).sqrt()
                                    guidance_delta += float(difference.sum())
                        items += len(hr)
            except Exception as exc:
                error = f"Momentum validation: {exc}"
            context.check_errors(error)
            items, guidance_delta, *values = context.sum(
                [items, guidance_delta, *[value for entry in scores.values() for value in entry]]
            )
            if int(items) != len(val_set) or not math.isfinite(guidance_delta) or not all(map(math.isfinite, values)):
                raise RuntimeError("Unexpected momentum validation accounting")
            validation = {name: tuple(value / items for value in values[3*i:3*i + 3])
                          for i, name in enumerate(scores)}
            primary_mse, primary_ssim, atlas_cost = validation["inference"]
            score = primary_mse if selection == "mse" else 1 - primary_ssim
            improved = score < best
            best = min(best, score)
            record = {"epoch": epoch + 1, "train_items": int(global_count), "validation_items": int(items),
                      "world_size": context.world_size, "validation": primary_mse,
                      **{k: v / global_count for k, v in zip(keys, sums)}}
            if report_ssim:
                record["validation_ssim"] = primary_ssim
                record["selection_metric"] = selection
                record["validation_inference_sigma"] = sample_sigma
                for name, (mse, ssim, energy) in validation.items():
                    if name != "inference":
                        record[f"validation_{name}"] = mse
                        record[f"validation_ssim_{name}"] = ssim
                        record[f"validation_atlas_cost_{name}"] = energy
                if guidance is not None:
                    record["validation_atlas_cost"] = atlas_cost
                    record["validation_guidance_delta_rms"] = guidance_delta / items

            def save():
                state = {"format_version": 1, "stage": "reconstruction", "training_stage": "momentum",
                         "algorithm": ("chart_trivialized_momentum_bridge" if chart_options is not None
                                       else "lr_routed_mls_momentum_bridge" if guidance is not None
                                       else "paired_geometry_conditioned_momentum_bridge"),
                         "epoch": epoch + 1,
                         "model": model.state_dict(), "optimizer": optimizer.state_dict(), "atlas": atlas,
                         "config": cfg, "best": best, "world_size": context.world_size,
                         "manifest_sha256": digest, "data_identity": identity}
                atomic_save(state, run_dir / "last.pt")
                if improved:
                    atomic_save(state, run_dir / "best.pt")
                with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps(record), flush=True)

            context.main_call(save)
        return run_dir
    finally:
        if lock is not None:
            lock.close()
            (run_dir / ".training.lock").unlink(missing_ok=True)
        context.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume")
    parser.add_argument("--case-inventory")
    args = parser.parse_args()
    run_training(args.config, args.device, args.resume, args.case_inventory)


if __name__ == "__main__":
    main()
