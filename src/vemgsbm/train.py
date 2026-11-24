"""Paired conditional variational GSBM: CondSOC followed by neural drift matching.

The empirical LR/HR pairing stays fixed. This is the paired restoration
specialization, not the upstream unpaired alternating endpoint-coupling solver.
No IPF or invertible-coordinate bridge is used.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .datasets import training_data
from .distributed import initialize_distributed, ValidationShardSampler
from .metric import Representation, representation_loss, fit_metric, metric_payload, metric_from_payload
from .bridge import reconstruct
from .mri_metrics import mean_xy_ssim
from .network import DriftNetwork, integrate
from .paths import optimize_path, sample_path
from .haar_residual import HAAR_MODES, HaarObjective, make_haar_model
from .training_state import ModelEMA, warmup_cosine
from .runtime import (read_config, project_path, setup, atomic_save, load_checkpoint,
                      sampling_geometry_guidance, sampling_noise)


class RepresentationObjective(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, hr):
        return representation_loss(self.encoder, hr)


class MatchingObjective(nn.Module):
    def __init__(self, model, metric_weight=0.0, ramp_power=0.0):
        super().__init__()
        self.model = model
        self.metric_weight = metric_weight
        self.ramp_power = ramp_power

    def forward(self, state, base, mask, time, forward_target, backward_target, geometry=None):
        free = 1 - mask
        denom = free.sum().clamp_min(1)
        forward = self.model(state, base, mask, time, "forward")
        backward = self.model(state, base, mask, time, "backward")
        ef, eb = forward - forward_target, backward - backward_target
        lf = (ef.square() * free).sum() / denom
        lb = (eb.square() * free).sum() / denom
        terms = {"loss": (lf + lb) / 2, "forward_matching": lf, "backward_matching": lb}
        if self.metric_weight:
            if geometry is None:
                raise ValueError("Metric drift matching requires a bound MLS chart mixture")
            weight = time.pow(self.ramp_power)
            metric_forward, metric_backward = geometry.drift_costs(state, ef * free, eb * free)
            metric_loss = ((metric_forward + metric_backward) * weight).mean() / 2
            terms["metric_matching"] = metric_loss
            terms["loss"] = terms["loss"] + self.metric_weight * metric_loss
        return terms


def _collect_metric(encoder, loader, cfg, context):
    limit = cfg["representation"].get("bank_samples", 1024)
    local_limit = max(1, (limit + context.world_size - 1) // context.world_size)
    features = []
    count = 0
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            value = encoder(batch["hr"].to(context.device))
            features.append(value[:local_limit - count].cpu())
            count += len(features[-1])
            if count >= local_limit:
                break
    local = torch.cat(features)
    if context.is_distributed:
        gathered = [None] * context.world_size
        torch.distributed.all_gather_object(gathered, local)
    else:
        gathered = [local]

    def fit():
        bank = torch.cat(gathered)[:limit].to(context.device)
        metric = fit_metric(encoder, bank, clusters=cfg["representation"].get("clusters", 32),
                            seed=cfg["train"]["seed"])
        return metric_payload(metric)
    return context.main_call(fit, broadcast_result=True)


def _whole_case_validation(model, cases, cfg, context, run_dir, epoch, atlas=None):
    from .mri_infer import _infer_case

    records, error = [], None
    try:
        for case in cases[context.rank::context.world_size]:
            with tempfile.TemporaryDirectory(dir=run_dir) as temp:
                _, row = _infer_case(case, Path(temp) / case.id, model, atlas,
                                     {**cfg, "inference_variant": "endpoint_safe"}, context.device,
                                     cfg["train"].get("validation_seed", 42), cfg["sampling"]["steps"], 1)
            records.append({key: value for key, value in row.items() if key != "slices"})
    except Exception as exc:
        error = f"Whole-case VAL: {type(exc).__name__}: {exc}"
    context.check_errors(error)
    if context.is_distributed:
        gathered = [None] * context.world_size
        torch.distributed.all_gather_object(gathered, records)
        records = [row for shard in gathered for row in shard]
    records.sort(key=lambda row: row["case_id"])
    if [row["case_id"] for row in records] != sorted(case.id for case in cases):
        raise RuntimeError("Incomplete whole-case validation")
    pixels = sum(row["pixel_count"] for row in records)
    mse = sum(row["squared_error_sum"] for row in records) / pixels
    n_ssim = sum(row["ssim_slice_count"] for row in records)
    ssim = (sum((row["mean_slice_ssim"] or 0.) * row["ssim_slice_count"] for row in records)
            / n_ssim if n_ssim else None)
    metrics = {"val_pooled_mse": mse, "val_pooled_psnr_db": -10 * math.log10(mse) if mse else None,
               "val_mean_slice_ssim": ssim, "val_pixel_count": pixels, "val_case_count": len(records)}
    edge_voxels = sum(row["edge_voxels"] for row in records)
    metrics["val_edge_mse"] = (sum(row["edge_sse"] for row in records) / edge_voxels if edge_voxels else None)
    band_count = sum(row.get("haar_band_count",0) for row in records)
    if band_count:
        metrics["val_haar_band_mse"] = [sum(row["haar_band_sse"][i] for row in records)/band_count for i in range(16)]
        metrics["val_haar_high_frequency_mse"] = sum(metrics["val_haar_band_mse"][1:])/15
    context.main_call(lambda: atomic_save({"cases": records, **metrics},
                                          run_dir / "whole_val" / f"epoch_{epoch:04}.json", json_file=True))
    return metrics


def _prepare_spatial_bank(cfg, dataset, identity, device, run_dir):
    """Build once on rank zero or verify a prebuilt bank against the TRAIN cohort and A."""
    from .spatial_atlas import bank_provenance, build_spatial_atlas

    options = cfg["spatial_atlas"]
    path = project_path(options["direct_checkpoint"])
    checkpoint = load_checkpoint(path)
    train_identity = [row for row in identity if row["split"] == "train"]
    if (checkpoint.get("format_version") != 3 or checkpoint.get("algorithm") != "haar_direct"
            or [row for row in checkpoint["data_identity"] if row["split"] == "train"] != train_identity):
        raise ValueError("Spatial descriptors require A trained on exactly the same TRAIN cohort")
    # Fit records deterministically across bridge seeds; routing/curvature settings
    # stay explicit so a cached fit cannot silently enter a different ablation.
    provenance = bank_provenance(path, identity, options)
    cached_path = project_path(options["bank"]) if options.get("bank") else None
    gate_bank = run_dir.parent / "smoke" / "spatial_bank.pt"
    if cached_path is None and run_dir.name == "bridge" and gate_bank.is_file():
        cached_path = gate_bank
    if cached_path is not None:
        cached = load_checkpoint(cached_path)
        if cached.get("provenance") != provenance:
            raise ValueError("Spatial bank identity, settings or descriptor checkpoint differs")
        return str(cached_path.resolve())
    bank_set = copy.copy(dataset)
    bank_set.seed = options.get("seed", 42)
    bank_set.set_epoch(0)
    payload = build_spatial_atlas(checkpoint, bank_set, options, device)
    destination = run_dir / "spatial_bank.pt"
    atomic_save({"provenance": provenance, "payload": payload}, destination)
    return str(destination.resolve())


def _publish_epoch_history(run_dir, records):
    # last.pt is the commit point. On resume, replay its complete history if
    # preemption fell between the checkpoint and this small JSONL replacement.
    with tempfile.NamedTemporaryFile("w", dir=run_dir, encoding="utf-8", delete=False) as stream:
        temp = Path(stream.name)
        for row in records:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    try:
        os.replace(temp, run_dir / "metrics.jsonl")
    finally:
        temp.unlink(missing_ok=True)


def run_training(config_path, stage="bridge", device_name="auto", resume=None, case_inventory=None):
    cfg = read_config(config_path)
    haar = cfg["mode"] in HAAR_MODES
    x1 = cfg.get("predict_target") == "x1"
    if haar and (stage != "bridge" or cfg.get("atlas")):
        raise ValueError("Train native LR residual models with stage=bridge and no legacy atlas")
    atlas_options = cfg.get("atlas")
    online_guidance = sampling_geometry_guidance(cfg)
    source_routing = bool(atlas_options and atlas_options.get("source_routing"))
    matching_weight = float(cfg["train"].get("metric_matching_weight", 0.0))
    if not math.isfinite(matching_weight) or matching_weight < 0 or (matching_weight and not source_routing):
        raise ValueError("metric_matching_weight must be nonnegative and requires LR chart routing")
    if not haar and cfg["mode"] != "gsbm" and not (atlas_options and stage == "representation"):
        raise ValueError("Use vemgsbm.momentum.train for momentum configs")
    if stage not in {"representation", "bridge"}:
        raise ValueError("Unknown stage")
    selection_metric = cfg["train"].get("selection_metric", "mse")
    if selection_metric not in ("mse", "ssim") and not (haar and selection_metric == "pooled_psnr"):
        raise ValueError("train.selection_metric must be mse or ssim")
    paired_mri = cfg["data"].get("kind") == "paired_mri_png"
    if selection_metric == "ssim" and not paired_mri:
        raise ValueError("SSIM checkpoint selection requires paired MRI")
    measure_ssim = stage == "bridge" and paired_mri and (selection_metric == "ssim" or online_guidance)
    inference_sigma = sampling_noise(cfg)
    cfg["train"]["epochs"] = cfg["train"].get(stage + "_epochs", cfg["train"]["epochs"])
    context = initialize_distributed(device_name)
    run_dir = project_path(cfg["run_dir"]) / stage
    lock = None
    lock_dir_fd = None

    def prepare():
        nonlocal lock, lock_dir_fd
        run_dir.mkdir(parents=True, exist_ok=True)
        if (run_dir / "last.pt").exists() and not resume:
            raise FileExistsError("Existing run requires --resume or a new run_dir")
        if haar and os.name == "posix":
            # Slurm preemption releases this kernel lock without leaving a stale sentinel.
            import fcntl
            lock_dir_fd = os.open(run_dir, os.O_RDONLY)
            fcntl.flock(lock_dir_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            lock = (run_dir / ".training.lock").open("x")
        atomic_save(cfg, run_dir / "config.json", json_file=True)

    try:
        context.assert_same((cfg, stage, resume), "config/stage/resume")
        context.main_call(prepare)
        device = setup(cfg["train"]["seed"], str(context.device), cfg["train"].get("threads", 4))
        if case_inventory is None:
            prepared = context.main_call(lambda: training_data(cfg), broadcast_result=True)
        else:
            # The Slurm preflight ran before NCCL. Each rank reads the compact
            # inventory locally instead of broadcasting millions of paths.
            prepared, error = None, None
            try:
                prepared = training_data(cfg, case_inventory)
            except Exception as exc:
                error = f"Prepared MRI inventory: {type(exc).__name__}: {exc}"
            context.check_errors(error)
        train_set, val_set, digest, identity = prepared
        if case_inventory is not None:
            context.assert_same(digest, "MRI inventory identity")
        if len(train_set) % context.world_size:
            raise ValueError("Global patch budget must divide WORLD_SIZE")
        sampler = DistributedSampler(train_set, context.world_size, context.rank, seed=cfg["train"]["seed"]) if context.is_distributed else None
        loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], sampler=sampler,
                            shuffle=sampler is None, num_workers=cfg["train"].get("workers", 0),
                            generator=torch.Generator().manual_seed(cfg["train"]["seed"]),
                            pin_memory=device.type == "cuda")
        val_loader = DataLoader(val_set, batch_size=cfg["train"]["batch_size"],
                               sampler=ValidationShardSampler(val_set, context.rank, context.world_size), num_workers=0)
        encoder = None if haar else Representation(width=cfg["representation"]["width"],
                                                   feature_dim=cfg["representation"]["feature_dim"],
                                                   packed_xy=bool(atlas_options)).to(device)
        model, metric = None, None
        atlas_payload = None
        spatial_payload = None
        old = load_checkpoint(project_path(resume)) if resume else None
        if haar:
            model = make_haar_model(cfg).to(device)
            if x1 and cfg.get("use_band_whitening",True):
                from .haar_stats import ensure_haar_statistics
                if old is not None:
                    band_stats = old["band_statistics"]
                else:
                    band_stats = context.main_call(lambda: ensure_haar_statistics(
                        train_set.all_cases, project_path(cfg["band_stats_cache"]), device=device),
                        broadcast_result=True)
                model.set_band_statistics(band_stats)
            else:
                band_stats = None
            teacher = None
            if x1 and cfg.get("mls_reg_weight",0.):
                from .spatial_atlas import TrainingMLSAtlas, prepare_training_mls
                mls_payload = (old["training_mls"] if old else context.main_call(
                    lambda: prepare_training_mls(train_set,model,cfg["mls_regularizer"]),broadcast_result=True))
                metric = TrainingMLSAtlas(mls_payload,cfg["mls_regularizer"]).to(device).eval()
            if x1 and cfg.get("straightening"):
                from .runtime import build_models
                saved_teacher = load_checkpoint(project_path(cfg["straightening"]["teacher_checkpoint"]))
                if saved_teacher["data_identity"] != identity:
                    raise ValueError("Straightening teacher must use the same data identity")
                teacher, _ = build_models(saved_teacher,device)
                if getattr(teacher,"predict_target",None) != "x1":
                    raise ValueError("Straightening requires an x1 teacher")
                if teacher.sigma_0 != model.sigma_0 or teacher.use_band_whitening != model.use_band_whitening:
                    raise ValueError("Straightening requires the teacher prior and whitening convention")
                if any(not torch.equal(getattr(teacher,key),getattr(model,key)) for key in ("band_mean","band_std")):
                    raise ValueError("Straightening requires identical teacher/student whitening")
                teacher.eval().requires_grad_(False)
            if cfg.get("spatial_atlas"):
                from .spatial_atlas import load_spatial_atlas

                if old is not None:
                    spatial_payload = old["spatial_atlas"]
                else:
                    bank_path = context.main_call(
                        lambda: _prepare_spatial_bank(cfg, train_set, identity, device, run_dir),
                        broadcast_result=True)
                    spatial_payload = load_checkpoint(bank_path)["payload"]
                metric = load_spatial_atlas(spatial_payload, cfg["spatial_atlas"]).to(device).requires_grad_(False)
                metric.eval()
            objective = HaarObjective(model, cfg, metric, teacher)
        elif stage == "representation":
            objective = RepresentationObjective(encoder)
        else:
            if atlas_options:
                from .atlas_geometry import load_mri_atlas
                atlas_payload, metric = load_mri_atlas(cfg, digest, identity, device)
            else:
                saved_rep = load_checkpoint(project_path(cfg["representation"]["checkpoint"]))
                if saved_rep.get("stage") != "representation" or saved_rep.get("metric") is None:
                    raise ValueError("Bridge training requires a completed representation checkpoint with a fitted metric")
                if saved_rep["manifest_sha256"] != digest or saved_rep["data_identity"] != identity:
                    raise ValueError("Representation data identity differs")
                if saved_rep["config"]["data"] != cfg["data"]:
                    raise ValueError("Representation input contract differs")
                metric = metric_from_payload(saved_rep["metric"], device=device)
            model = DriftNetwork(cfg["model"]["width"]).to(device)
            objective = MatchingObjective(model, matching_weight,
                                          cfg["path"].get("geometry_ramp_power", 0.0))
        parameters = [p for p in objective.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=cfg["train"]["learning_rate"], weight_decay=1e-5)
        scheduler = warmup_cosine(optimizer,cfg["train"],len(loader)) if x1 else None
        ema = ModelEMA(model,cfg["train"].get("ema_decay",.999)) if x1 else None
        start, best, stale_epochs = 0, float("inf"), 0
        history = []
        if resume:
            if old["training_stage"] != stage or old["manifest_sha256"] != digest or old["data_identity"] != identity:
                raise ValueError("Resume identity differs")
            if old["world_size"] != context.world_size:
                raise ValueError("Resume requires the same WORLD_SIZE")
            current, previous = copy.deepcopy(cfg), copy.deepcopy(old["config"])
            for value in (current, previous):
                for key in ("epochs", "representation_epochs", "bridge_epochs"):
                    value["train"].pop(key, None)
            if current != previous:
                raise ValueError("Resume config changed beyond epoch budget")
            if atlas_payload is not None:
                fields = ("atlas_version", "source_routing_version")
                if any(old["metric"]["diagnostics"].get(key) != atlas_payload["diagnostics"].get(key)
                       for key in fields):
                    raise ValueError("Resume requires the original MRI atlas and LR chart routing")
            (encoder if stage == "representation" else model).load_state_dict(old["model"])
            optimizer.load_state_dict(old["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(old["scheduler"])
                ema.model.load_state_dict(old["ema_model"])
                ema.updates = old["ema_updates"]
            start, best = old["epoch"], old["best"]
            stale_epochs = old.get("stale_epochs", 0)
            if haar:
                if "epoch_history" in old:
                    history = old["epoch_history"]
                    context.main_call(lambda: _publish_epoch_history(run_dir, history))
                elif (run_dir / "metrics.jsonl").is_file():
                    history = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
                    history = [row for row in history if row["epoch"] <= start]
        raw_objective = objective
        if context.is_distributed:
            objective = DistributedDataParallel(objective, device_ids=[context.local_rank] if device.type == "cuda" else None,
                                                broadcast_buffers=False)
        if context.is_main:
            print(json.dumps({"stage": stage, "world_size": context.world_size,
                              "per_rank_batch": cfg["train"]["batch_size"],
                              "global_patches": len(train_set), "method": cfg["mode"] if haar else "paired conditional variational GSBM",
                              "geometry": ((cfg.get("mls_regularizer",{}).get("geometry_mode","training_mls")
                                            if x1 and cfg.get("mls_reg_weight",0.) else
                                            cfg.get("spatial_atlas", {}).get("method", "none")) if haar else
                                           ("source_conditioned_mls" if source_routing else ("mls" if atlas_options else "land"))),
                              "atlas": atlas_payload["diagnostics"] if atlas_payload else None}), flush=True)
        for epoch in range(start, cfg["train"]["epochs"]):
            if (selection_metric == "pooled_psnr" and cfg["train"].get("whole_val_patience") is not None
                    and epoch >= cfg["train"].get("minimum_epochs", 0)
                    and stale_epochs >= cfg["train"]["whole_val_patience"]):
                break
            torch.manual_seed(cfg["train"]["seed"] + epoch + 100003 * context.rank)
            loader.generator.manual_seed(cfg["train"]["seed"] + epoch)
            train_set.set_epoch(epoch)
            if sampler:
                sampler.set_epoch(epoch)
            objective.train()
            if spatial_payload is not None or (x1 and cfg.get("mls_reg_weight",0.)):
                metric.eval()
            totals = {}
            count = 0
            previous_update = {}
            reported_count = 0
            iterator = iter(loader)
            for iteration in range(len(loader)):
                error = None
                try:
                    batch = next(iterator)
                    hr, base, mask = (batch[k].to(device) for k in ("hr", "base", "mask"))
                    path_diagnostics = {}
                    if haar:
                        lr = batch["lr"].to(device)
                    elif stage == "bridge":
                        geometry = metric.bind(base if source_routing else hr) if atlas_options else metric
                        path, path_diagnostics = optimize_path(base, hr, mask, geometry, cfg["path"])
                        path_diagnostics["updated"] = float(path_diagnostics["accepted_steps"] > 0)
                        cutoff = cfg["train"].get("matching_time_min", 0.02)
                        times = torch.rand(len(hr), device=device) * (1 - 2 * cutoff) + cutoff
                        noise = torch.randn_like(hr)
                        state, forward_target = sample_path(path, times, noise=noise, direction="forward")
                        _, backward_target = sample_path(path, times, noise=noise, direction="backward")
                except Exception as exc:
                    error = f"Batch/path: {type(exc).__name__}: {exc}"
                context.check_errors(error)
                optimizer.zero_grad(set_to_none=True)
                error = None
                try:
                    if haar:
                        terms = objective(lr, base, hr, batch["id"])
                    elif stage == "representation":
                        terms = objective(hr)
                    else:
                        terms = objective(state.detach(), base, mask, times, forward_target.detach(),
                                          backward_target.detach(), geometry=geometry if source_routing else None)
                    loss = terms["loss"]
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite matching/representation loss")
                except Exception as exc:
                    error = f"Forward: {type(exc).__name__}: {exc}"
                context.check_errors(error)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg["train"].get("clip_grad_norm",1.0),
                                                          error_if_nonfinite=True)
                optimizer.step()
                if ema is not None:
                    ema.update(model)
                    scheduler.step()
                    terms["grad_norm"] = grad_norm.detach()
                for key, value in terms.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach()) * len(hr)
                for key, value in path_diagnostics.items():
                    if isinstance(value, (int, float)):
                        totals["path_" + key] = totals.get("path_" + key, 0.0) + value * len(hr)
                count += len(hr)
                interval = cfg["train"].get("report_every_updates", 16 if haar else 0)
                if interval and (iteration == 0 or (iteration + 1) % interval == 0):
                    names = [k for k in ("loss", "endpoint_pixel_mse", "rollout_pixel_mse",
                                         "coefficient_loss", "mls_normal_loss", "grad_norm",
                                         "start_pixel_mse", "start_hf_mse", "start_edge_mse",
                                         "start_mls_reconstruction_mse", "rollout_hf_mse",
                                         "rollout_edge_mse", "rollout_mls_reconstruction_mse",
                                         "forward_endpoint_equivalent_mse", "reverse_endpoint_equivalent_mse")
                             if k in totals]
                    values = context.sum([count - reported_count] +
                                         [totals[k] - previous_update.get(k, 0.) for k in names])
                    if context.is_main:
                        print(json.dumps(dict(epoch=epoch+1, update=iteration+1, updates_per_epoch=len(loader),
                                              patches_in_interval=int(values[0]),
                                              **{k: v/values[0] for k, v in zip(names, values[1:])})), flush=True)
                    previous_update = {k: totals[k] for k in names}
                    reported_count = count
            keys = sorted(totals)
            reduced = context.sum([count] + [totals[k] for k in keys])
            count, values = int(reduced[0]), reduced[1:]
            if count != len(train_set):
                raise RuntimeError("DDP training sample count mismatch")
            record = {"epoch": epoch + 1, "train_items": count, "world_size": context.world_size,
                      **{k: v / count for k, v in zip(keys, values)}}
            if haar:
                record.update(cumulative_patches=(epoch + 1) * count,
                              optimizer_steps=(epoch + 1) * len(loader),
                              global_batch_size=context.world_size * cfg["train"]["batch_size"])
                if cfg["mode"] == "haar_direct" or "rollout_pixel_mse" in record:
                    mse = record.get("rollout_pixel_mse", record["endpoint_pixel_mse"])
                    record["train_patch_recon_psnr_db"] = -10 * math.log10(mse) if mse > 0 else None
            raw_objective.eval()
            eval_model = ema.model if ema is not None else model
            total, unguided_total, guidance_delta, items, error = 0.0, 0.0, 0.0, 0, None
            ssim_total, unguided_ssim_total, atlas_total, unguided_atlas_total = 0.0, 0.0, 0.0, 0.0
            try:
                val_seed = cfg["train"]["seed"] + 12345 + context.rank
                torch.manual_seed(val_seed)
                if online_guidance and stage == "bridge":
                    guided_rng = torch.Generator(device=device).manual_seed(val_seed)
                    unguided_rng = torch.Generator(device=device).manual_seed(val_seed)
                with torch.no_grad():
                    for batch in val_loader:
                        hr, base, mask = (batch[k].to(device) for k in ("hr", "base", "mask"))
                        if haar:
                            prediction = eval_model.restore(batch["lr"].to(device), base, cfg["sampling"]["steps"],
                                                       inference_sigma, atlas=None if x1 else metric).clamp(0, 1)
                            value = (prediction - hr).square().mean()
                        elif stage == "representation":
                            value = representation_loss(encoder, hr)["loss"]
                        elif online_guidance:
                            steps = cfg["sampling"]["steps"]
                            prediction = reconstruct(model, base, mask, geometry=metric, steps=steps,
                                                     sigma=inference_sigma,
                                                     generator=guided_rng, terminal_noise=False,
                                                     geometry_guidance=online_guidance)
                            unguided = integrate(model, base, base, mask, steps, inference_sigma,
                                                 generator=unguided_rng, terminal_noise=False)
                            free = 1 - mask
                            denom = free.sum().clamp_min(1)
                            delta = ((prediction - unguided).square() * free).sum() / denom
                            guidance_delta += float(delta) * len(hr)
                            prediction, unguided = prediction.clamp(0, 1), unguided.clamp(0, 1)
                            value = ((prediction - hr).square() * free).sum() / denom
                            reference = ((unguided - hr).square() * free).sum() / denom
                            unguided_total += float(reference) * len(hr)
                            if measure_ssim:
                                unguided_ssim_total += float(mean_xy_ssim(unguided, hr)) * len(hr)
                            chart = metric.bind(base)
                            atlas_total += float(chart.state_cost(prediction).sum())
                            unguided_atlas_total += float(chart.state_cost(unguided).sum())
                        else:
                            prediction = integrate(model, base, base, mask,
                                                   cfg["sampling"]["steps"], inference_sigma,
                                                   terminal_noise=not atlas_options).clamp(0, 1)
                            value = ((prediction - hr).square() * (1 - mask)).sum() / (1 - mask).sum().clamp_min(1)
                        if measure_ssim:
                            ssim_total += float(mean_xy_ssim(prediction, hr)) * len(hr)
                        total += float(value) * len(hr)
                        items += len(hr)
            except Exception as exc:
                error = f"Validation: {type(exc).__name__}: {exc}"
            context.check_errors(error)
            reduced = context.sum([total, items, unguided_total, guidance_delta, ssim_total,
                                   unguided_ssim_total, atlas_total, unguided_atlas_total])
            (total, items, unguided_total, guidance_delta, ssim_total,
             unguided_ssim_total, atlas_total, unguided_atlas_total) = reduced
            if int(items) != len(val_set) or not all(map(math.isfinite, reduced)):
                raise RuntimeError("Invalid validation accounting/score")
            validation = total / items
            score = 1 - ssim_total / items if stage == "bridge" and selection_metric == "ssim" else validation
            if selection_metric == "pooled_psnr":
                whole_metrics = _whole_case_validation(eval_model, val_set.all_cases, cfg, context, run_dir, epoch + 1,
                                                      atlas=None if x1 else metric)
                record.update(whole_metrics)
                score = whole_metrics["val_pooled_mse"]
            improved = score < best
            stale_epochs = 0 if improved else stale_epochs + 1
            best = min(best, score)
            record.update(validation=validation, validation_items=int(items), selection_score=score,
                          selection_metric=selection_metric if stage == "bridge" else "representation_loss")
            if x1:
                record.update(learning_rate=optimizer.param_groups[0]["lr"],ema_updates=ema.updates,
                              peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
            if stage == "bridge":
                record["validation_inference_sigma"] = inference_sigma
            if measure_ssim:
                record["validation_ssim"] = ssim_total / items
            if online_guidance and stage == "bridge":
                record["validation_without_online_guidance"] = unguided_total / items
                record["validation_guidance_delta_rms"] = math.sqrt(guidance_delta / items)
                record["validation_atlas_cost"] = atlas_total / items
                record["validation_atlas_cost_without_online_guidance"] = unguided_atlas_total / items
                if measure_ssim:
                    record["validation_ssim_without_online_guidance"] = unguided_ssim_total / items
            # Fit only on distributed TRAIN HR features; validation selects the
            # encoder checkpoint, never contributes centers or metric statistics.
            payload = atlas_payload
            if stage == "representation":
                if atlas_options:
                    from .atlas_geometry import collect_features, fit_mri_atlas
                    limit = cfg["representation"]["bank_samples"]
                    local_limit = (limit + context.world_size - 1) // context.world_size
                    features = collect_features(encoder, loader, local_limit, context, paired=source_routing)
                    bank, sources = features if source_routing else (features, None)
                    bank = bank[:limit]
                    held_out = collect_features(encoder, val_loader, len(val_set), context)
                    payload = context.main_call(
                        lambda: fit_mri_atlas(encoder, bank, held_out, atlas_options, cfg["train"]["seed"],
                                              source_features=sources[:limit] if sources is not None else None),
                        broadcast_result=True,
                    )
                    record["atlas"] = payload["diagnostics"]
                else:
                    payload = _collect_metric(encoder, loader, cfg, context)
            elif not atlas_options and not haar:
                payload = metric_payload(metric)

            def save():
                checkpoint = {"format_version": 5 if x1 else (4 if spatial_payload is not None else (3 if haar else 2)),
                              "algorithm": cfg["mode"] if haar else "paired_conditional_variational_gsbm",
                              "stage": "representation" if stage == "representation" else "reconstruction",
                              "training_stage": stage, "config": cfg, "epoch": epoch + 1,
                              "model": (encoder if stage == "representation" else model).state_dict(),
                              "optimizer": optimizer.state_dict(), "best": best, "world_size": context.world_size,
                              "manifest_sha256": digest, "data_identity": identity,
                              "metric": payload, "stale_epochs": stale_epochs}
                if spatial_payload is not None:
                    checkpoint["spatial_atlas"] = spatial_payload
                if x1:
                    checkpoint.update(ema_model=ema.model.state_dict(),ema_updates=ema.updates,
                                      evaluation_weights="ema",scheduler=scheduler.state_dict(),
                                      band_statistics=band_stats)
                    if cfg.get("mls_reg_weight",0.):
                        checkpoint["training_mls"] = mls_payload
                if haar:
                    history.append(record)
                    checkpoint["epoch_history"] = history
                if improved:
                    atomic_save(checkpoint, run_dir / "best.pt")
                atomic_save(checkpoint, run_dir / "last.pt")
                if x1 and epoch+1 in cfg["train"].get("checkpoint_epochs",[15,24,40]):
                    atomic_save(checkpoint, run_dir / f"epoch_{epoch+1:04}.pt")
                if haar:
                    _publish_epoch_history(run_dir, history)
                else:
                    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            context.main_call(save)
        return run_dir
    finally:
        if lock_dir_fd is not None:
            os.close(lock_dir_fd)
        if lock is not None:
            lock.close()
            (run_dir / ".training.lock").unlink(missing_ok=True)
        context.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["representation", "bridge"], default="bridge")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume")
    parser.add_argument("--case-inventory")
    args = parser.parse_args()
    run_training(args.config, args.stage, args.device, args.resume, args.case_inventory)


if __name__ == "__main__":
    main()
