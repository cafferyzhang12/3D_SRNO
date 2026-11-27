"""Train an upstream volumetric baseline on the project's fixed paired MRI cohort."""

import argparse
import copy
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .mri_data import PairedMRIPatchDataset
from .mri_infer import _infer_case, checked_test_subset
from .mri_inventory import load_inventory
from .mri_perceptual import PROTOCOL, FeatureMoments, consume_case, frechet_distance, load_extractors
from .runtime import atomic_save, fingerprint, read_config, setup


UPSTREAM = {
    "3d-edm": "d9d82789f1b182cf05f7b654e77bf30a71a57250",
    "medgssr": "77c9f1ccdbe5215d0163f22ccbe1d615be08c4ed",
}
STOP_REQUESTED = False


def request_stop(*_):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def check_stop():
    if STOP_REQUESTED:
        raise SystemExit(75)


def build_baseline(method, root, device):
    if method == "3d-edm":
        sys.path.insert(0, str(root / "3D-EDM" / "src"))
        from models.diffusion import (Denoiser3d, Denoiser3dConfig, EDMSamplerConfig,
                                      InnerModel3dConfig, SigmaDistributionConfig, edm_sample)

        inner = InnerModel3dConfig(img_channels=1, cond_channels=1024, depths=[2, 2, 2, 2],
                                  channels=[32, 64, 128, 256], attn_depths=[0, 0, 0, 1])
        model = Denoiser3d(Denoiser3dConfig(inner, .5, .1, 4)).to(device)
        model.setup_training(SigmaDistributionConfig(loc=-1.2, scale=1.2, sigma_min=.002, sigma_max=20))
        sampler = EDMSamplerConfig(num_steps=20, sigma_min=.002, sigma_max=5., rho=7, order=1)

        def loss(lr, hr):
            return model(lr.mul(2).sub(1), hr.mul(2).sub(1))[0]

        def predict(lr, generator):
            lr_up = F.interpolate(lr.mul(2).sub(1), scale_factor=(1, 4, 4),
                                  mode="trilinear", align_corners=False)
            denoise = lambda volume, sigma: model.denoise(volume, sigma, lr_up)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                result = edm_sample(denoise, lr_up.shape, sampler, device, generator)
            return result.add(1).mul(.5)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
        return model, optimizer, loss, predict, 20

    sys.path.insert(0, str(root / "MedGSSR"))
    from models.medgssr import Net
    from utils import make_coord

    model = Net().to(device)

    @lru_cache(maxsize=3)
    def coordinates(shape):
        return make_coord(shape, flatten=False).to(device)

    def predict(lr, generator):
        coarse = lr.shape[-3:]
        fine = tuple(2 * size for size in coarse)
        target = (coarse[0], 4 * coarse[1], 4 * coarse[2])
        return model(lr, *(coordinates(shape).unsqueeze(0).expand(len(lr), -1, -1, -1, -1)
                           for shape in (coarse, fine, target)))

    def loss(lr, hr):
        return F.l1_loss(predict(lr, None), hr)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return model, optimizer, loss, predict, 1


def verify_cuda_voxelizer(device):
    """Compare the patched CUDA kernel and all three gradients with Eq. 7."""
    from gs_utils.Compute_intensity import compute_intensity
    from utils import make_coord

    shape = (3, 7, 9)
    grid = make_coord(shape).to(device)
    centers = torch.tensor([[.4, .52, .63], [.6, .35, .69]], device=device, requires_grad=True)
    intensity = torch.tensor([[.2], [.3]], device=device, requires_grad=True)
    axis_std = torch.tensor([[.08, .17, .12], [.15, .07, .1]], device=device)
    inverse = torch.diag_embed(axis_std.reciprocal().square()).view(-1, 9).requires_grad_()
    delta = grid[None, :, :] - centers[:, None, :]
    distance = torch.einsum("nvi,nij,nvj->nv", delta, inverse.view(-1, 3, 3), delta)
    reference = torch.where(distance < 9, intensity * torch.exp(-.5 * distance), 0).sum(0)
    actual = compute_intensity(centers, grid, intensity, inverse, axis_std, grid.new_zeros(1, *shape))
    torch.testing.assert_close(actual.reshape(-1), reference, rtol=1e-5, atol=1e-7)
    variables = centers, intensity, inverse
    for measured, expected in zip(torch.autograd.grad(actual.sum(), variables),
                                  torch.autograd.grad(reference.sum(), variables), strict=True):
        torch.testing.assert_close(measured, expected, rtol=2e-5, atol=2e-6)


def aggregate(rows):
    pixels = sum(row["pixel_count"] for row in rows)
    slices = sum(row["ssim_slice_count"] for row in rows)
    mse = sum(row["squared_error_sum"] for row in rows) / pixels
    return dict(case_count=len(rows), slice_count=slices, pixel_count=pixels, all_mse=mse,
                all_psnr_db=-10 * math.log10(mse),
                mean_slice_ssim=sum(row["mean_slice_ssim"] * row["ssim_slice_count"] for row in rows) / slices)


@torch.no_grad()
def evaluate(cases, directory, cfg, device, steps, predict, model, identity, weights_dir=None):
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / "identity.json"
    if receipt.exists():
        if json.loads(receipt.read_text()) != identity:
            raise ValueError(f"Evaluation directory witnesses different inputs: {directory}")
    else:
        atomic_save(identity, receipt, json_file=True)
    if (directory / "summary.json").exists():
        return json.loads((directory / "summary.json").read_text())
    rows = []
    perceptual = None if weights_dir is None else load_extractors(weights_dir, device)
    predicted, reference = FeatureMoments(), FeatureMoments()
    for index, case in enumerate(cases):
        check_stop()
        row_file = directory / f"case-{index:04}.json"
        if perceptual:
            moment_file = directory / f"moments-{index:04}.npz"
        else:
            moment_file = None
        if row_file.exists() and (moment_file is None or moment_file.exists()):
            row = json.loads(row_file.read_text())
            if moment_file is not None:
                with np.load(moment_file) as moments:
                    for prefix, accumulator in (("pred", predicted), ("ref", reference)):
                        accumulator.n += row["evaluated_slices"]
                        accumulator.total += moments[prefix + "_total"]
                        accumulator.outer += moments[prefix + "_outer"]
        else:
            with tempfile.TemporaryDirectory(prefix="mri-baseline-", dir=os.environ.get("SLURM_TMPDIR")) as temp:
                folder = Path(temp) / "tiles"
                _, row = _infer_case(case, folder, model, None, cfg, device, cfg["train"]["validation_seed"],
                                     steps, 1, volume_predictor=predict)
                row.pop("slices")
                if perceptual:
                    volume = np.load(folder / "mean.npy", mmap_mode="r", allow_pickle=False)
                    case_predicted, case_reference = FeatureMoments(), FeatureMoments()
                    row.update(consume_case(volume, case, *perceptual, case_predicted, case_reference, device, 8))
                    volume._mmap.close()
                    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                        np.savez(stream, pred_total=case_predicted.total, pred_outer=case_predicted.outer,
                                 ref_total=case_reference.total, ref_outer=case_reference.outer)
                        temp_moments = stream.name
                    os.replace(temp_moments, moment_file)
                    for accumulator, single in ((predicted, case_predicted), (reference, case_reference)):
                        accumulator.n += single.n
                        accumulator.total += single.total
                        accumulator.outer += single.outer
            atomic_save(row, row_file, json_file=True)
        if row["case_id"] != case.id or row["evaluated_slices"] != len(case.lq):
            raise ValueError("Unexpected case in evaluation cache")
        rows.append(row)

    summary = {**identity, **aggregate(rows), "cases": rows}
    if perceptual:
        if summary["slice_count"] != predicted.n or predicted.n != reference.n:
            raise ValueError("Perceptual and pixel metrics used different cohorts")
        summary.update(lpips=sum(row["lpips_sum"] for row in rows) / predicted.n,
                       fid=frechet_distance(predicted, reference), perceptual_protocol=PROTOCOL,
                       weights=json.loads((weights_dir / "weights.json").read_text()))
    atomic_save(summary, directory / "summary.json", json_file=True)
    return summary


def train(args, cfg, cohorts, device, model, optimizer, loss, predict, steps, identity):
    settings = cfg["train"]
    epochs, seed = settings["epochs"], settings["seed"]
    effective_batch = 4 * settings["batch_size"]
    budget = settings["samples_per_epoch"]
    updates_per_epoch = math.ceil(budget / effective_batch)
    dataset = PairedMRIPatchDataset(None, None, "train", cfg["data"]["scale"], tuple(cfg["data"]["patch_size"]),
                                    cases=cohorts["train"], seed=seed, samples_per_epoch=budget)
    last_file, best_file = args.output / "last.pt", args.output / "best.pt"
    last = torch.load(last_file, map_location="cpu", weights_only=True) if last_file.exists() else None
    start_epoch, cursor = 0, 0
    if last:
        if last["identity"] != identity:
            raise ValueError("Checkpoint belongs to a different training protocol")
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        torch.cuda.set_rng_state(last["cuda_rng"])
        start_epoch, cursor = last["epoch"], last["cursor"]
    best = torch.load(best_file, map_location="cpu", weights_only=True) if best_file.exists() else None
    if best and best["identity"] != identity:
        raise ValueError("Best checkpoint belongs to a different training protocol")
    selected_score = best["score"] if best else (-math.inf, -math.inf)
    del last, best

    def save_progress(epoch, offset):
        atomic_save({"identity": identity, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "cuda_rng": torch.cuda.get_rng_state(), "epoch": epoch, "cursor": offset}, last_file)

    # Match the main study's DDP replacement-free shuffle and global update boundaries.
    for epoch in range(start_epoch, epochs):
        dataset.set_epoch(epoch)
        model.train()
        order = torch.randperm(budget, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        batches = (order[start:min(start + args.micro_batch, end)]
                   for left in range(cursor, budget, effective_batch)
                   for end in (min(left + effective_batch, budget),)
                   for start in range(left, end, args.micro_batch))
        loader = DataLoader(dataset, batch_sampler=list(batches), num_workers=settings["workers"],
                            pin_memory=True, generator=torch.Generator().manual_seed(seed + epoch))
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = torch.zeros((), device=device)
        started = time.monotonic()
        group_end = min(cursor + effective_batch, budget)
        group_size = group_end - cursor
        for batch in loader:
            lr, hr = (batch[key].to(device, non_blocking=True) for key in ("lr", "hr"))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.method == "3d-edm"):
                micro_loss = loss(lr, hr)
            weight = len(lr) / group_size
            (micro_loss * weight).backward()
            accumulated_loss += micro_loss.detach() * weight
            cursor += len(lr)
            if cursor != group_end:
                continue
            update = math.ceil(cursor / effective_batch)
            if args.method == "medgssr":
                progress = (epoch * updates_per_epoch + update - 1) / max(1, epochs * updates_per_epoch - 1)
                optimizer.param_groups[0]["lr"] = 1e-4 * 10 ** -progress
            if args.method == "3d-edm":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            if not torch.isfinite(accumulated_loss):
                raise FloatingPointError("Nonfinite training loss")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            print(json.dumps(dict(epoch=epoch + 1, update=update, samples=cursor, loss=float(accumulated_loss),
                                  elapsed_seconds=time.monotonic() - started)), flush=True)
            accumulated_loss.zero_()
            if update % 4 == 0 or cursor == budget or STOP_REQUESTED:
                save_progress(epoch, cursor)
            check_stop()
            group_end = min(cursor + effective_batch, budget)
            group_size = group_end - cursor

        model.eval()
        validation_id = {**identity, "epoch": epoch + 1, "split": "val"}
        validation = evaluate(cohorts["val"], args.output / f"val-epoch-{epoch + 1:02}", cfg, device, steps,
                              predict, model, validation_id)
        score = (validation["all_psnr_db"], validation["mean_slice_ssim"])
        if score > selected_score:
            selected_score = score
            atomic_save({"identity": identity, "epoch": epoch + 1, "score": score,
                         "model": model.state_dict()}, best_file)
        print(json.dumps({key: val for key, val in validation.items() if key != "cases"}), flush=True)
        save_progress(epoch + 1, 0)
        cursor = 0

    return best_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=UPSTREAM, required=True)
    for option in ("config", "inventory", "manifest", "weights-dir", "upstream", "output"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    parser.add_argument("--micro-batch", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() != 1:
        raise RuntimeError("Use a single-A800 Slurm allocation")
    if args.micro_batch <= 0:
        raise ValueError("Expected a positive microbatch")
    signal.signal(signal.SIGUSR1, request_stop)
    cfg = read_config(args.config)
    settings = cfg["train"]
    if (tuple(cfg["data"]["patch_size"]), cfg["data"]["scale"], settings["epochs"],
            settings["samples_per_epoch"], settings["batch_size"]) != ((5, 128, 128), 4, 8, 262144, 544):
        raise ValueError("Configuration differs from the active four-GPU comparison")
    device = setup(settings["seed"], "cuda", settings.get("threads", 4))
    cohorts, _ = load_inventory(args.inventory, cfg["data"], ("train", "val", "test"))
    cohorts["test"], _ = checked_test_subset(cohorts["test"], cfg["test"]["fraction"],
                                             cfg["test"]["subset_seed"], args.manifest)
    expected = {"train": (3919, 292657), "val": (10, 595), "test": (84, 14991)}
    for split, cases in cohorts.items():
        if (len(cases), sum(len(case.lq) for case in cases)) != expected[split]:
            raise ValueError(f"{split} cohort differs from the main experiment")
    model, optimizer, loss, predict, steps = build_baseline(args.method, args.upstream.resolve(), device)
    if args.method == "medgssr":
        verify_cuda_voxelizer(device)
    cfg = copy.deepcopy(cfg)
    cfg["sampling"]["tile_batch"] = min(args.micro_batch, 8)
    identity = {"method": args.method, "upstream_commit": UPSTREAM[args.method],
                "config_sha256": fingerprint(args.config), "inventory_sha256": fingerprint(args.inventory),
                "manifest_sha256": fingerprint(args.manifest), "source_sha256": os.environ["MRI_BASELINE_SOURCE_SHA256"],
                "steps": steps, "effective_batch": settings["batch_size"] * 4, "micro_batch": args.micro_batch,
                "train_patches_per_epoch": settings["samples_per_epoch"], "epochs": settings["epochs"],
                "initialization": "scratch", "selection": "max whole-VAL pooled PSNR, then slice SSIM",
                "model_parameters": sum(parameter.numel() for parameter in model.parameters())}
    print(json.dumps(identity), flush=True)
    if args.smoke:
        dataset = PairedMRIPatchDataset(None, None, "train", 4, (5, 128, 128),
                                        cases=cohorts["train"], seed=settings["seed"],
                                        samples_per_epoch=args.micro_batch)
        batch = next(iter(DataLoader(dataset, batch_size=args.micro_batch)))
        lr, hr = (batch[key].to(device) for key in ("lr", "hr"))
        before = time.monotonic()
        value = loss(lr, hr)
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            estimate = predict(lr, torch.Generator(device=device).manual_seed(42))
            if estimate.shape != hr.shape or not torch.isfinite(estimate).all():
                raise RuntimeError("Nonfinite or misaligned volumetric prediction")
        load_extractors(args.weights_dir, device)
        print(json.dumps({"smoke": "passed", "method": args.method, "loss": float(value.detach()),
                          "seconds": time.monotonic() - before, "memory": torch.cuda.max_memory_allocated()}), flush=True)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    best_file = train(args, cfg, cohorts, device, model, optimizer, loss, predict, steps, identity)
    selected = torch.load(best_file, map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model"])
    model.eval()
    frozen = {**identity, "checkpoint_sha256": fingerprint(best_file), "epoch": selected["epoch"]}
    freeze_file = args.output / "freeze.json"
    if freeze_file.exists() and json.loads(freeze_file.read_text()) != frozen:
        raise ValueError("Frozen TEST checkpoint changed")
    atomic_save(frozen, freeze_file, json_file=True)
    for split in ("val", "test"):
        summary = evaluate(cohorts[split], args.output / f"final-{split}", cfg, device, steps, predict, model,
                           {**frozen, "split": split}, args.weights_dir)
        print(json.dumps({key: val for key, val in summary.items() if key != "cases"}), flush=True)


if __name__ == "__main__":
    main()
