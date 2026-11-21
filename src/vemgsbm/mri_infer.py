"""Full-case XY super-resolution from paired-MRI-style LR PNG directories.

Real LR slices define every input voxel. Slice coordinates remain unchanged,
and quarantined interior gaps split the volume into independent contiguous runs.
GPU inference is limited to one overlapping 3D patch at a time.
Under torchrun, ranks reconstruct disjoint cases and publish one complete result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from functools import lru_cache
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
from time import perf_counter
import time

import numpy as np
from scipy.ndimage import gaussian_laplace, sobel
from skimage.metrics import structural_similarity
import torch
import torch.nn.functional as functional

from .bridge import reconstruct
from .distributed import initialize_distributed
from .infer import tile_starts
from .mri_data import index_cases, read_plane
from .network import diffusion_adjusted_model
from .runtime import (atomic_save, build_models, fingerprint, load_checkpoint, project_path,
                      sampling_geometry_guidance, sampling_momentum_guidance, sampling_noise, setup)


def _read_lq(case, index):
    plane = read_plane(
        case.lq[index], normalization_scale=case.normalization_scale,
        normalization_offset=case.normalization_offset, channel_policy=case.channel_policy,
        expected_bits=case.bits,
    )
    if plane.shape != case.lr_shape:
        raise ValueError(f"MRI LR shape changed after indexing: {case.lq[index]}")
    return plane


def make_mri_tile(case, run, start, patch, scale, reader=None, with_native_lr=False):
    """Build a zero-mask HR-grid tile solely from a contiguous run of real LR.

    A one-pixel LR halo preserves global align_corners=False bilinear values at
    overlapping XY tile boundaries. Padding repeats only the current run's edge.
    """
    run_start, run_stop = run
    z, y, x = start
    dz, dy, dx = patch
    height, width = case.lr_shape
    output_height, output_width = height * scale, width * scale
    if not 0 <= run_start < run_stop <= len(case.lq):
        raise ValueError("Invalid contiguous slice run")
    if min(start) < 0 or z >= run_stop - run_start or y >= output_height or x >= output_width:
        raise ValueError("Tile must start within its contiguous run and output XY grid")
    y0, x0 = max(0, y // scale - 1), max(0, x // scale - 1)
    y1 = min(height, math.ceil(min(y + dy, output_height) / scale) + 1)
    x1 = min(width, math.ceil(min(x + dx, output_width) / scale) + 1)

    def read_index(index):
        return _read_lq(case, index)

    reader = reader or read_index
    indices = range(run_start + z, min(run_start + z + dz, run_stop))
    planes = [reader(index)[y0:y1, x0:x1] for index in indices]
    lr = torch.from_numpy(np.stack(planes)).float()[:, None]
    upsampled = functional.interpolate(lr, scale_factor=scale, mode="bilinear", align_corners=False)
    local_y, local_x = y - y0 * scale, x - x0 * scale
    upsampled = upsampled[:, 0, local_y:local_y + dy, local_x:local_x + dx][None, None]
    padding = (0, dx - upsampled.shape[-1], 0, dy - upsampled.shape[-2], 0, dz - upsampled.shape[-3])
    base = functional.pad(upsampled, padding, mode="replicate")
    result = (base, torch.zeros_like(base))
    if not with_native_lr:
        return result
    if any(value % scale for value in (y, x, dy, dx)):
        raise ValueError("Native LR tile boundaries must align with the LR grid")
    ly, lx = y // scale - y0, x // scale - x0
    native = lr[:, 0, ly:ly + dy // scale, lx:lx + dx // scale][None, None]
    pad = (0, dx // scale - native.shape[-1], 0, dy // scale - native.shape[-2],
           0, dz - native.shape[-3])
    return (*result, functional.pad(native, pad, mode="replicate"))


def _psnr(mse):
    # JSON-safe representation of the infinite PSNR of a perfect reconstruction.
    return -10 * math.log10(mse) if mse > 0 else None


METRIC_PROTOCOL = {
    "foreground": "HR > 2/255 in normalized intensity units",
    "edge_mse": "foreground voxels with nonzero XY Sobel magnitude >= per-slice foreground p90",
    "hfen": "pooled L2(XY LoG(pred-HR)) / L2(XY LoG(HR)), sigma 1.5 HR pixels, reflect",
    "z_difference_mse": "voxel MSE of first differences over consecutive original slice indices",
    "all_psnr_db": "10 log10(1 / pooled normalized voxel MSE)",
    "mean_slice_ssim": "arithmetic mean of slice SSIM at unit data range",
}


def _regional_metrics(totals):
    foreground_mse = (totals["foreground_sse"] / totals["foreground_voxels"]
                      if totals["foreground_voxels"] else None)
    return {
        "foreground_mse": foreground_mse,
        "edge_mse": totals["edge_sse"] / totals["edge_voxels"] if totals["edge_voxels"] else None,
        "foreground_psnr_db": _psnr(foreground_mse) if foreground_mse is not None else None,
        "hfen": (math.sqrt(totals["log_error_energy"] / totals["log_reference_energy"])
                 if totals["log_reference_energy"] > 0 else None),
        "z_difference_mse": (totals["z_difference_sse"] / totals["z_difference_voxels"]
                             if totals["z_difference_voxels"] else None),
    }


def _region_totals():
    return {"foreground_voxels": 0, "foreground_sse": 0., "edge_voxels": 0, "edge_sse": 0., "log_error_energy": 0.,
            "log_reference_energy": 0., "z_difference_sse": 0., "z_difference_voxels": 0}


def case_metrics(prediction, case):
    """Evaluate each retained original slice once, with no Z-gap exclusions."""
    if case.hq is None:
        raise ValueError("Full-case metrics require HQ paths")
    if prediction.shape[0] != len(case.hq):
        raise ValueError("Prediction must cover every retained HQ slice exactly once")
    records, squared_error, pixel_count = [], 0.0, 0
    regional = _region_totals()
    previous_error = None
    for z, (stem, original_index, reference_path) in enumerate(zip(case.stems, case.indices, case.hq)):
        reference = read_plane(
            reference_path, normalization_scale=case.normalization_scale,
            normalization_offset=case.normalization_offset, channel_policy=case.channel_policy,
            expected_bits=case.bits,
        ).astype(np.float64)
        pred = np.asarray(prediction[z], dtype=np.float64)
        if pred.shape != reference.shape:
            raise ValueError(f"Prediction/HQ XY shapes differ for {case.id}/{stem}")
        if not np.isfinite(pred).all() or pred.min() < 0 or pred.max() > 1:
            raise ValueError("Metric inputs must be normalized finite values in [0, 1]")
        residual = pred - reference
        error = np.square(residual)
        foreground = reference > float(np.float32(2 / 255))
        regional["foreground_voxels"] += int(foreground.sum())
        regional["foreground_sse"] += float(error[foreground].sum())
        if foreground.any():
            gradient = np.hypot(sobel(reference, axis=0), sobel(reference, axis=1))
            edges = foreground & (gradient > 0) & (gradient >= np.quantile(gradient[foreground], .9))
            regional["edge_voxels"] += int(edges.sum())
            regional["edge_sse"] += float(error[edges].sum())
        regional["log_error_energy"] += float(np.square(gaussian_laplace(residual, 1.5)).sum())
        regional["log_reference_energy"] += float(np.square(gaussian_laplace(reference, 1.5)).sum())
        if previous_error is not None and original_index == case.indices[z - 1] + 1:
            regional["z_difference_sse"] += float(np.square(residual - previous_error).sum())
            regional["z_difference_voxels"] += residual.size
        previous_error = residual
        mse = float(error.mean())
        squared_error += float(error.sum())
        pixel_count += error.size
        window = min(7, min(reference.shape))
        window -= 1 - window % 2
        ssim = None
        if window >= 3:
            ssim = float(structural_similarity(reference, pred, data_range=1.0, win_size=window))
        records.append({
            "stem": stem, "original_index": int(original_index), "output_z": z,
            "mse": mse, "psnr_db": _psnr(mse), "perfect_match": mse == 0, "ssim": ssim,
        })
    mse = squared_error / pixel_count
    ssim_values = [record["ssim"] for record in records if record["ssim"] is not None]
    return {
        "case_id": case.id, "evaluated_slices": len(records), "pixel_count": pixel_count,
        "squared_error_sum": squared_error, "all_mse": mse, "all_psnr_db": _psnr(mse),
        "perfect_match": mse == 0,
        "mean_slice_ssim": float(np.mean(ssim_values)) if ssim_values else None,
        "ssim_slice_count": len(ssim_values), "slices": records,
        **regional, **_regional_metrics(regional),
    }


def _validate_mri_checkpoint(checkpoint):
    if checkpoint.get("stage") != "reconstruction":
        raise ValueError("MRI inference requires a reconstruction checkpoint")
    config = checkpoint["config"]
    data = config["data"]
    if data.get("kind") != "paired_mri_png" or data.get("degradation") != "paired_xy":
        raise ValueError("Use a paired_mri_png / paired_xy checkpoint for MRI inference")
    scale = data["scale"]
    patch = data["patch_size"]
    if scale not in (2, 4) or isinstance(scale, bool):
        raise ValueError("MRI XY scale must be 2 or 4")
    if len(patch) != 3 or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in patch):
        raise ValueError("MRI patch_size must contain three positive ZYX integers")
    return config


@torch.no_grad()
def _infer_case(case, folder, model, geometry, config, device, seed, steps, samples):
    scale = config["data"]["scale"]
    patch = tuple(config.get("external", {}).get("eval_patch_zyx", config["data"]["patch_size"]))
    guidance = sampling_geometry_guidance(config)
    momentum_guidance = sampling_momentum_guidance(config)
    sigma = sampling_noise(config)
    improved = guidance is not None or config.get("inference_variant") == "endpoint_safe" or config["mode"] == "momentum"
    shape = (len(case.lq), case.lr_shape[0] * scale, case.lr_shape[1] * scale)
    if config.get("external"):
        patch = (1, min(patch[1], shape[1]), min(patch[2], shape[2]))
    folder.mkdir()
    mean = np.lib.format.open_memmap(folder / "mean.npy", mode="w+", dtype="float32", shape=shape)
    variance = np.lib.format.open_memmap(folder / "variance.npy", mode="w+", dtype="float32", shape=shape)
    mean[:] = 0
    variance[:] = 0
    axes = [np.hanning(n + 2)[1:-1].astype("float32") for n in patch]
    window = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    native_input = config["mode"] in {"haar_direct", "haar_bridge", "external_2d"}
    def spatial_starts(length, size):
        unit = scale if native_input else 1
        return [i * unit for i in tile_starts(length // unit, size // unit, max(1, size // unit // 2))]
    y_starts = spatial_starts(shape[1], patch[1])
    x_starts = spatial_starts(shape[2], patch[2])

    @lru_cache(maxsize=max(1, patch[0] * 2))
    def reader(index):
        return _read_lq(case, index)

    tile_batch = config.get("sampling", {}).get("tile_batch", 1) if config["mode"] in {"haar_direct", "haar_bridge"} else 1
    if type(tile_batch) is not int or tile_batch < 1:
        raise ValueError("tile_batch must be a positive integer")

    def predict_tiles(run, starts, generator):
        while group := list(itertools.islice(starts, tile_batch)):
            inputs = [make_mri_tile(case, run, start, patch, scale, reader=reader,
                                    with_native_lr=native_input) for start in group]
            bases, masks, *native = [torch.cat(tensors).to(device) for tensors in zip(*inputs, strict=True)]
            if config.get("predictor") == "bilinear":
                predictions = bases
            else:
                options = {}
                if native:
                    options["native_lr"] = native[0]
                if momentum_guidance is not None:
                    options["momentum_guidance"] = momentum_guidance
                if len(group) > 1 and config["mode"] == "haar_bridge" and steps > 1 and sigma:
                    # Draw in the historical tile-major order, with identical tensor sizes.
                    # CUDA Philox offsets depend on the shape of each individual draw.
                    noise_shape = (1, 16, *native[0].shape[2:])
                    by_tile = [[torch.randn(noise_shape, dtype=bases.dtype, device=device, generator=generator)
                                for _ in range(steps - 1)] for _ in group]
                    options["noise_steps"] = [torch.cat(step) for step in zip(*by_tile, strict=True)]
                predictions = reconstruct(model, bases, masks, config["mode"], geometry=geometry,
                                          steps=steps, sigma=sigma, generator=generator, terminal_noise=not improved,
                                          geometry_guidance=guidance, **options)
            if not improved:
                predictions = predictions.clamp(0, 1)
            yield from zip(group, predictions[:, 0].cpu().numpy(), strict=True)

    try:
        for sample in range(samples):
            accumulated = np.memmap(folder / "accumulated.tmp", mode="w+", dtype="float32", shape=shape)
            weights = np.memmap(folder / "weights.tmp", mode="w+", dtype="float32", shape=shape)
            try:
                accumulated[:] = 0
                weights[:] = 0
                generator = torch.Generator(device=device).manual_seed(seed + sample)
                for run in case.runs:
                    run_length = run[1] - run[0]
                    z_starts = tile_starts(run_length, patch[0], max(1, patch[0] // 2))
                    starts = itertools.product(z_starts, y_starts, x_starts)
                    for start, array in predict_tiles(run, starts, generator):
                        valid = tuple(min(p, length - offset) for p, length, offset in zip(patch, (run_length, *shape[1:]), start))
                        global_start = (run[0] + start[0], start[1], start[2])
                        region = tuple(slice(offset, offset + size) for offset, size in zip(global_start, valid))
                        crop = tuple(slice(0, size) for size in valid)
                        accumulated[region] += array[crop] * window[crop]
                        weights[region] += window[crop]
                for z in range(shape[0]):
                    if np.any(weights[z] <= 0):
                        raise RuntimeError(f"Uncovered output pixels in {case.id}, slice {z}")
                    value = accumulated[z] / weights[z]
                    if not np.isfinite(value).all():
                        raise FloatingPointError(f"Nonfinite prediction in {case.id}, slice {z}")
                    if improved:
                        np.clip(value, 0, 1, out=value)
                    delta = value - mean[z]
                    mean[z] += delta / (sample + 1)
                    variance[z] += delta * (value - mean[z])
            finally:
                accumulated._mmap.close()
                weights._mmap.close()
            print(f"{case.id}: completed sample {sample + 1}/{samples}, {shape[0]} slices", flush=True)
        for z in range(shape[0]):
            variance[z] = np.maximum(variance[z] / max(samples - 1, 1), 0)
        mean.flush()
        variance.flush()
        metrics = case_metrics(mean, case) if case.hq is not None else None
    finally:
        reader.cache_clear()
        mean._mmap.close()
        variance._mmap.close()
        (folder / "accumulated.tmp").unlink(missing_ok=True)
        (folder / "weights.tmp").unlink(missing_ok=True)
    return shape, metrics


def _write_case(case, staged, model, geometry, config, device, seed, steps, samples,
                checkpoint_hash, root_lq, root_hq, save_volume=True):
    folder = staged / case.id
    started = perf_counter()
    # Each case resets its generators to seed + sample. Keeping this original
    # convention makes stochastic predictions independent of rank assignment.
    started = time.perf_counter()
    shape, metrics = _infer_case(case, folder, model, geometry, config, device, seed, steps, samples)
    elapsed = time.perf_counter() - started
    variant = config.get("inference_variant", "legacy")
    momentum = config["mode"] == "momentum"
    guidance = sampling_geometry_guidance(config)
    improved = guidance is not None or variant == "endpoint_safe" or momentum
    metadata = {
        "case_id": case.id, "checkpoint_sha256": checkpoint_hash, "evaluation_seconds": elapsed,
        "root_lq": str(Path(root_lq).resolve()),
        "root_hq": str(Path(root_hq).resolve()) if root_hq else None,
        "shape_zyx": list(shape), "scale_xy": config["data"]["scale"], "scale_z": 1,
        "stems": list(case.stems), "original_indices": [int(index) for index in case.indices],
        "index_source": (
            "original sequence positions preserved by quarantine sidecar" if case.metadata.get("sidecar")
            else "filename numeric offsets when unambiguous, otherwise natural-order positions"
        ),
        "input_fingerprint": case.fingerprint,
        "contiguous_runs_output_z": [[int(start), int(stop)] for start, stop in case.runs],
        "normalization_scale": case.normalization_scale,
        "normalization_offset": case.normalization_offset, "channel_policy": case.channel_policy,
        "mode": config["mode"], "seed": seed, "steps": steps, "samples": samples,
        "sampling_sigma": sampling_noise(config), "reference_sigma": config["sampling"]["sigma"],
        "inference_and_metrics_seconds": perf_counter() - started,
        "base": "real LR, per-slice bilinear XY, align_corners=False", "observed_mask": "all_zero",
        "volume_exported": save_volume,
        "mean_file": "mean.npy" if save_volume else None,
        "variance_file": "variance.npy" if save_volume else None,
        "variance_estimator": "unbiased sample variance" if samples > 1 else "zero, one sample",
        "uncertainty_kind": "empirical variability of clipped patch reconstructions across random seeds",
        "uncertainty_calibration": "not established",
        "sampling_scope": "overlapping 3D patches with independent contiguous slice runs",
        "inference_variant": "momentum_bridge" if momentum else variant,
        "terminal_position_noise": not improved,
        "intensity_output": "overlap blended then clipped to [0,1]" if improved else "per-patch clipped to [0,1], overlap blended",
    }
    metadata.update(_gsbm_sampler_metadata(config, steps))
    if config.get("external"):
        metadata["external"] = config["external"]
        metadata.update(inference_variant=config["external"]["name"], terminal_position_noise=False)
    metadata["predictor"] = config.get("predictor", "model")
    if metadata["predictor"] == "bilinear":
        metadata.update(inference_variant="bilinear", terminal_position_noise=False)
    if guidance is not None:
        metadata["geometry_guidance"] = guidance
    momentum_guidance = sampling_momentum_guidance(config)
    if momentum_guidance is not None:
        metadata["momentum_geometry"] = momentum_guidance
        metadata["inference_variant"] = "mls_guided_momentum_bridge"
    if config.get("chart_momentum") is not None:
        metadata["chart_momentum"] = config["chart_momentum"]
    if metrics is not None:
        atomic_save(metrics, folder / "metrics.json", json_file=True)
    if not save_volume:
        if metrics is None:
            raise ValueError("HQ references are required before omitting reconstructed volumes")
        # Both files were created by this invocation inside its private staging
        # directory; all mmap handles closed after complete full-case evaluation.
        for filename in ("mean.npy", "variance.npy"):
            (folder / filename).unlink()
    # Metadata is the final completion record, after evaluation and export policy.
    atomic_save(metadata, folder / "metadata.json", json_file=True)


def _collect_summary(staged, cases, checkpoint_hash, exported_case_ids=None):
    """Check complete outputs and aggregate in canonical order, regardless of rank."""
    expected = {case.id for case in cases}
    exported_case_ids = expected if exported_case_ids is None else exported_case_ids
    if {path.name for path in staged.iterdir()} != expected:
        raise RuntimeError("MRI result directories do not match the requested cases")
    case_summaries = []
    for case in cases:
        folder = staged / case.id
        shape = (len(case.lq), case.lr_shape[0] * case.scale, case.lr_shape[1] * case.scale)
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if (metadata["case_id"] != case.id or metadata["checkpoint_sha256"] != checkpoint_hash
                or metadata["input_fingerprint"] != case.fingerprint
                or metadata["shape_zyx"] != list(shape)
                or metadata["stems"] != list(case.stems)
                or metadata["original_indices"] != list(case.indices)):
            raise RuntimeError(f"Incomplete or mismatched MRI result metadata: {case.id}")
        exported = case.id in exported_case_ids
        if (metadata.get("volume_exported") is not exported
                or metadata["mean_file"] != ("mean.npy" if exported else None)
                or metadata["variance_file"] != ("variance.npy" if exported else None)):
            raise RuntimeError(f"MRI volume export policy mismatch: {case.id}")
        for filename in ("mean.npy", "variance.npy"):
            if exported:
                array = np.load(folder / filename, mmap_mode="r", allow_pickle=False)
                try:
                    if array.shape != shape or array.dtype != np.dtype("float32"):
                        raise RuntimeError(f"Incomplete MRI result array: {case.id}/{filename}")
                finally:
                    array._mmap.close()
            elif case.hq is None or (folder / filename).exists():
                raise RuntimeError(f"Invalid omitted MRI result volume: {case.id}/{filename}")
        metrics = None
        if case.hq is not None:
            metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            if (metrics["case_id"] != case.id or metrics["evaluated_slices"] != shape[0]
                    or metrics["pixel_count"] != math.prod(shape)
                    or len(metrics["slices"]) != shape[0]
                    or [row["stem"] for row in metrics["slices"]] != list(case.stems)
                    or [row["original_index"] for row in metrics["slices"]] != list(case.indices)):
                raise RuntimeError(f"Incomplete MRI result metrics: {case.id}")
        case_summaries.append({
            "case_id": case.id, "slices": shape[0],
            "input_fingerprint": case.fingerprint, "evaluation_seconds": metadata["evaluation_seconds"],
            "metrics": {key: value for key, value in metrics.items() if key != "slices"} if metrics else None,
        })
    summary = {"cases": case_summaries, "total_slices": sum(case["slices"] for case in case_summaries)}
    available_metrics = [case["metrics"] for case in case_summaries if case["metrics"] is not None]
    if available_metrics:
        pixels = sum(value["pixel_count"] for value in available_metrics)
        mse = sum(value["squared_error_sum"] for value in available_metrics) / pixels
        ssim_count = sum(value["ssim_slice_count"] for value in available_metrics)
        ssim_sum = sum((value["mean_slice_ssim"] or 0) * value["ssim_slice_count"] for value in available_metrics)
        summary.update(all_mse=mse, all_psnr_db=_psnr(mse), perfect_match=mse == 0,
                       pixel_count=pixels, ssim_slice_count=ssim_count,
                       mean_slice_ssim=ssim_sum / ssim_count if ssim_count else None)
        regional = {key: sum(row[key] for row in available_metrics) for key in _region_totals()}
        summary.update(regional, **_regional_metrics(regional), metric_protocol=METRIC_PROTOCOL)
    return summary


def write_metrics_csv(staged, summary):
    """Write spreadsheet-readable views of the same verified statistics."""
    if "all_mse" not in summary:
        return
    cases = [{"evaluation_seconds": case["evaluation_seconds"], **case["metrics"]} for case in summary["cases"]]
    row = {k: v for k, v in summary.items() if v is None or isinstance(v, (str, float, int))}
    row["case_count"] = len(cases)
    for name, rows in (("per_case.csv", cases), ("summary.csv", [row])):
        with (staged / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def select_case_subset(cases, fraction=0.15, seed=20261110):
    """Stable pseudo-random whole-case sample; independent of GPU rank and order."""
    if not 0 < fraction <= 1:
        raise ValueError("test_fraction must be in (0,1]")
    count = max(1, math.ceil(len(cases) * fraction))
    selected = sorted(cases, key=lambda c: hashlib.sha256(f"{seed}:{c.id}".encode()).hexdigest())[:count]
    identifiers = {c.id for c in selected}
    return [c for c in cases if c.id in identifiers]


def checked_test_subset(cases, fraction=0.15, seed=20261110, manifest_path=None):
    """Describe the selected whole cases and verify an optional locked cohort."""
    available_ids = [case.id for case in cases]
    selected = select_case_subset(cases, fraction, seed)
    subset = {"fraction": fraction, "seed": seed, "available_cases": len(available_ids),
              "selected_cases": len(selected), "case_ids": [case.id for case in selected],
              "available_ids_sha256": hashlib.sha256("\n".join(available_ids).encode()).hexdigest()}
    if manifest_path is not None:
        frozen = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if subset != frozen:
            raise ValueError("Indexed test cohort differs from the locked comparison manifest")
    return selected, subset


def _checked_rank_call(context, function):
    """All ranks finish a local phase before any may enter the next collective."""
    result, error = None, None
    try:
        result = function()
    except Exception as exception:
        if not context.is_distributed:
            raise
        error = f"{type(exception).__name__}: {exception}"
    context.check_errors(error)
    return result


def _gsbm_sampler_metadata(config, steps):
    if config.get("predictor") == "bilinear":
        return {"drift_network_calls_per_tile_sample": 0}
    if config["mode"] == "external_2d":
        return {"network_calls_per_tile_sample": steps, "input_dimensionality": "2D"}
    if config["mode"] != "gsbm":
        return {}
    sampler = config["sampling"].get("drift_sampler", "forward")
    doubled = sampler == "diffusion_corrected" and sampling_noise(config) != config["sampling"]["sigma"]
    return {"drift_sampler": sampler, "drift_network_calls_per_tile_sample": steps * (1 + int(doubled))}


def infer_mri(
    checkpoint_path, root_lq, output_root, root_hq=None, device_name="auto", seed=42,
    steps=None, samples=1, case_ids=None, max_cases=None, scan_cache=None,
    channel_policy="identical_rgb_to_gray", save_volume_limit=None, test_fraction=None, subset_seed=20261110,
    inference_variant="legacy", subset_manifest=None, case_inventory=None, split=None, predictor="model",
    drift_sampler=None, inference_sigma=None, disable_guidance=False, readout=None,
):
    """Publish all cases atomically; torchrun assigns each case to exactly one rank.

    Ordinary rank failures are reported to every rank before staging is removed.
    A killed process can leave an unpublished staging directory and output lock;
    this implementation deliberately does not resume or reuse partial outputs.
    save_volume_limit retains arrays for only the first N selected, naturally
    sorted cases. All remaining cases still require complete HQ evaluation.
    """
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    if split not in {None, "val", "test"}:
        raise ValueError("MRI evaluation split must be val or test")
    if predictor not in {"model", "bilinear"}:
        raise ValueError("Unknown MRI predictor")
    if case_inventory is not None and split is None:
        split = "test"  # Legacy inventory callers always evaluated the test split.
    if test_fraction is None:
        test_fraction = 1.0 if split == "val" else 0.15
    if inference_variant not in {"legacy", "endpoint_safe"}:
        raise ValueError("Unknown inference variant")
    if steps is not None and (isinstance(steps, bool) or not isinstance(steps, int)
                              or steps < (0 if predictor == "bilinear" else 1)):
        raise ValueError("steps must be a positive integer")
    if max_cases is not None and (isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases < 1):
        raise ValueError("max_cases must be a positive integer")
    if save_volume_limit is not None and (
        isinstance(save_volume_limit, bool) or not isinstance(save_volume_limit, int) or save_volume_limit < 0
    ):
        raise ValueError("save_volume_limit must be a nonnegative integer or None")
    output_root = Path(output_root).expanduser().resolve()
    context = initialize_distributed(device_name)
    lock_path = output_root.parent / (output_root.name + ".inference.lock")
    lock, temporary = None, None

    def prepare_output():
        nonlocal lock, temporary
        if output_root.exists():
            raise FileExistsError(f"Choose a new MRI output directory: {output_root}")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        lock = lock_path.open("x")
        if output_root.exists():
            raise FileExistsError(f"MRI output appeared during initialization: {output_root}")
        temporary = tempfile.TemporaryDirectory(prefix="mri-infer-", dir=output_root.parent)
        staged = Path(temporary.name) / "results"
        staged.mkdir()
        return str(staged)

    def load_models():
        checkpoint = load_checkpoint(checkpoint_path)
        config = _validate_mri_checkpoint(checkpoint)
        sampling = dict(config["sampling"])
        if readout is not None:
            if config["mode"] != "haar_bridge" or readout not in {"integral", "endpoint"}:
                raise ValueError("Readout override requires a Haar residual bridge")
            if readout == "endpoint" and config["train"].get("endpoint_weight", 1.) == 0:
                raise ValueError("Endpoint readout requires endpoint supervision")
            sampling["readout"] = readout
        if drift_sampler is not None:
            sampling["drift_sampler"] = drift_sampler
        if disable_guidance:
            sampling["geometry_guidance"] = None
        if inference_sigma is not None:
            sampling["inference_sigma"] = inference_sigma
        sampler = sampling.get("drift_sampler", "forward")
        if sampler not in {"forward", "diffusion_corrected"}:
            raise ValueError("Unknown drift sampler")
        if (disable_guidance or drift_sampler is not None or sampler != "forward") and config["mode"] != "gsbm":
            raise ValueError("Drift sampler and online guidance overrides require GSBM")
        config = {**config, "sampling": sampling, "inference_variant": inference_variant, "predictor": predictor}
        if config["mode"] in {"haar_direct", "haar_bridge"}:
            config["inference_variant"] = "endpoint_safe"
        device = setup(seed, str(context.device), config["train"].get("threads", 4))
        if predictor == "bilinear":
            sampling.update(inference_sigma=0., geometry_guidance=None, momentum_geometry=None)
            config.update(inference_variant="endpoint_safe", chart_momentum=None)
            return config, device, None, None, checkpoint.get("data_identity")
        model, geometry = build_models({**checkpoint, "config": config}, device)
        if sampler == "diffusion_corrected":
            model = diffusion_adjusted_model(model, sampling["sigma"], sampling_noise(config))
        return config, device, model, geometry, checkpoint.get("data_identity")

    try:
        staged = Path(context.main_call(prepare_output, broadcast_result=True))
        config, device, model, geometry, fitted_identity = _checked_rank_call(context, load_models)
        data = config["data"]
        steps = config["sampling"]["steps"] if steps is None else steps
        if predictor == "bilinear":
            steps = 0

        def prepare_cases():
            if split is not None:
                roots = data[split]
                if (root_hq is None or Path(root_lq).expanduser().resolve() != project_path(roots["root_lq"])
                        or Path(root_hq).expanduser().resolve() != project_path(roots["root_hq"])
                        or channel_policy != data.get("channel_policy", "identical_rgb_to_gray")):
                    raise ValueError(f"MRI {split} evaluation requires its configured paired data roots and channel policy")
            if case_inventory is None:
                cases = index_cases(
                    root_lq, root_hq=root_hq, scale=data["scale"], cache_path=scan_cache,
                    channel_policy=channel_policy, normalization_scale=data.get("normalization_scale"),
                    normalization_offset=data.get("normalization_offset", 0.0), case_limit=max_cases,
                    scan_workers=data.get("scan_workers", 4), progress=True,
                )
            else:
                from .mri_inventory import load_inventory

                if split == "test" and (max_cases is not None or data.get("debug_case_limit") is not None):
                    raise ValueError("Prepared MRI test inventory requires the full paired test input contract")
                indexed, identity = load_inventory(case_inventory, data, (split,))
                if fitted_identity is not None and fitted_identity != identity:
                    raise ValueError("Prepared MRI case fingerprints differ from the fitted checkpoint")
                all_ids = [row["id"].casefold() for row in identity]
                if len(all_ids) != len(set(all_ids)):
                    raise ValueError("Prepared MRI inventory contains duplicate case IDs across splits")
                cases = indexed[split][:max_cases]
            if case_ids:
                requested = set(case_ids)
                found = {case.id for case in cases}
                if requested - found:
                    raise ValueError(f"Unknown or excluded MRI case IDs: {sorted(requested - found)}")
                cases = [case for case in cases if case.id in requested]
            if not cases:
                raise ValueError("No nonempty MRI cases to reconstruct")
            if len({case.id.casefold() for case in cases}) != len(cases):
                raise ValueError("Case identifiers must be unique")
            if save_volume_limit is not None and any(case.hq is None for case in cases[save_volume_limit:]):
                raise ValueError("root_hq is required to omit reconstructed MRI volumes")
            for case in cases:
                if case.id in (".", "..") or Path(case.id).name != case.id or "/" in case.id or "\\" in case.id:
                    raise ValueError("Case identifiers must be simple directory names")
            cases, subset = checked_test_subset(cases, test_fraction, subset_seed, subset_manifest)
            atomic_save(subset, output_root.parent / (output_root.name + "_subset.json"), json_file=True)
            return cases, fingerprint(checkpoint_path), subset

        cases, checkpoint_hash, subset = context.main_call(prepare_cases, broadcast_result=True)
        exported_case_ids = {case.id for case in cases[:save_volume_limit]}

        def reconstruct_cases():
            for case in cases[context.rank::context.world_size]:
                _write_case(case, staged, model, geometry, config, device, seed, steps, samples,
                            checkpoint_hash, root_lq, root_hq, save_volume=case.id in exported_case_ids)

        # No collectives occur inside a case; a failed worker still meets every
        # other worker here instead of skipping a barrier and stranding them.
        _checked_rank_call(context, reconstruct_cases)

        def publish():
            summary = _collect_summary(staged, cases, checkpoint_hash, exported_case_ids)
            summary["subset"] = subset
            summary.update(split=split or "custom", checkpoint_sha256=checkpoint_hash,
                           predictor=predictor, steps=steps, samples=samples, seed=seed)
            summary["inference_variant"] = "momentum_bridge" if config["mode"] == "momentum" else config["inference_variant"]
            if config["mode"] in {"haar_direct", "haar_bridge"} and predictor == "model":
                summary["readout"] = config["sampling"].get("readout", "integral")
                summary["network_calls_per_tile_sample"] = steps if config["mode"] == "haar_bridge" else 1
            summary["sampling_sigma"] = sampling_noise(config)
            summary.update(_gsbm_sampler_metadata(config, steps))
            summary["reference_sigma"] = config["sampling"]["sigma"]
            if config.get("chart_momentum") is not None:
                summary["inference_variant"] = "chart_trivialized_momentum_bridge"
                summary["chart_momentum"] = config["chart_momentum"]
            guidance = sampling_geometry_guidance(config)
            if guidance is not None:
                summary["geometry_guidance"] = guidance
            momentum_guidance = sampling_momentum_guidance(config)
            if momentum_guidance is not None:
                summary["inference_variant"] = "mls_guided_momentum_bridge"
                summary["momentum_geometry"] = momentum_guidance
            if predictor == "bilinear":
                summary["inference_variant"] = "bilinear"
            elif config.get("external"):
                summary.update(inference_variant=config["external"]["name"], external=config["external"])
            atomic_save(subset, staged / "subset_manifest.json", json_file=True)
            atomic_save(summary, staged / "summary.json", json_file=True)
            write_metrics_csv(staged, summary)
            if output_root.exists():
                raise FileExistsError(f"MRI output appeared before publication: {output_root}")
            os.rename(staged, output_root)

        context.main_call(publish)
    finally:
        try:
            if context.is_main:
                try:
                    if temporary is not None:
                        temporary.cleanup()
                finally:
                    if lock is not None:
                        lock.close()
                        lock_path.unlink(missing_ok=True)
        finally:
            context.cleanup()
    return output_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root-lq", required=True, help="LR root with case/slice.png layout")
    parser.add_argument("--root-hq", help="Optional matching HQ root, for full-case image metrics")
    parser.add_argument("--output-root", required=True, help="New directory for per-case volumes and metadata")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--case", action="append", dest="case_ids", help="Case ID, repeat to select multiple")
    parser.add_argument("--max-cases", type=int, help="Limit sorted cases for a smoke test")
    parser.add_argument("--save-volume-limit", type=int, help=(
        "Keep mean/variance volumes only for the first N naturally sorted selected cases; "
        "0 keeps only metrics/metadata. All cases are evaluated; omitting volumes requires HQ."
    ))
    parser.add_argument("--split", choices=["val", "test"], help="Validate paired roots against this checkpoint split")
    parser.add_argument("--predictor", choices=["model", "bilinear"], default="model",
                        help="Use the checkpoint protocol and the selected reconstruction method")
    parser.add_argument("--fraction", "--test-fraction", type=float, dest="test_fraction",
                        help="Whole-case sampling fraction (VAL 1, TEST 0.15 by default)")
    parser.add_argument("--subset-seed", type=int, default=20261110)
    parser.add_argument("--subset-manifest")
    parser.add_argument("--inference-variant", choices=["legacy", "endpoint_safe"], default="legacy")
    parser.add_argument("--drift-sampler", choices=["forward", "diffusion_corrected"])
    parser.add_argument("--inference-sigma", type=float)
    parser.add_argument("--disable-guidance", action="store_true")
    parser.add_argument("--readout", choices=["integral", "endpoint"])
    parser.add_argument("--scan-cache")
    parser.add_argument("--case-inventory")
    parser.add_argument("--channel-policy", default="identical_rgb_to_gray")
    args = parser.parse_args()
    output = infer_mri(
        args.checkpoint, args.root_lq, args.output_root, root_hq=args.root_hq, device_name=args.device,
        seed=args.seed, steps=args.steps, samples=args.samples, case_ids=args.case_ids,
        max_cases=args.max_cases, scan_cache=args.scan_cache, channel_policy=args.channel_policy,
        save_volume_limit=args.save_volume_limit, test_fraction=args.test_fraction, subset_seed=args.subset_seed,
        inference_variant=args.inference_variant, subset_manifest=args.subset_manifest,
        drift_sampler=args.drift_sampler, inference_sigma=args.inference_sigma, disable_guidance=args.disable_guidance,
        case_inventory=args.case_inventory, split=args.split, predictor=args.predictor, readout=args.readout,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps({"output_root": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
