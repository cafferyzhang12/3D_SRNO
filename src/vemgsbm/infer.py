"""Overlap-blended 3D inference on a normalized, physically sparse ZYX volume."""
import argparse
from contextlib import ExitStack
import itertools
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn.functional as functional

from .bridge import reconstruct
from .data import load_volume
from .runtime import atomic_save, build_models, fingerprint, load_checkpoint, setup, sampling_noise


def tile_starts(length, patch, stride, alignment=1):
    if length <= patch:
        return [0]
    stride = max(alignment, stride // alignment * alignment)
    starts = list(range(0, length - patch + 1, stride))
    last = length - patch
    if last % alignment:
        raise ValueError("Z tile endpoint must lie on the measured slice grid")
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_tile(lr, start, patch, scale):
    z, y, x = start
    dz, dy, dx = patch
    coarse = np.array(lr[z // scale:(z + dz - 1) // scale + 1, y:y + dy, x:x + dx], copy=True)
    if not np.isfinite(coarse).all() or coarse.min() < 0 or coarse.max() > 1:
        raise ValueError("Input must be normalized finite float32 in [0,1]. Run vemgsbm.prepare first.")
    tensor = torch.from_numpy(coarse).float()[None, None]
    needed_z = (dz - 1) // scale + 1
    pad = (0, dx - tensor.shape[-1], 0, dy - tensor.shape[-2], 0, needed_z - tensor.shape[-3])
    tensor = functional.pad(tensor, pad, mode="replicate")
    base = functional.interpolate(tensor, size=patch, mode="trilinear", align_corners=True)
    mask = torch.zeros_like(base)
    mask[:, :, ::scale] = 1
    base[:, :, ::scale] = tensor
    return base, mask


@torch.no_grad()
def infer_volume(checkpoint_path, input_path, output_path, device_name="auto", seed=42, steps=None, samples=1):
    output_path = Path(output_path).resolve()
    if output_path.suffix.lower() != ".npy":
        raise ValueError("Output must use the .npy extension")
    if samples < 1:
        raise ValueError("samples must be positive")
    if steps is not None and steps < 1:
        raise ValueError("steps must be positive")
    variance_path = output_path.with_name(output_path.stem + "_variance.npy")
    metadata_path = output_path.with_suffix(".json")
    if any(p.exists() for p in (output_path, variance_path, metadata_path)):
        raise FileExistsError("An output already exists. Choose a new output path.")
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint["stage"] != "reconstruction":
        raise ValueError("Inference requires a reconstruction checkpoint")
    cfg = checkpoint["config"]
    if cfg["data"].get("kind") == "paired_mri_png" or cfg["data"].get("degradation") != "slice_select":
        raise ValueError("This entry restores sparse Z slices. Use vemgsbm.mri_infer for paired MRI XY checkpoints.")
    device = setup(seed, device_name, cfg["train"].get("threads", 4))
    model, geometry = build_models(checkpoint, device)
    lr = load_volume(input_path)
    if lr.shape[0] < 2:
        raise ValueError("At least two measured slices are required")
    scale = cfg["data"]["scale"]
    shape = ((lr.shape[0] - 1) * scale + 1, lr.shape[1], lr.shape[2])
    patch = tuple(cfg["data"]["patch_size"])
    z_starts = tile_starts(shape[0], patch[0], max(scale, (patch[0] - 1) // 2), scale)
    y_starts = tile_starts(shape[1], patch[1], max(1, patch[1] // 2))
    x_starts = tile_starts(shape[2], patch[2], max(1, patch[2] // 2))
    axes = [np.hanning(n + 2)[1:-1].astype("float32") for n in patch]
    window = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_suffix(".lock")
    lock_fd = lock_path.open("x")
    published = []
    try:
        with lock_fd, tempfile.TemporaryDirectory(prefix="vem-infer-", dir=output_path.parent) as temporary, ExitStack() as maps:
            if any(p.exists() for p in (output_path, variance_path, metadata_path)):
                raise FileExistsError("Output appeared while loading the model. Choose a new output path.")
            temporary = Path(temporary)
            mean = np.lib.format.open_memmap(temporary / "mean.npy", mode="w+", dtype="float32", shape=shape)
            maps.callback(mean._mmap.close)
            m2 = np.lib.format.open_memmap(temporary / "variance.npy", mode="w+", dtype="float32", shape=shape)
            maps.callback(m2._mmap.close)
            for sample in range(samples):
                acc = np.memmap(temporary / "acc.bin", mode="w+", dtype="float32", shape=shape)
                maps.callback(acc._mmap.close)
                weights = np.memmap(temporary / "weights.bin", mode="w+", dtype="float32", shape=shape)
                maps.callback(weights._mmap.close)
                generator = torch.Generator(device=device).manual_seed(seed + sample)
                for start in itertools.product(z_starts, y_starts, x_starts):
                    base, mask = make_tile(lr, start, patch, scale)
                    prediction = reconstruct(model, base.to(device), mask.to(device), cfg["mode"],
                                             geometry=geometry, steps=steps or cfg["sampling"]["steps"],
                                             sigma=sampling_noise(cfg), generator=generator)
                    array = prediction[0, 0].clamp(0, 1).cpu().numpy()
                    valid_shape = tuple(min(p, length - offset) for p, length, offset in zip(patch, shape, start))
                    region = tuple(slice(offset, offset + size) for offset, size in zip(start, valid_shape))
                    crop = tuple(slice(0, size) for size in valid_shape)
                    acc[region] += array[crop] * window[crop]
                    weights[region] += window[crop]
                # Slice-wise aggregation keeps full-volume arrays on disk.
                for z in range(shape[0]):
                    if np.any(weights[z] <= 0):
                        raise RuntimeError("Uncovered output voxel")
                    value = acc[z] / weights[z]
                    if z % scale == 0:
                        value = np.asarray(lr[z // scale])
                    if not np.isfinite(value).all():
                        raise FloatingPointError("Nonfinite prediction")
                    delta = value - mean[z]
                    mean[z] += delta / (sample + 1)
                    m2[z] += delta * (value - mean[z])
                acc._mmap.close()
                weights._mmap.close()
                del acc, weights
                print(f"Completed sample {sample + 1}/{samples}", flush=True)
            for z in range(shape[0]):
                m2[z] = np.maximum(m2[z] / max(samples - 1, 1), 0)
            mean.flush()
            m2.flush()
            mean._mmap.close()
            m2._mmap.close()
            del mean, m2
            os.replace(temporary / "mean.npy", output_path)
            published.append(output_path)
            if samples > 1:
                os.replace(temporary / "variance.npy", variance_path)
                published.append(variance_path)
        atomic_save({"checkpoint_sha256": fingerprint(checkpoint_path), "input": str(Path(input_path).resolve()),
                     "shape_zyx": shape, "scale_z": scale, "mode": cfg["mode"], "seed": seed,
                     "steps": steps or cfg["sampling"]["steps"], "samples": samples,
                     "output_kind": "sample_mean" if samples > 1 else "single_reconstruction",
                     "variance": str(variance_path) if samples > 1 else None,
                     "axial_extent": "between first and last measured slice, without extrapolation",
                     "sampling_scope": "overlapping patch bridges, not a global volume posterior"}, metadata_path, json_file=True)
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        lock_path.unlink(missing_ok=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Normalized sparse ZYX NPY")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    print(infer_volume(args.checkpoint, args.input, args.output, args.device, args.seed, args.steps, args.samples))


if __name__ == "__main__":
    main()
