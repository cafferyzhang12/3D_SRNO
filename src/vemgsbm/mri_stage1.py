"""Validation-only ablations on the epoch-38 LR-routed MRI bridge."""

import argparse
from collections import Counter
from functools import lru_cache
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import sobel
import torch
from torch.nn import functional as F

from .atlas_geometry import BoundSourceMRIAtlas, MRIAtlasGeometry, source_chart_distribution
from .infer import tile_starts
from .momentum.model import HaarXY
from .mri_data import read_plane
from .mri_infer import _read_lq, infer_mri, make_mri_tile
from .mri_inventory import load_inventory
from .runtime import atomic_save, fingerprint, load_checkpoint, project_path, sampling_geometry_guidance, setup


# Sampler, noise, steps. Each corrected step uses both direction embeddings.
ARMS = {
    "plain": ("forward", .05, 20),
    "guided": ("forward", .05, 20),
    "corrected": ("diffusion_corrected", .05, 20),
    "equal_calls": ("diffusion_corrected", .05, 10),
    "reference": ("forward", .10, 20),
}


def _totals(error, change):
    return Counter(count=error.size, error_sq=float(np.square(error).sum()),
                   dot=float((error * change).sum()), change_sq=float(np.square(change).sum()))


def correction_regions(hr, plain, guided, codec):
    """Post-Hann, clipped predictions, compared once per output plane."""
    hr = np.asarray(hr, dtype=np.float64)
    error = hr - np.asarray(plain, dtype=np.float64)
    change = np.asarray(guided, dtype=np.float64) - np.asarray(plain, dtype=np.float64)
    foreground = hr > float(np.float32(2 / 255))
    gradient = np.hypot(sobel(hr, axis=0, mode="reflect"), sobel(hr, axis=1, mode="reflect"))
    edge = np.zeros_like(foreground)
    if foreground.any():
        edge = foreground & (gradient > 0) & (gradient >= np.quantile(gradient[foreground], .9))
    def haar_high(array):
        return codec.encode(torch.from_numpy(array)[None, None, None])[:, 1:].numpy()
    return {"whole": _totals(error, change), "foreground": _totals(error[foreground], change[foreground]),
            "edges": _totals(error[edge], change[edge]), "haar_high": _totals(haar_high(error), haar_high(change))}


def correction_report(total):
    count, dot, change_sq = (total[key] for key in ("count", "dot", "change_sq"))
    return {**total, "delta_mse": (change_sq - 2 * dot) / count if count else None,
            "optimal_change_multiplier": dot / change_sq if change_sq else None}


@torch.no_grad()
def oracle_fit_scores(geometry, hr):
    """Fit each HR tile to every deployed bounded decoder; lower MSE is better."""
    features = geometry.encode(hr)
    fits = []
    for chunk in torch.arange(geometry.atlas.chart_count, device=hr.device).split(8):
        all_ids = chunk[None].expand(len(hr), -1)
        oracle = BoundSourceMRIAtlas(geometry, all_ids, torch.ones_like(all_ids, dtype=hr.dtype),
                                     torch.ones(len(hr), dtype=hr.dtype, device=hr.device))
        fits.append(oracle._local_geometry(features, derivative=False)["residual"])
    return torch.cat(fits, dim=1)


@torch.no_grad()
def chart_fit(geometry, base, hr, output):
    """Compare LR-only selection and reconstructed endpoint scores against an HR oracle."""
    source = (geometry.encoder(base) - geometry.source_mean) / geometry.source_scale
    distribution, confidence = source_chart_distribution(
        source, geometry.source_bank, geometry.chart_probabilities, geometry.routing_settings)
    top_mass, ids = distribution.topk(min(3, distribution.shape[1]), dim=1)
    prior = top_mass / top_mass.sum(1, keepdim=True)
    bound = BoundSourceMRIAtlas(geometry, ids, prior, confidence)
    fits = oracle_fit_scores(geometry, hr)
    best = fits.amin(1)
    selected = fits.gather(1, ids)
    endpoint_weights = bound._local_geometry(geometry.encode(output), derivative=False)["weight"]
    return {
        "tiles": len(base), "oracle_recalled": int((selected.amin(1) == best).sum()),
        "top3_mass": float(top_mass.sum()), "hr_best_feature_mse": float(best.sum()),
        "lr_top1_gap": float((selected[:, 0] - best).sum()),
        "best_of_lr_top3_gap": float((selected.amin(1) - best).sum()),
        "lr_prior_weighted_gap": float(((selected - best[:, None]) * prior).sum()),
        "reconstructed_endpoint_weighted_gap": float(((selected - best[:, None]) * endpoint_weights).sum()),
    }


def _tile_from_planes(reader, run, start, patch):
    z, y, x = start
    dz, dy, dx = patch
    planes = [reader(min(run[0] + z + i, run[1] - 1))[y:y+dy, x:x+dx] for i in range(dz)]
    volumes = torch.from_numpy(np.stack(planes))[None, None]
    return F.pad(volumes, (0, dx - volumes.shape[-1], 0, dy - volumes.shape[-2], 0, 0), mode="replicate")


def _case_diagnostics(folder, case, config, geometry, device):
    plain = np.load(folder / "plain" / case.id / "mean.npy", mmap_mode="r", allow_pickle=False)
    guided = np.load(folder / "guided" / case.id / "mean.npy", mmap_mode="r", allow_pickle=False)
    patch = config["data"]["patch_size"]
    shape = (len(case.hq), *(v * case.scale for v in case.lr_shape))
    if guided.shape != plain.shape or plain.shape != shape:
        raise ValueError("Guidance diagnostic requires matching full-case volumes")
    @lru_cache(maxsize=2 * patch[0])
    def hr_plane(index):
        return read_plane(case.hq[index], normalization_scale=case.normalization_scale,
                          normalization_offset=case.normalization_offset, channel_policy=case.channel_policy,
                          expected_bits=case.bits)
    @lru_cache(maxsize=2 * patch[0])
    def lr_plane(index):
        return _read_lq(case, index)
    regions, chart = {}, Counter()
    codec = HaarXY().double()
    for z in range(shape[0]):
        for name, values in correction_regions(hr_plane(z), plain[z], guided[z], codec).items():
            regions.setdefault(name, Counter()).update(values)
    ys, xs = (tile_starts(size, width, max(1, width // 2)) for size, width in zip(shape[1:], patch[1:]))
    for run in case.runs:
        zs = tile_starts(run[1] - run[0], patch[0], max(1, patch[0] // 2))
        for start in itertools.product(zs, ys, xs):
            base, _ = make_mri_tile(case, run, start, patch, case.scale, reader=lr_plane)
            hr = _tile_from_planes(hr_plane, run, start, patch)
            output = _tile_from_planes(guided.__getitem__, run, start, patch)
            chart.update(chart_fit(geometry, base.to(device), hr.to(device), output.to(device)))
    return regions, chart


def _check_arm(output, cases, arm, digest, seed):
    summary = json.loads((output / "summary.json").read_text())
    sampler, sigma, steps = ARMS[arm]
    if summary["subset"]["case_ids"] != [case.id for case in cases] or summary.get("split") != "val":
        raise ValueError(f"Inconsistent validation cohort in {output}")
    for case in cases:
        case_dir = output / case.id
        meta = json.loads((case_dir / "metadata.json").read_text())
        expected = {"checkpoint_sha256": digest, "input_fingerprint": case.fingerprint, "seed": seed,
                    "samples": 1, "steps": steps, "sampling_sigma": sigma, "reference_sigma": .10,
                    "drift_sampler": sampler, "inference_variant": "endpoint_safe", "terminal_position_noise": False}
        if (any(meta.get(key) != value for key, value in expected.items())
                or bool(meta.get("geometry_guidance")) != (arm == "guided")):
            raise ValueError(f"Mismatched validation artifact in {case_dir}")
        if arm in ("guided", "plain") and not (case_dir / "mean.npy").is_file():
            raise ValueError(f"Missing paired output for {case.id}")


def _chart_means(counts):
    return {"tiles": counts["tiles"], **{k: v / counts["tiles"] for k, v in counts.items() if k != "tiles"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=[*ARMS, "diagnostics"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--case-inventory", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--case", dest="case_ids", action="append")
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    if (config["mode"] != "gsbm" or checkpoint["epoch"] != 38 or config["sampling"]["sigma"] != .1
            or config["data"]["scale"] != 4 or sampling_geometry_guidance(config) is None
            or checkpoint["metric"]["routing"]["settings"]["candidates"] != 3):
        raise ValueError("Expected the epoch-38 guided XYx4 checkpoint with three routed charts and sigma=.10")
    digest = fingerprint(args.checkpoint)
    data = config["data"]
    indexed, _ = load_inventory(args.case_inventory, data, ("val",))
    cases = indexed["val"]
    if args.case_ids:
        cases = [c for c in cases if c.id in args.case_ids]
        if {c.id for c in cases} != set(args.case_ids):
            raise ValueError("Requested case missing from VAL inventory")
    if not cases:
        raise ValueError("Empty VAL cohort")
    output = args.output_root / args.arm
    if args.arm != "diagnostics":
        del checkpoint
        if not (output / "summary.json").is_file():
            sampler, eta, steps = ARMS[args.arm]
            infer_mri(args.checkpoint, project_path(data["val"]["root_lq"]), output,
                      root_hq=project_path(data["val"]["root_hq"]), device_name=args.device, seed=args.seed,
                      steps=steps, samples=1, split="val", case_ids=args.case_ids, case_inventory=args.case_inventory,
                      channel_policy=data.get("channel_policy", "identical_rgb_to_gray"),
                      save_volume_limit=None if args.arm in ("plain", "guided") else 0,
                      inference_variant="endpoint_safe", drift_sampler=sampler,
                      inference_sigma=eta, disable_guidance=args.arm != "guided")
        _check_arm(output, cases, args.arm, digest, args.seed)
        return
    for arm in ("plain", "guided"):
        _check_arm(args.output_root / arm, cases, arm, digest, args.seed)
    device = setup(args.seed, args.device, config["train"].get("threads", 4))
    geometry = MRIAtlasGeometry(checkpoint["metric"]).to(device)
    pooled, charts, rows = {}, Counter(), []
    for case in cases:
        regions, chart = _case_diagnostics(args.output_root, case, config, geometry, device)
        for name, values in regions.items():
            pooled.setdefault(name, Counter()).update(values)
        charts.update(chart)
        rows.append({"case_id": case.id, "chart": _chart_means(chart),
                     "correction": {k: correction_report(v) for k, v in regions.items()}})
        print(f"Diagnosed VAL case {case.id}", flush=True)
    atomic_save({"split": "val", "checkpoint_sha256": digest, "seed": args.seed, "cases": rows,
                 "regions": {"foreground": "HR > 2/255", "edges": "Nonzero XY Sobel magnitude >= foreground slice p90",
                             "haar_high": "15 non-DC Haar coefficients on complete 4x4 blocks"},
                 "fit_units": "Mean squared standardized feature residual using deployed bounded MLS decoder",
                 "chart": _chart_means(charts),
                 "correction": {k: correction_report(v) for k, v in pooled.items()}},
                output / "summary.json", json_file=True)


if __name__ == "__main__":
    main()
