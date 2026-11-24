"""Streaming TRAIN-only population statistics for all 16 HaarXY residual bands."""
import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from .mri_data import read_plane
from .momentum.model import HaarXY
from .runtime import atomic_save, load_checkpoint, project_path, read_config


def statistics_identity(cases):
    return {"version": 1, "split": "train", "codec": "HaarXY-orthonormal-4x4-v1",
            "baseline": "bilinear-align_corners_false-full-plane", "eps": 1.e-6,
            "cases": [{"id": c.id, "fingerprint": c.fingerprint,
                       "scale": c.scale, "slices": len(c.lq), "shape": list(c.lr_shape),
                       "normalization": [c.normalization_scale, c.normalization_offset, c.channel_policy, c.bits],
                       "roots": [str(c.lq[0].parent.resolve()), str(c.hq[0].parent.resolve())]}
                      for c in cases]}


@torch.no_grad()
def compute_haar_statistics(cases, *, device="cpu", progress=True):
    """Process every retained TRAIN slice once, including short Z runs/background."""
    codec = HaarXY().to(device)
    mean = torch.zeros(16, dtype=torch.float64, device=device)
    m2, count, slices = torch.zeros_like(mean), 0, 0
    for i, case in enumerate(cases):
        if case.hq is None or case.scale != 4:
            raise ValueError("Whitening requires paired TRAIN XYx4 cases")
        for low_path, high_path in zip(case.lq, case.hq, strict=True):
            def plane(path):
                return torch.from_numpy(read_plane(path, case.normalization_scale,
                                        case.normalization_offset, case.channel_policy, case.bits)).to(device)
            low, high = plane(low_path), plane(high_path)
            if low.shape != case.lr_shape or high.shape != tuple(4*s for s in case.lr_shape):
                raise ValueError("Whitening input shape changed after indexing")
            base = F.interpolate(low[None,None], scale_factor=4, mode="bilinear", align_corners=False)
            coefficients = codec.encode((high[None,None]-base).unsqueeze(2))
            values = coefficients.movedim(1,0).reshape(16,-1).double()
            if not values.isfinite().all():
                raise ValueError("Nonfinite TRAIN residual")
            n = values.shape[1]
            local_mean = values.mean(1)
            delta = local_mean - mean
            m2 += ((values-local_mean[:,None])**2).sum(1) + delta.square()*(count*n/(count+n))
            mean += delta*(n/(count+n))
            count += n
            slices += 1
        if progress and (i == 0 or (i+1) % 50 == 0 or i+1 == len(cases)):
            print(f"Haar statistics: {i+1}/{len(cases)} TRAIN cases, {slices} slices", flush=True)
    if count == 0:
        raise ValueError("No TRAIN coefficients to whiten")
    return dict(mean=mean.cpu().float(), std=(m2/count).clamp_min(0).sqrt().cpu().float(),
                coefficient_count_per_band=count, slice_count=slices, case_count=len(cases))


def ensure_haar_statistics(cases, path, *, split="train", device="cpu"):
    from .mri_case_cache import acquire_lock
    if split != "train":
        raise ValueError("Whitening may only be fitted on TRAIN")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = statistics_identity(cases)
    with acquire_lock(path.with_suffix(path.suffix+".lock")):
        if path.is_file():
            payload = load_checkpoint(path)
            if payload.get("identity") != identity:
                raise ValueError("Haar statistics cache has a different TRAIN identity; use a new path")
        else:
            payload = {"identity": identity, **compute_haar_statistics(cases, device=device)}
            atomic_save(payload, path)
    return payload


def main():
    from .datasets import training_data
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--case-inventory")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = read_config(args.config)
    torch.set_num_threads(cfg["train"].get("threads",4))
    train, _, _, _ = training_data(cfg, args.case_inventory)
    result = ensure_haar_statistics(train.all_cases, project_path(cfg["band_stats_cache"]), device=args.device)
    print({k:v for k,v in result.items() if k != "identity"}, flush=True)


if __name__ == "__main__":
    main()
