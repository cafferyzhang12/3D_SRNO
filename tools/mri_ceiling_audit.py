"""VAL-only interpolation/noise audit; synthetic mode requires no dataset.

No TEST image is opened. Noise diagnostics estimate an iid-noise oracle under
explicit assumptions; a physical dataset ceiling needs repeat acquisitions.
Run full VAL/perceptual/old-model diagnostics inside an allocated Slurm job.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from vemgsbm.mri_metrics import mean_xy_ssim
from vemgsbm.mri_v2_losses import ReconstructionLoss, gradient_diagnostics, local_moments
from vemgsbm.mri_v2_diagnostics import (upsample_xy, local_variance_ratio,
    textured_region_ssim, noise_ceiling_estimate)


def synthetic_pair(size=64, depth=6, device="cpu", seed=42):
    generator = torch.Generator(device=device).manual_seed(seed)
    y, x = torch.meshgrid(torch.linspace(-1, 1, size, device=device),
                           torch.linspace(-1, 1, size, device=device), indexing="ij")
    slices = []
    for z in range(depth):
        tissue = ((x/.8)**2+(y/.9)**2 < 1).float()
        signal = .42+.19*torch.exp(-4*(x*x+y*y))+.10*torch.sin(18*x+.1*z)*torch.cos(16*y-.1*z)
        signal += .08*torch.exp(-90*((x-.26)**2+(y+.12)**2))
        slices.append((tissue*signal+.003*torch.randn(x.shape, generator=generator, device=device)).clamp(0, 1))
    hr = torch.stack(slices)[None, None]
    lr = F.avg_pool2d(hr.reshape(depth, 1, size, size), 4, 4).reshape(1, 1, depth, size//4, size//4)
    return lr, hr


def _read_case(case, device):
    from vemgsbm.mri_data import read_plane
    def load(paths):
        return torch.from_numpy(np.stack([read_plane(p, case.normalization_scale,
            case.normalization_offset, case.channel_policy, case.bits) for p in paths]))[None, None].to(device)
    return load(case.lq), load(case.hq)


def _val_cases(inventory, config=None, max_cases=None):
    from vemgsbm.mri_inventory import load_inventory
    if config:
        import yaml
        cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    else:
        with gzip.open(inventory, "rt", encoding="utf-8") as stream:
            cfg = dict(data=json.load(stream)["data"])
    cases, _ = load_inventory(inventory, cfg["data"], ("val",))
    return cases["val"][:max_cases] if max_cases else cases["val"]


@torch.no_grad()
def perceptual_update(prediction, reference, extractors, moments, batch_size=8):
    from vemgsbm.mri_perceptual import rgb_inputs
    paired, features = extractors
    a = prediction[0, 0].clamp(0, 1).cpu().numpy()
    b = reference[0, 0].clamp(0, 1).cpu().numpy()
    total = 0.
    for start in range(0, len(a), batch_size):
        p_lp, p_fid = rgb_inputs(a[start:start+batch_size], prediction.device)
        r_lp, r_fid = rgb_inputs(b[start:start+batch_size], prediction.device)
        total += float(paired(p_lp, r_lp).double().sum())
        moments[0].update(features(p_fid)[0].cpu().numpy())
        moments[1].update(features(r_fid)[0].cpu().numpy())
    return total, len(a)


def _legacy_clipping(atlas, prediction, target, case_ids, base_coefficients):
    """Observe exact pre-clamp coordinates in the old TRAIN-only MLS path."""
    from vemgsbm.spatial_atlas import _sites
    clipped, samples = 0, 0
    with torch.no_grad():
        for batch, case in enumerate(case_ids):
            truth = _sites(target[batch:batch+1]).float()
            pred = _sites(prediction[batch:batch+1]).float()
            base = _sites(base_coefficients[batch:batch+1]).float()
            fold = atlas.folds[case]
            lo, hi = atlas.offsets[fold:fold+2]
            # Deterministic bounded probe; no mutation of production fitting or queries.
            ids = torch.linspace(0, len(truth)-1, min(len(truth), 256), device=truth.device).long()
            source = torch.cat((truth[ids], base[ids]), -1) if atlas.use_paired_base else truth[ids]
            source = (source-atlas.query_mean[fold])/atlas.query_std[fold]
            chart = torch.cdist(source, atlas.centers[lo:hi]).argmin(1)+lo
            physical = pred[ids]*atlas.physical_scale
            q = ((physical-atlas.mean[chart]).unsqueeze(-2)@atlas.basis[chart]).squeeze(-2)/atlas.bandwidth[chart]
            limit = float(atlas.settings.get("coordinate_limit", 3.))
            lower, upper = atlas.domain_lo[chart].clamp(-limit, limit), atlas.domain_hi[chart].clamp(-limit, limit)
            clipped += int(((q < lower) | (q > upper)).any(-1).sum())
            samples += len(ids)
    return dict(probed_sites=samples, hard_clipped_fraction=clipped/max(samples, 1),
                quadratic_supported_fraction=float(atlas.quadratic_supported.float().mean()))


def legacy_gradient_audit(checkpoint_path, inventory, atlas_bank, device, items=2):
    """Single TRAIN forward/backward diagnostic using unchanged legacy terms."""
    if not checkpoint_path:
        return dict(status="unavailable", reason="No legacy checkpoint supplied; synthetic gradients are reported separately")
    if not inventory:
        return dict(status="unavailable", reason="Legacy gradients require the matching TRAIN inventory")
    from vemgsbm.runtime import load_checkpoint, build_models
    from vemgsbm.mri_inventory import load_inventory
    from vemgsbm.mri_data import PairedMRIPatchDataset
    ck = load_checkpoint(checkpoint_path)
    cfg = ck["config"]
    if cfg.get("predict_target") != "x1" or cfg["train"].get("objective") != "sr_reconstruction":
        raise ValueError("Legacy gradient diagnostic requires the x1 SR reconstruction checkpoint")
    cases, _ = load_inventory(inventory, cfg["data"], ("train",))
    ds = PairedMRIPatchDataset(".", ".", "train", scale=4, patch_size=tuple(cfg["data"]["patch_size"]),
        samples_per_volume=1, seed=42, cases=cases["train"])
    batch = [ds[i] for i in range(min(items, len(ds)))]
    lr, base, hr = [torch.stack([item[name] for item in batch]).to(device) for name in ("lr", "base", "hr")]
    model, _ = build_models(ck, device)
    model.train()
    model.zero_grad(set_to_none=True)
    truth = model.normalize_haar(model.codec.encode(hr-base)).detach()
    prediction, _ = model(model.initial_state(lr, base), model.lr_encoder(lr), 0.)
    coefficients = model.denormalize_haar(prediction)
    error = base+model.codec.decode(coefficients)-hr
    scale = model.band_std+1.e-6 if model.use_band_whitening else torch.ones_like(model.band_std)
    normalization = scale.square().mean()
    coefficient_error = coefficients-model.denormalize_haar(truth)
    terms = dict(L_pixel=error.square().mean()/normalization,
        L_HaarHF=torch.cat((torch.zeros_like(coefficient_error[:, :1]), coefficient_error[:, 1:]), 1).square().mean()/normalization,
        L_edge=.5*((error[..., 1:, :]-error[..., :-1, :]).div(2).square().mean()+(error[..., 1:]-error[..., :-1]).div(2).square().mean())/normalization,
        L_SSIM=1-mean_xy_ssim((error+hr)[:max(1, len(lr)//8)].clamp(0, 1), hr[:max(1, len(lr)//8)]))
    weights = dict(L_pixel=1., L_HaarHF=cfg["train"].get("hf_weight", .1),
        L_edge=cfg["train"].get("edge_weight", .05), L_SSIM=cfg["train"].get("ssim_weight", 0.),
        L_MLS=cfg.get("mls_reg_weight", 0.))
    clipping = dict(status="unavailable", reason="No matching TRAIN atlas bank supplied")
    if atlas_bank or ck.get('training_mls') is not None:
        from vemgsbm.spatial_atlas import TrainingMLSAtlas
        saved = load_checkpoint(atlas_bank) if atlas_bank else ck['training_mls']
        payload = saved.get("payload", saved)
        atlas = TrainingMLSAtlas(payload, cfg["mls_regularizer"]).to(device)
        case_ids = [item["id"] for item in batch]
        terms["L_MLS"] = atlas.reconstruction_loss(prediction, truth, case_ids, scale,
            base_coefficients=model.codec.encode(base), reference_image=hr)/normalization
        clipping = _legacy_clipping(atlas, prediction, truth, case_ids, model.codec.encode(base))
    norms = gradient_diagnostics(terms, model.parameters(), weights)
    return dict(status="measured_train", items=len(lr), ssim_items=max(1, len(lr)//8),
        values={name: float(value.detach()) for name, value in terms.items()}, gradient_norms=norms,
        atlas_coordinate_clipping=clipping, peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None)


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    device = torch.device(args.device)
    if args.split != "val":
        raise ValueError("Ceiling audit accepts VAL only; fixed TEST is forbidden")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    from vemgsbm.mri_perceptual import FeatureMoments, frechet_distance, load_extractors, PROTOCOL
    extractors, perceptual_error = None, None
    if args.weights_dir:
        try:
            extractors = load_extractors(args.weights_dir, device)
        except (ImportError, FileNotFoundError, ValueError, RuntimeError) as exc:
            perceptual_error = f"{type(exc).__name__}: {exc}"
            if args.require_perceptual:
                raise
    else:
        perceptual_error = "No weights directory supplied"
        if args.require_perceptual:
            raise ValueError("--require-perceptual requires --weights-dir")
    synthetic = args.synthetic or not args.inventory
    if synthetic:
        pairs = [("synthetic_mri", *synthetic_pair(args.synthetic_size, device=device))]
    else:
        pairs = ((case.id, *_read_case(case, device)) for case in _val_cases(args.inventory, args.config, args.max_cases))
    aggregate = {mode: dict(sse=0., voxels=0, ssim_sum=0., slices=0, variance_pred=0., variance_ref=0.,
                            lpips_sum=0., lpips_count=0, moments=(FeatureMoments(), FeatureMoments()))
                 for mode in ("bicubic", "lanczos")}
    records = []
    gradient_probe = None
    for case_id, lr, hr in pairs:
        row = dict(case_id=case_id, noise=noise_ceiling_estimate(hr), baselines={})
        for mode, accumulator in aggregate.items():
            # Full-case postprocessing mirrors standard inference's [0,1] clip.
            prediction = upsample_xy(lr, 4, mode).clamp(0, 1)
            score = float(mean_xy_ssim(prediction, hr))
            squared_error = float((prediction.double()-hr.double()).square().sum())
            mse = squared_error/hr.numel()
            accumulator["sse"] += squared_error
            accumulator["voxels"] += hr.numel()
            accumulator["ssim_sum"] += score*hr.shape[2]
            accumulator["slices"] += hr.shape[2]
            _, vp = local_moments(prediction); _, vr = local_moments(hr)
            accumulator["variance_pred"] += float(vp.double().sum())
            accumulator["variance_ref"] += float(vr.double().sum())
            case_metrics = dict(psnr=-10*math.log10(mse) if mse else None, ssim=score,
                local_variance_ratio=local_variance_ratio(prediction, hr),
                textured_region_ssim=textured_region_ssim(prediction, hr))
            if extractors:
                lpips_sum, n = perceptual_update(prediction, hr, extractors, accumulator["moments"], args.perceptual_batch)
                accumulator["lpips_sum"] += lpips_sum; accumulator["lpips_count"] += n
                case_metrics["lpips"] = lpips_sum/n
            row["baselines"][mode] = case_metrics
            if gradient_probe is None:
                # Diagnostic uses known synthetic contrast loss, not a claim of
                # gradients measured from an absent historical checkpoint.
                p = prediction[:, :, :2].detach().requires_grad_()
                criterion = ReconstructionLoss()
                terms = criterion(p, hr[:, :, :2])
                gradient_probe = dict(wrt="prediction intensities", source="synthetic" if synthetic else "VAL interpolation",
                    terms={key: float(value.detach()) for key, value in terms.items()},
                    gradients=gradient_diagnostics(terms, (p,), criterion.weights))
        records.append(row)
    if not records:
        raise ValueError("No VAL cases available for the requested audit")
    summary = {}
    for mode, accumulator in aggregate.items():
        mse = accumulator["sse"]/accumulator["voxels"]
        summary[mode] = dict(psnr=-10*math.log10(mse) if mse else None,
            ssim=accumulator["ssim_sum"]/accumulator["slices"],
            local_variance_ratio=accumulator["variance_pred"]/accumulator["variance_ref"] if accumulator["variance_ref"] else None,
            lpips=accumulator["lpips_sum"]/accumulator["lpips_count"] if extractors else None,
            fid=frechet_distance(*accumulator["moments"]) if extractors else None,
            slice_count=accumulator["slices"], pixel_count=accumulator["voxels"])
    legacy = legacy_gradient_audit(args.checkpoint, getattr(args,'legacy_inventory',None) or args.inventory,
                                  args.atlas_bank, device, args.gradient_items)
    result = dict(source="synthetic verification" if synthetic else "VAL", test_used=False,
        physical_ceiling_identified=False, baselines=summary, cases=records,
        v2_loss_gradient_probe=gradient_probe, legacy_gradient_probe=legacy,
        perceptual_status="real pretrained models" if extractors else "unavailable",
        perceptual_unavailable_reason=perceptual_error, perceptual_protocol=PROTOCOL,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, allow_nan=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-size", type=int, default=64)
    parser.add_argument("--inventory")
    parser.add_argument("--config")
    parser.add_argument("--split", choices=["val"], default="val")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument("--legacy-inventory", help="Original checkpoint inventory; distinct from v2 VAL contract")
    parser.add_argument("--atlas-bank")
    parser.add_argument("--gradient-items", type=int, default=2)
    parser.add_argument("--weights-dir")
    parser.add_argument("--require-perceptual", action="store_true")
    parser.add_argument("--perceptual-batch", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="reports/mri/v2_ceiling_audit.json")
    args = parser.parse_args()
    if (args.synthetic_size < 16 or args.synthetic_size % 4 or args.gradient_items < 1
            or args.perceptual_batch < 1 or args.threads < 1 or (args.max_cases is not None and args.max_cases < 1)):
        parser.error("Synthetic XY size must be >=16 and divisible by 4; gradient items positive")
    run(args)


if __name__ == "__main__":
    main()
