"""Paired slice LPIPS and cohort FID after unchanged whole-case MRI inference.

All slices, including background, are evaluated once. Grayscale is repeated in
RGB. FID is a 2D ImageNet feature statistic, not a 3D or MRI-specific metric.
Pretrained assets must be prepared explicitly; evaluation is offline.
"""

import argparse
import copy
from importlib.metadata import version
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy import linalg
import torch
import torch.distributed as dist

from .distributed import initialize_distributed
from .mri_data import read_plane
from .mri_infer import _infer_case, checked_test_subset
from .mri_inventory import load_inventory
from .runtime import atomic_save, build_models, fingerprint, load_checkpoint, setup


ASSETS = {
    "alexnet": ("alexnet-owt-7be5be79.pth", "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth", "7be5be79"),
    "inception": ("weights-inception-2015-12-05-6726825d.pth",
                  "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth", "6726825d"),
}
PROTOCOL = {
    "version": 1,
    "lpips": "lpips 0.1.4, alex backbone, calibrated v0.1; grayscale repeated RGB, float32 [-1,1], native XY resolution; equal slice mean",
    "fid": "torch-fidelity 0.3.0 TensorFlow-compatible InceptionV3 pool3 2048; grayscale repeated RGB; round(255*x) uint8; extractor TensorFlow-compatible bilinear resize to 299x299",
    "fid_aggregation": "one cohort FID from float64 global feature sums and outer products, unbiased sample covariance (N-1); never mean case/rank FID",
    "scope": "all retained slices, full field of view, no foreground cropping or per-image intensity normalization",
    "interpretation": "2D ImageNet metrics on correlated MRI slices; FID is cohort-size dependent and is not a 3D anatomical or clinical metric",
    "selection": "reporting only; LPIPS/FID never select TEST cases or checkpoints",
}


def _packages():
    expected = {"lpips": "0.1.4", "torch-fidelity": "0.3.0"}
    actual = {key: version(key) for key in expected}
    if actual != expected:
        raise ValueError(f"Perceptual protocol requires {expected}; found {actual}")
    return {**actual, "torch": torch.__version__, "torchvision": version("torchvision")}


def _asset_identity(weights_dir):
    import lpips
    paths = {key: Path(weights_dir) / "checkpoints" / row[0] for key, row in ASSETS.items()}
    paths["lpips_calibration"] = Path(lpips.__file__).parent / "weights" / "v0.1" / "alex.pth"
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("Pretrained perceptual weights missing; run prepare-weights in an allocated job first")
    return {key: fingerprint(path) for key, path in paths.items()}


def load_extractors(weights_dir, device, *, verify=True):
    packages = _packages()
    directory = Path(weights_dir).resolve()
    identity = _asset_identity(directory)
    if verify:
        record = json.loads((directory / "weights.json").read_text())
        if record["sha256"] != identity or record["packages"] != packages:
            raise ValueError("Perceptual weights or package versions differ from the prepared manifest")
    # torchvision's LPIPS trunk loads this existing, verified cache file.
    torch.hub.set_dir(str(directory))
    import lpips
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    paired = lpips.LPIPS(net="alex", version="0.1", pretrained=True, pnet_rand=False, verbose=False)
    features = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"],
        feature_extractor_weights_path=str(directory / "checkpoints" / ASSETS["inception"][0]))
    return paired.eval().requires_grad_(False).to(device), features.eval().requires_grad_(False).to(device)


@torch.no_grad()
def prepare_weights(directory, device):
    directory = Path(directory).resolve()
    folder = directory / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    packages = _packages()
    for filename, url, prefix in ASSETS.values():
        path = folder / filename
        if not path.exists():
            temporary = path.with_suffix(".download")
            torch.hub.download_url_to_file(url, str(temporary), hash_prefix=prefix)
            temporary.replace(path)
        if not fingerprint(path).startswith(prefix):
            raise ValueError(f"Invalid official pretrained weight digest: {filename}")
    paired, features = load_extractors(directory, device, verify=False)
    x = torch.zeros(2, 3, 64, 64, device=device)
    score = paired(x, x)
    embedding = features(x.to(torch.uint8))[0]
    if not torch.isfinite(embedding).all() or embedding.shape != (2, 2048) or not torch.allclose(score, torch.zeros_like(score), atol=1.e-7):
        raise RuntimeError("Pretrained perceptual model smoke check failed")
    record = {"sha256": _asset_identity(directory), "packages": packages, "protocol": PROTOCOL,
              "preflight": "real pretrained models: finite 2048 features and identical-image LPIPS zero"}
    atomic_save(record, directory / "weights.json", json_file=True)
    return record


def rgb_inputs(planes, device):
    x = torch.as_tensor(np.asarray(planes).copy(), device=device, dtype=torch.float32)
    if x.ndim != 3 or not len(x) or not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("Perceptual inputs must be finite [0,1] BHW slices")
    rgb = x[:, None].repeat(1, 3, 1, 1)
    return rgb.mul(2).sub(1), rgb.mul(255).round().to(torch.uint8)


class FeatureMoments:
    """Additive sufficient statistics, mergeable across disjoint GPU shards."""
    def __init__(self, dimension=2048):
        self.n = 0
        self.total = np.zeros(dimension, np.float64)
        self.outer = np.zeros((dimension, dimension), np.float64)

    def update(self, features):
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.total) or not np.isfinite(x).all():
            raise ValueError("Invalid FID features")
        self.n += len(x)
        self.total += x.sum(axis=0)
        self.outer += x.T @ x

    def reduce(self, context):
        if context.is_distributed:
            count = torch.tensor(self.n, dtype=torch.int64, device=context.device)
            dist.all_reduce(count)
            self.n = int(count)
            for field in ("total", "outer"):
                value = torch.as_tensor(getattr(self, field), device=context.device)
                dist.all_reduce(value)
                setattr(self, field, value.cpu().numpy())

    def statistics(self):
        if self.n < 2:
            raise ValueError("FID requires at least two slices in each cohort")
        mean = self.total / self.n
        covariance = (self.outer - self.n * np.outer(mean, mean)) / (self.n - 1)
        return mean, (covariance + covariance.T) * .5


def frechet_distance(first, second):
    """Stable symmetric PSD form of the standard Gaussian Frechet distance.

    Handles singular covariance (e.g. 595 VAL slices vs 2048 features). Tiny
    negative eigenvalues from floating-point arithmetic are clipped to zero.
    """
    mean_a, a = first.statistics()
    mean_b, b = second.statistics()
    eigenvalues, eigenvectors = linalg.eigh(a, check_finite=True)
    tolerance = 1.e-8 * max(1., float(np.abs(eigenvalues).max()))
    if eigenvalues.min() < -tolerance:
        raise FloatingPointError("FID covariance is not positive semidefinite")
    root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
    middle = root @ b @ root
    middle = (middle + middle.T) * .5
    cross = np.sqrt(np.maximum(linalg.eigvalsh(middle), 0)).sum()
    value = float(np.square(mean_a - mean_b).sum() + np.trace(a) + np.trace(b) - 2 * cross)
    if not np.isfinite(value) or value < -1.e-6 * max(1., np.trace(a) + np.trace(b)):
        raise FloatingPointError("Invalid FID after covariance square root")
    return max(0., value)


@torch.no_grad()
def consume_case(volume, case, paired, features, predicted, reference, device, batch_size):
    if tuple(volume.shape) != (len(case.hq), *[s * case.scale for s in case.lr_shape]):
        raise ValueError("Reconstruction does not match the complete reference case")
    lpips_sum = 0.
    for start in range(0, len(case.hq), batch_size):
        paths = case.hq[start:start + batch_size]
        truth = np.stack([read_plane(path, case.normalization_scale, case.normalization_offset,
                                    case.channel_policy, case.bits) for path in paths])
        p_lp, p_fid = rgb_inputs(volume[start:start + len(paths)], device)
        y_lp, y_fid = rgb_inputs(truth, device)
        distances = paired(p_lp, y_lp).reshape(-1)
        if len(distances) != len(paths) or not torch.isfinite(distances).all():
            raise FloatingPointError("Invalid per-slice LPIPS values")
        lpips_sum += float(distances.double().sum())
        predicted.update(features(p_fid)[0].cpu().numpy())
        reference.update(features(y_fid)[0].cpu().numpy())
    return {"lpips_sum": lpips_sum, "slice_count": len(case.hq), "lpips": lpips_sum / len(case.hq)}


def evaluate(args):
    context = initialize_distributed(args.device)
    try:
        setup(42, str(context.device), threads=4)
        checkpoint = load_checkpoint(args.checkpoint)
        cfg = copy.deepcopy(checkpoint["config"])
        cfg["inference_variant"] = "endpoint_safe"
        cases, _ = load_inventory(args.inventory, cfg["data"], (args.split,))
        cases = cases[args.split]
        if args.split == "test":
            if not args.manifest:
                raise ValueError("TEST perceptual metrics require the fixed case manifest")
            cases, _ = checked_test_subset(cases, .15, 20261110, args.manifest)
        elif args.manifest:
            raise ValueError("VAL uses every case; no TEST manifest accepted")
        root = Path(args.output).resolve()
        context.main_call(lambda: root.mkdir(parents=True, exist_ok=False))
        model = geometry = paired = features = None
        error = None
        try:
            model, geometry = build_models(checkpoint, context.device)
            paired, features = load_extractors(args.weights_dir, context.device)
        except Exception as exc:
            error = f"Model setup: {type(exc).__name__}: {exc}"
        context.check_errors(error)
        predicted, reference = FeatureMoments(), FeatureMoments()
        rows, error = [], None
        try:
            for case in cases[context.rank::context.world_size]:
                with tempfile.TemporaryDirectory(prefix=f"case-{case.id}-", dir=root) as temp:
                    folder = Path(temp) / "reconstruction"
                    _, metrics = _infer_case(case, folder, model, geometry, cfg, context.device,
                                              cfg["train"].get("validation_seed", 42), cfg["sampling"]["steps"], 1)
                    volume = np.load(folder / "mean.npy", mmap_mode="r", allow_pickle=False)
                    try:
                        perceptual = consume_case(volume, case, paired, features, predicted, reference,
                                                  context.device, args.batch_size)
                    finally:
                        volume._mmap.close()
                row = {"case_id": case.id, "input_fingerprint": case.fingerprint,
                       "metrics": {k: v for k, v in metrics.items() if k != "slices"}, **perceptual}
                atomic_save(row, root / f"case-{case.id}.json", json_file=True)
                rows.append(row)
                print(json.dumps({"perceptual_case": case.id, "slices": row["slice_count"], "lpips": row["lpips"]}), flush=True)
        except Exception as exc:
            error = f"Perceptual case evaluation: {type(exc).__name__}: {exc}"
        context.check_errors(error)
        if context.is_distributed:
            gathered = [None] * context.world_size
            dist.all_gather_object(gathered, rows)
            rows = [row for shard in gathered for row in shard]
        rows.sort(key=lambda row: row["case_id"])
        if [row["case_id"] for row in rows] != sorted(case.id for case in cases):
            raise RuntimeError("Incomplete or duplicated perceptual case cohort")
        predicted.reduce(context)
        reference.reduce(context)

        def publish():
            n = sum(row["slice_count"] for row in rows)
            if n != predicted.n or n != reference.n:
                raise RuntimeError("FID feature counts differ from paired slice counts")
            pixels = sum(row["metrics"]["pixel_count"] for row in rows)
            mse = sum(row["metrics"]["squared_error_sum"] for row in rows) / pixels
            summary = {"checkpoint_sha256": fingerprint(args.checkpoint), "checkpoint_epoch": checkpoint["epoch"],
                       "evaluation_weights": checkpoint.get("evaluation_weights", "model"),
                       "inventory_sha256": fingerprint(args.inventory), "split": args.split,
                       "manifest_sha256": fingerprint(args.manifest) if args.manifest else None,
                       "case_count": len(rows), "slice_count": n, "pixel_count": pixels,
                       "all_mse": mse, "all_psnr_db": -10 * np.log10(mse) if mse else None,
                       "mean_slice_ssim": sum(row["metrics"]["mean_slice_ssim"] * row["slice_count"] for row in rows) / n,
                       "lpips": sum(row["lpips_sum"] for row in rows) / n,
                       "fid": frechet_distance(predicted, reference),
                       "fid_feature_dimension": 2048, "fid_covariance_rank_deficient_by_sample_count": n <= 2048,
                       "steps": cfg["sampling"]["steps"], "world_size": context.world_size,
                       "weights": json.loads((Path(args.weights_dir) / "weights.json").read_text()),
                       "protocol": PROTOCOL, "cases": rows}
            atomic_save(summary, root / "summary.json", json_file=True)
            print(json.dumps({k: v for k, v in summary.items() if k not in {"cases", "weights", "protocol"}}), flush=True)
        context.main_call(publish)
    finally:
        context.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare-weights")
    prepare.add_argument("--weights-dir", required=True)
    prepare.add_argument("--device", default="cuda")
    run = sub.add_parser("evaluate")
    for key in ("checkpoint", "inventory", "weights-dir", "output"):
        run.add_argument("--" + key, required=True)
    run.add_argument("--split", required=True, choices=("val", "test"))
    run.add_argument("--manifest")
    run.add_argument("--device", default="cuda")
    run.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.action == "prepare-weights":
        print(json.dumps(prepare_weights(args.weights_dir, setup(42, args.device, threads=4)), indent=2))
    else:
        if args.batch_size < 1:
            parser.error("--batch-size must be positive")
        evaluate(args)


if __name__ == "__main__":
    main()
