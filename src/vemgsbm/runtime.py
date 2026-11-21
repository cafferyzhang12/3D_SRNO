"""Configuration, reproducibility, and atomic local artifacts."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODES = {"gsbm", "momentum", "haar_direct", "haar_bridge"}


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if cfg.get("mode") not in MODES:
        raise ValueError(f"Unsupported mode, expected one of {sorted(MODES)}")
    patch = cfg["data"]["patch_size"]
    if len(patch) != 3 or any(not isinstance(v, int) or isinstance(v, bool) or v < 3 for v in patch):
        raise ValueError("patch_size must be ZYX")
    if cfg["train"]["samples_per_epoch"] < 1 or cfg["train"]["batch_size"] < 1:
        raise ValueError("Positive training budget required")
    if cfg["mode"] == "gsbm" and cfg["train"].get("matching_time_min", 0.02) != 0.02:
        raise ValueError("Current SDE sampler and training require matching_time_min=0.02")
    sigma = cfg["path"]["sigma"]
    if not math.isfinite(sigma) or sigma < 0 or sigma != cfg["sampling"]["sigma"]:
        raise ValueError("Path and SDE reference diffusion sigma must agree and be finite")
    sampling_noise(cfg)
    data = cfg["data"]
    if cfg.get("spatial_atlas") is not None:
        options = cfg["spatial_atlas"]
        if (cfg["mode"] != "haar_bridge" or options.get("method") not in {"mean", "pca", "mls"}
                or options.get("query", "endpoint") not in {"state", "endpoint"}
                or options.get("reference", "identity") not in {"identity", "geometry", "shuffled"}
                or options.get("dim") not in {2, 4, 6, 8}):
            raise ValueError("Invalid spatial atlas method, reference, query, or local dimension")
        if options.get("path_geometry") and (cfg["path"]["iterations"] < 1 or
                                             min(cfg["path"]["state_weight"], cfg["path"]["metric_weight"]) <= 0):
            raise ValueError("Geometry path optimization requires positive iterations and geometry weights")
    if cfg["mode"] in {"haar_direct", "haar_bridge"}:
        if data.get("kind") != "paired_mri_png" or data.get("scale") != 4:
            raise ValueError("Haar residual MRI requires XY×4 paired PNGs")
        widths = cfg["model"]["widths"]
        if len(widths) != 3 or any(type(v) is not int or v < 2 for v in widths):
            raise ValueError("Expected three positive model widths >=2")
        weight = cfg["train"].get("endpoint_weight", 1.)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Invalid endpoint_weight")
        if cfg["train"].get("matching_time_min", .02) != .02:
            raise ValueError("Haar MRI sampler uses matching_time_min=.02")
        if cfg["mode"] == "haar_bridge" and weight == 0 and cfg["sampling"].get("readout") == "endpoint":
            raise ValueError("Endpoint readout requires endpoint supervision")
    if cfg.get("atlas") and (data.get("kind") != "paired_mri_png" or any(p % 4 for p in patch[1:])):
        raise ValueError("The MRI MLS atlas requires paired MRI and XY patches divisible by four")
    sampling_geometry_guidance(cfg)
    sampling_momentum_guidance(cfg)
    if cfg.get("chart_momentum") is not None:
        from .momentum.trivialized import chart_momentum_options

        chart_momentum_options(cfg)
    if data.get("kind") == "paired_mri_png":
        if data.get("degradation") != "paired_xy" or data["scale"] not in (2, 4):
            raise ValueError("MRI requires paired XY scaling 2 or 4")
        if any(p % data["scale"] for p in patch[1:]):
            raise ValueError("MRI XY patches must align with the LR grid")
    elif data.get("degradation") != "slice_select" or (patch[0] - 1) % data["scale"]:
        raise ValueError("EM slice-select patches must have aligned Z anchors")
    if cfg["mode"] == "momentum" and any(p % 4 for p in patch[1:]):
        raise ValueError("Momentum requires XY patch sizes divisible by four")
    return cfg


def sampling_noise(config):
    """Return the inference noise, with older checkpoints using reference noise."""
    sampling = config["sampling"]
    sigma = sampling.get("inference_sigma", sampling["sigma"])
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or not math.isfinite(sigma) or sigma < 0:
        raise ValueError("Inference diffusion sigma must be a nonnegative finite number")
    return float(sigma)


def sampling_geometry_guidance(config):
    from .bridge import geometry_guidance_options

    options = geometry_guidance_options(config.get("sampling", {}).get("geometry_guidance"))
    if options is not None and (config["mode"] != "gsbm" or not (config.get("atlas") or {}).get("source_routing")
                                or config["data"].get("kind") != "paired_mri_png"):
        raise ValueError("Online MLS guidance requires an LR-routed MLS GSBM MRI configuration")
    return options


def sampling_momentum_guidance(config):
    if config.get("sampling", {}).get("momentum_geometry") is None:
        return None
    from .momentum.guidance import momentum_geometry_options

    return momentum_geometry_options(config)


def setup(seed, device_name="auto", threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu for a small smoke run.")
    return torch.device(device_name)


def atomic_save(value, path, *, json_file=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        if json_file:
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        else:
            torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def build_models(checkpoint, device):
    cfg = checkpoint["config"]
    guidance = sampling_geometry_guidance(cfg)
    geometry = None
    if cfg["mode"] == "external_2d":
        from .mri_external import SlicePredictor

        model = SlicePredictor(cfg["external"]["name"]).to(device)
    elif cfg["mode"] in {"haar_direct", "haar_bridge"}:
        from .haar_residual import make_haar_model

        if checkpoint.get("format_version") != (4 if cfg.get("spatial_atlas") else 3):
            raise ValueError("Unsupported native-LR Haar MRI checkpoint version")
        model = make_haar_model(cfg).to(device)
        if cfg.get("spatial_atlas"):
            from .spatial_atlas import load_spatial_atlas

            geometry = load_spatial_atlas(checkpoint["spatial_atlas"], cfg["spatial_atlas"]).to(device).eval()
    elif cfg["mode"] == "momentum":
        from .momentum.model import AccelerationNetwork
        from .momentum.trivialized import ChartMomentumAtlas, chart_momentum_options

        chart_options = chart_momentum_options(cfg)
        model = AccelerationNetwork(**cfg["model"], chart_features=chart_options is not None).to(device)
        if chart_options is not None:
            geometry = ChartMomentumAtlas(checkpoint["atlas"], chart_options).to(device)
        elif sampling_momentum_guidance(cfg) is not None:
            from .atlas_geometry import MRIAtlasGeometry

            payload = checkpoint.get("atlas")
            if not isinstance(payload, dict) or payload.get("kind") != "mls" or "routing" not in payload:
                raise ValueError("Guided momentum checkpoint requires an LR-routed MRI atlas")
            geometry = MRIAtlasGeometry(payload).to(device)
    else:
        from .network import DriftNetwork
        model = DriftNetwork(cfg["model"]["width"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if guidance is not None:
        from .atlas_geometry import MRIAtlasGeometry

        payload = checkpoint.get("metric")
        if not isinstance(payload, dict) or payload.get("kind") != "mls":
            raise ValueError("Guided GSBM checkpoint is missing its MRI MLS geometry")
        geometry = MRIAtlasGeometry(payload).to(device).eval()
        if not geometry.source_routing:
            raise ValueError("Online guidance requires MRI MLS geometry fitted with LR chart routing")
    return model, geometry
