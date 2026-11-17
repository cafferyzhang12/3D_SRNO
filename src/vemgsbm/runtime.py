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
MODES = {"gsbm", "momentum"}


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if cfg.get("mode") not in MODES:
        raise ValueError("Expected mode=gsbm or mode=momentum")
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
    data = cfg["data"]
    if cfg.get("atlas") and (data.get("kind") != "paired_mri_png" or any(p % 4 for p in patch[1:])):
        raise ValueError("The MRI MLS atlas requires paired MRI and XY patches divisible by four")
    sampling_geometry_guidance(cfg)
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


def sampling_geometry_guidance(config):
    from .bridge import geometry_guidance_options

    options = geometry_guidance_options(config.get("sampling", {}).get("geometry_guidance"))
    if options is not None and (config["mode"] != "gsbm" or not (config.get("atlas") or {}).get("source_routing")
                                or config["data"].get("kind") != "paired_mri_png"):
        raise ValueError("Online MLS guidance requires an LR-routed MLS GSBM MRI configuration")
    return options


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
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def build_models(checkpoint, device):
    cfg = checkpoint["config"]
    guidance = sampling_geometry_guidance(cfg)
    geometry = None
    if cfg["mode"] == "momentum":
        from .momentum.model import AccelerationNetwork
        from .momentum.trivialized import ChartMomentumAtlas, chart_momentum_options

        chart_options = chart_momentum_options(cfg)
        model = AccelerationNetwork(**cfg["model"], chart_features=chart_options is not None).to(device)
        if chart_options is not None:
            geometry = ChartMomentumAtlas(checkpoint["atlas"], chart_options).to(device)
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
