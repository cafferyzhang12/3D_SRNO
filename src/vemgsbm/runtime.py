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
    from .mri_v2 import is_v2, validate_config
    if is_v2(cfg):
        return validate_config(cfg)
    if cfg.get("mode") not in MODES:
        raise ValueError(f"Unsupported mode, expected one of {sorted(MODES)}")
    if cfg.get("predict_target") not in {None,"drift","x1"}:
        raise ValueError("predict_target must be drift or x1")
    if cfg.get("predict_target") == "x1":
        sampling = cfg["sampling"]
        sampling.setdefault("sampler_type",cfg.get("infer_sampler","ode"))
        if cfg.get("infer_sampler",sampling["sampler_type"]) != sampling["sampler_type"]:
            raise ValueError("infer_sampler and sampling.sampler_type must agree")
        if sampling.get("solver","euler") not in {"euler","heun"} or (
                sampling["sampler_type"] == "sde" and sampling.get("solver","euler") != "euler"):
            raise ValueError("SDE uses Euler; ODE supports Euler or Heun")
        if type(sampling["steps"]) is not int or sampling["steps"] < 1:
            raise ValueError("Positive integer inference steps required")
        clip = cfg["train"].get("clip_grad_norm",1.)
        decay = cfg["train"].get("ema_decay",.999)
        if not math.isfinite(clip) or clip <= 0 or not 0 <= decay < 1:
            raise ValueError("Invalid gradient clip or EMA decay")
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
        if cfg.get("predict_target") == "x1":
            if cfg["mode"] != "haar_bridge" or cfg.get("spatial_atlas") or cfg.get("atlas"):
                raise ValueError("x1 uses training-only MLS; online atlas branches are unsupported")
            if not 1.e-4 <= cfg["train"].get("matching_time_min",1.e-4) < .5:
                raise ValueError("x1 training times must stay in [1e-4, 1-1e-4]")
            for key in ("sigma_0", "sigma_bridge", "mls_reg_weight"):
                value = cfg.get(key, {"sigma_0":.03,"sigma_bridge":.08,"mls_reg_weight":0.}[key])
                if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid {key}")
            sr_reconstruction = cfg["train"].get("objective") == "sr_reconstruction"
            if (not sr_reconstruction and cfg.get("sigma_0",.03) <= 0) or cfg.get("sigma_bridge",.08) != sigma:
                raise ValueError("Bridge x1 needs positive sigma_0 and matching sigma_bridge/path.sigma")
            if cfg.get("use_band_whitening",True) and not cfg.get("band_stats_cache"):
                raise ValueError("x1 whitening requires band_stats_cache")
            if cfg["sampling"].get("readout","endpoint") != "endpoint":
                raise ValueError("x1 has only an endpoint readout")
            if not sr_reconstruction and not 2 <= cfg["train"].get("rollout_steps",3) <= 4:
                raise ValueError("x1 rollout_steps must be 2..4")
            if sr_reconstruction:
                if (cfg.get("sigma_0",.03) != 0 or cfg.get("sigma_bridge",.08) != 0
                        or sampling["sampler_type"] != "ode" or sampling.get("solver","euler") != "euler"):
                    raise ValueError("SR reconstruction requires a zero prior, zero bridge noise and Euler ODE")
                if cfg["train"].get("rollout_steps") != sampling["steps"]:
                    raise ValueError("SR rollout and deployment must use the same time grid")
                if cfg.get("straightening"):
                    raise ValueError("SR reconstruction is supervised by paired HR, not a teacher endpoint")
                for key, default in (("start_weight",1.),("rollout_weight",1.),("hf_weight",.1),("edge_weight",.05),("ssim_weight",0.)):
                    value = cfg["train"].get(key,default)
                    if not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"Invalid SR {key}")
                    if key in {"start_weight","rollout_weight"} and value == 0:
                        raise ValueError("SR reconstruction requires both start and rollout supervision")
            if cfg["train"].get("coefficient_loss","charbonnier") not in {"l1","charbonnier"}:
                raise ValueError("Unknown coefficient loss")
            for key in ("coefficient_weight","pixel_weight","rollout_weight","charbonnier_eps"):
                value = cfg["train"].get(key,1.e-3 if key == "charbonnier_eps" else 1.)
                if not math.isfinite(value) or value < 0 or (key == "charbonnier_eps" and value == 0):
                    raise ValueError(f"Invalid {key}")
            if cfg.get("mls_reg_weight",0.) and not cfg.get("mls_regularizer"):
                raise ValueError("Positive mls_reg_weight requires mls_regularizer settings")
            if cfg.get("mls_reg_weight",0.):
                geometry_mode = cfg["mls_regularizer"].get("geometry_mode","normal")
                expected_mode = "paired_curved" if sr_reconstruction else "normal"
                if geometry_mode != expected_mode:
                    raise ValueError(f"This objective requires MLS geometry_mode={expected_mode}")
            if cfg.get("straightening") and cfg.get("sigma_bridge",.08) != 0:
                raise ValueError("Straightening trains straight paths: set sigma_bridge/path.sigma/sampling.sigma to zero")
            if cfg.get("infer_steps",cfg["sampling"]["steps"]) != cfg["sampling"]["steps"]:
                raise ValueError("infer_steps and sampling.steps must agree")
        elif cfg["train"].get("matching_time_min", .02) != .02:
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
    from .mri_v2 import is_v2
    if is_v2(config):
        return (float(config.get('sigma_bridge',0.))*float(sampling.get('temperature',0.))
                if sampling.get('sampler_type') == 'sde' else 0.)
    sampler = sampling.get("sampler_type", config.get("infer_sampler"))
    if sampler is not None:
        if sampler not in {"ode", "sde"}:
            raise ValueError("sampler_type must be ode or sde")
        if sampler == "ode":
            return 0.0
        reference = config.get("sigma_bridge", config["path"]["sigma"])
        supplied = sampling.get("inference_sigma", reference)
        if not math.isfinite(reference) or reference < 0 or supplied != reference:
            raise ValueError("SDE inference_sigma must equal sigma_bridge")
        return float(reference)
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
    if device_name.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
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
    from .mri_v2 import is_v2, make_model
    if is_v2(cfg):
        if checkpoint.get('format_version') != 6:
            raise ValueError('Hybrid MRI v2 requires checkpoint format 6')
        model = make_model(cfg).to(device)
        model.set_band_statistics(checkpoint['band_statistics'])
        model.load_state_dict(checkpoint['ema_model'] if checkpoint.get('evaluation_weights') == 'ema' else checkpoint['model'])
        geometry = None
        if cfg.get('online_atlas',False):
            from .jetatlas import JetAtlasRegularizer
            geometry = JetAtlasRegularizer.from_payload(checkpoint['jetatlas']).to(device)
            geometry.load_state_dict(checkpoint['jetatlas_state'])
            geometry.eval()
        return model.eval(), geometry
    guidance = sampling_geometry_guidance(cfg)
    geometry = None
    if cfg["mode"] == "external_2d":
        from .mri_external import SlicePredictor

        model = SlicePredictor(cfg["external"]["name"]).to(device)
    elif cfg["mode"] in {"haar_direct", "haar_bridge"}:
        from .haar_residual import make_haar_model

        if checkpoint.get("format_version") != (5 if cfg.get("predict_target") == "x1" else
                                                 (4 if cfg.get("spatial_atlas") else 3)):
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
    weights = "ema_model" if checkpoint.get("evaluation_weights") == "ema" else "model"
    model.load_state_dict(checkpoint[weights])
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
