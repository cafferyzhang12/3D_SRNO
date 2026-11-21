"""Import official 2D SwinIR-M x4 or prostate Res-SRDiff weights for the paired MRI evaluator."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn
from torch.nn import functional as F

from .runtime import PROJECT_ROOT, atomic_save, fingerprint, load_checkpoint, read_config


SOURCES = {
    "swinir": ("JingyunLiang/SwinIR", "6545850fbf8df298df73d81f3e8cba638787c8bd", "DF2K"),
    "res_srdiff": ("mosaf/Res-SRDiff", "2a0e0a8c6de924f61f496f1eaad9557038971c99", "prostate MRI"),
}


class SlicePredictor(nn.Module):
    def __init__(self, name):
        super().__init__()
        repo, revision, _ = SOURCES[name]
        root = Path(os.environ.get("MRI_EXTERNAL_ROOT", PROJECT_ROOT / "references")) / repo.split("/")[1]
        checkout = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if checkout != revision:
            raise ValueError(f"{name} requires upstream source revision {revision}")
        source_folder = root / ("script" if name == "res_srdiff" else "")
        source_dirs = ("networks", "diffusion") if name == "res_srdiff" else ("models",)
        subprocess.run(["git", "-C", str(source_folder), "diff", "--exit-code", "--quiet", "HEAD", "--",
                        *source_dirs], check=True)
        if str(source_folder) not in sys.path:
            sys.path.insert(0, str(source_folder))
        self.name = name
        if name == "swinir":
            from models.network_swinir import SwinIR

            self.net = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.,
                              depths=[6] * 6, embed_dim=180, num_heads=[6] * 6, mlp_ratio=2,
                              upsampler="pixelshuffle", resi_connection="1conv")
        elif name == "res_srdiff":
            from networks.unet import UNetModelSwin
            from diffusion.Gaussian_model import create_gaussian_diffusion

            self.net = UNetModelSwin(image_size=256, in_channels=1, out_channels=1, model_channels=160,
                                     num_res_blocks=[2] * 4, attention_resolutions=(64, 32, 16, 8), lq_size=256)
            self.diffusion = create_gaussian_diffusion(
                sf=1., schedule_name="exponential", schedule_kwargs={"power": .3}, etas_end=.99,
                steps=4, min_noise_level=.2, kappa=2., scale_factor=1., normalize_input=False, latent_flag=False)
        else:
            raise ValueError(f"Unknown external baseline {name!r}")

    def restore(self, native_lr, base, steps, sigma, generator):
        if native_lr.shape[:3] != (1, 1, 1):
            raise ValueError("External MRI baselines reconstruct one original XY plane at a time")
        low = native_lr[:, :, 0]
        if self.name == "swinir":
            if steps != 1:
                raise ValueError("SwinIR uses one forward pass")
            h, w = low.shape[-2:]
            # Symmetric image extension used by the official classical-SR evaluation.
            # Even an exactly window-aligned image receives one extra window.
            for axis in (2, 3):
                size = low.shape[axis]
                padded_size = (size // 8 + 1) * 8
                while low.shape[axis] < padded_size:
                    low = torch.cat((low, low.flip([axis])), dim=axis)
                low = low.narrow(axis, 0, padded_size)
            rgb = self.net(low.expand(-1, 3, -1, -1).contiguous())[..., :4 * h, :4 * w]
            return rgb.mean(dim=1, keepdim=True).unsqueeze(2)
        if steps != 4:
            raise ValueError("The released Res-SRDiff checkpoint uses four diffusion evaluations")
        # Nearest-neighbour lift of the measured LR. Use a fixed linear unit conversion.
        condition = F.interpolate(low, scale_factor=4, mode="nearest") * 2 - 1
        h, w = condition.shape[-2:]
        condition = F.pad(condition, (0, -w % 64, 0, -h % 64), mode="replicate")
        # Upstream p_sample_loop draws from the default device generator. Isolate its
        # stream with one seed from the evaluator's case-local, rank-independent generator.
        seed = int(torch.randint(2**31, (), generator=generator, device=low.device))
        devices = [low.device] if low.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            if low.is_cuda:
                with torch.cuda.device(low.device):
                    torch.cuda.manual_seed(seed)
            else:
                torch.random.default_generator.manual_seed(seed)
            image = self.diffusion.p_sample_loop(y=condition, model=self.net,
                                                 model_kwargs={"lq": condition}, device=low.device)
        return ((image[..., :h, :w] + 1.) * .5).unsqueeze(2)


def package(name, weights, config, output):
    """Save tensor-only deployment weights alongside the exact local data contract."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if config["data"].get("kind") != "paired_mri_png" or config["data"]["scale"] != 4:
        raise ValueError("External x4 models require paired MRI x4")
    repo, revision, pretraining = SOURCES[name]
    sha = fingerprint(weights)
    with torch.serialization.safe_globals([argparse.Namespace]):
        source = torch.load(weights, weights_only=True, map_location="cpu", mmap=True)
    key = "params" if name == "swinir" else "model_state_dict"
    net_weights = source[key] if key in source else source
    predictor = SlicePredictor(name)
    predictor.net.load_state_dict(net_weights, strict=True)
    config = {
        "data": config["data"], "mode": "external_2d", "train": {"threads": 4},
        "sampling": {"sigma": 0., "steps": 1 if name == "swinir" else 4},
        "external": {"name": name, "pretraining": pretraining, "repository": repo, "revision": revision,
                     "weights_sha256": sha, "upstream_download_sha256": source.get("upstream_download_sha256", sha),
                     "input": "single native LR slice",
                     "eval_patch_zyx": [1, 512, 512]},
    }
    atomic_save({"format_version": 1, "stage": "reconstruction", "config": config,
                 "model": predictor.state_dict()}, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, choices=SOURCES)
    parser.add_argument("--weights", required=True)
    protocol = parser.add_mutually_exclusive_group(required=True)
    protocol.add_argument("--data-config", help="Effective config for this cluster dataset")
    protocol.add_argument("--protocol-checkpoint", help="Use the paired data contract in an archived checkpoint")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = read_config(args.data_config) if args.data_config else load_checkpoint(args.protocol_checkpoint)["config"]
    package(args.name, args.weights, config, args.output)


if __name__ == "__main__":
    main()
