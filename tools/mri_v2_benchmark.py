"""Benchmark MRI v2 loss and atlas updates against a source snapshot."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from vemgsbm import jetatlas, mri_v2_losses


def baseline_module(name, revision, source):
    relative = Path("src", "vemgsbm", f"{name}.py")
    data = ((source / relative).read_bytes() if source else
            subprocess.check_output(["git", "show", f"{revision}:{relative.as_posix()}"], cwd=ROOT))
    module = ModuleType(f"vemgsbm._baseline_{name}")
    sys.modules[module.__name__] = module
    exec(compile(data, str(relative), "exec"), module.__dict__)
    return module


def measure(before, after, device, warmup, repeats):
    procedures = (before, after)
    for _ in range(warmup):
        for procedure in procedures:
            procedure()
    samples = [[], []]
    for repeat in range(repeats):
        for index in (range(2) if repeat % 2 == 0 else (1, 0)):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            procedures[index]()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples[index].append((time.perf_counter()-started)*1.e3)
    baseline, updated = (statistics.median(row) for row in samples)
    return dict(baseline_ms=baseline, updated_ms=updated, speedup=baseline/updated,
                samples_ms=dict(baseline=samples[0], updated=samples[1]))


def loss_updates(module, device):
    target = torch.rand(8, 1, 7, 96, 96, device=device)
    prediction = (.85*target+.03*torch.randn_like(target)).requires_grad_()
    # Same weights in both versions isolate implementation speed.
    weights = mri_v2_losses.DEFAULT_WEIGHTS
    objectives = (module.ReconstructionLoss(weights), mri_v2_losses.ReconstructionLoss(weights))

    def update(objective):
        prediction.grad = None
        objective(prediction, target)["loss"].backward()

    before, after = objectives
    return (lambda: update(before), lambda: update(after), dict(shape=list(target.shape), weights=weights))


def atlas_updates(module, device):
    bank = torch.rand(5*96, 135)*.15
    case_ids = [f"TRAIN-{i}" for i in range(5)]
    bank_ids = [case for case in case_ids for _ in range(96)]
    settings = dict(folds=5, centers=8, projection_dim=32, intrinsic=8, neighbors=128,
                    normal_rank=8, query_sites=128, lr_dim=1)
    before = module.JetAtlasRegularizer.fit_from_bank(bank, bank_ids, settings).to(device)
    after = jetatlas.JetAtlasRegularizer.from_payload(before.payload()).to(device)
    after.load_state_dict(before.state_dict())
    target = torch.randn(48, 16, 3, 16, 16, device=device)*.08
    prediction = (target+.02*torch.randn_like(target)).requires_grad_()
    lr = torch.rand(48, 1, 3, 16, 16, device=device)
    hr = torch.rand(48, 1, 3, 64, 64, device=device)
    state = (.5*target).requires_grad_()
    cases = [case_ids[i % len(case_ids)] for i in range(len(lr))]

    def update(objective):
        objective.zero_grad(set_to_none=True)
        prediction.grad = state.grad = None
        objective._forward(prediction, target, cases, lr, state, hr)["loss"].backward()

    return (lambda: update(before), lambda: update(after), dict(shape=list(target.shape), settings=settings))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--baseline-revision")
    source.add_argument("--baseline-source", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--part", choices=("loss", "atlas", "both"), default="both")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.threads, args.repeats) < 1 or args.warmup < 0:
        parser.error("Expected positive threads and repeats and nonnegative warmup")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    device = torch.device(args.device)
    rows = {}
    for name, module, fixture in (("loss", "mri_v2_losses", loss_updates),
                                  ("atlas", "jetatlas", atlas_updates)):
        if args.part not in (name, "both"):
            continue
        old = baseline_module(module, args.baseline_revision, args.baseline_source)
        before, after, details = fixture(old, device)
        rows[name] = {**details, **measure(before, after, device, args.warmup, args.repeats)}
    result = dict(device=args.device, torch=torch.__version__, threads=args.threads,
                  warmup=args.warmup, repeats=args.repeats, results=rows)
    report = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(report+"\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
