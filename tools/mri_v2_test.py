"""Confirm the frozen whole-VAL winner on the historical MRI XY×4 TEST cohort."""
import argparse
import gzip
import json
import math
from pathlib import Path
import shutil

from vemgsbm.distributed import initialize_distributed
from vemgsbm.mri_infer import checked_test_subset
from vemgsbm.mri_inventory import load_inventory
from vemgsbm.mri_v2_train import validate
from vemgsbm.runtime import atomic_save, build_models, fingerprint, load_checkpoint, setup


def select_candidate(arms):
    candidates = []
    for arm in arms:
        checkpoint = arm / "best.pt"
        if not checkpoint.is_file():
            checkpoint = arm / "last.pt"
        report = json.loads((arm / "final_val.json").read_text(encoding="utf-8"))
        digest = fingerprint(checkpoint)
        if report["checkpoint_sha256"] != digest:
            raise ValueError("Whole-VAL result does not match the candidate checkpoint")
        if not math.isfinite(report["metrics"]["psnr"]):
            raise ValueError("Whole-VAL pooled PSNR must be finite")
        candidates.append((checkpoint, digest, report))
    return max(candidates, key=lambda value: value[2]["metrics"]["psnr"])


def test_cases(inventory, manifest, checkpoint, validation):
    with gzip.open(inventory, "rt", encoding="utf-8") as stream:
        data = json.load(stream)["data"]
    cfg = checkpoint["config"]
    expected = {k: v for k, v in data.items()
                if k not in ("test", "scan_cache", "legacy_scan_cache", "patch_size")}
    observed = {k: v for k, v in cfg["data"].items()
                if k not in ("scan_cache", "legacy_scan_cache", "patch_size")}
    if expected != observed or data["scale"] != 4 or data.get("debug_case_limit") is not None:
        raise ValueError("The TEST inventory does not share the fitted TRAIN/VAL data contract")
    cases, identity = load_inventory(inventory, data, ("test",))
    if [row for row in identity if row["split"] in ("train", "val")] != checkpoint["data_identity"]:
        raise ValueError("TRAIN/VAL fingerprints differ from the checkpoint")
    case_ids = [row["id"].casefold() for row in identity]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("TRAIN, VAL and TEST case identities overlap")
    if sorted(row["id"] for row in identity if row["split"] == "val") != sorted(
            row["case_id"] for row in validation["cases"]):
        raise ValueError("Checkpoint selection requires the complete whole-VAL cohort")
    selected, subset = checked_test_subset(cases["test"], .15, 20261110, manifest)
    slices = sum(len(case.stems) for case in selected)
    pixels = sum(len(case.stems)*case.lr_shape[0]*case.lr_shape[1]*case.scale**2 for case in selected)
    if (len(selected), slices, pixels) != (84, 14991, 3929800704):
        raise ValueError("Historical XY×4 TEST extent differs")
    return selected, subset


def run(arms, inventory, manifest, output, device):
    context = initialize_distributed(device)
    try:
        source, digest, validation = context.main_call(lambda: select_candidate(arms), broadcast_result=True)

        def freeze():
            output.mkdir(parents=True, exist_ok=False)
            path = output / "checkpoint.pt"
            shutil.copyfile(source, path)
            if fingerprint(path) != digest:
                raise ValueError("Checkpoint changed during freeze")
            return path

        checkpoint = load_checkpoint(context.main_call(freeze, broadcast_result=True))
        cases, subset = context.main_call(
            lambda: test_cases(inventory, manifest, checkpoint, validation), broadcast_result=True)
        cfg = checkpoint["config"]
        setup(cfg["train"]["seed"], str(context.device), cfg["train"].get("threads", 4))
        model, atlas = build_models(checkpoint, context.device)
        metrics, records = validate(model, cases, cfg, context, output, atlas)
        if (metrics["slice_count"], metrics["pixel_count"]) != (14991, 3929800704):
            raise RuntimeError("Incomplete historical TEST coverage")
        report = dict(checkpoint_sha256=digest, source_checkpoint=str(source), subset=subset,
                      selection="whole_val_pooled_psnr", validation=validation["metrics"],
                      inventory_sha256=fingerprint(inventory), manifest_sha256=fingerprint(manifest),
                      metrics=metrics, cases=records)
        context.main_call(lambda: atomic_save(report, output/"test.json", json_file=True))
    finally:
        context.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, action="append", required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run(args.arm, args.inventory, args.manifest, args.output, args.device)
