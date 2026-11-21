"""Select by whole-validation pooled PSNR and replay the frozen protocol on TEST."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from .mri_infer import checked_test_subset, infer_mri
from .mri_inventory import load_inventory
from .runtime import fingerprint, load_checkpoint, project_path


def read(path):
    return Path(path).read_text(encoding="utf-8")


def freeze(candidates, inventory, manifest, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    rows = []
    for checkpoint, result in candidates:
        checkpoint = Path(checkpoint).resolve()
        result = Path(result).resolve()
        report = json.loads(read(result / "summary.json"))
        cfg = load_checkpoint(checkpoint)["config"]
        data = cfg["data"]
        cases, _ = load_inventory(inventory, data, ("val", "test"))
        _, subset = checked_test_subset(cases["val"], 1, report["subset"]["seed"], None)
        if (report["split"] != "val" or report["subset"] != subset
                or report["checkpoint_sha256"] != fingerprint(checkpoint)
                or data.get("debug_case_limit") is not None
                or [r["case_id"] for r in report["cases"]] != subset["case_ids"]
                or not math.isfinite(report["all_psnr_db"])):
            raise ValueError("Freeze requires a complete whole-VAL result from the candidate checkpoint")
        for case, row in zip(cases["val"], report["cases"], strict=True):
            metadata = json.loads(read(result / case.id / "metadata.json"))
            if (metadata["input_fingerprint"] != case.fingerprint or row["slices"] != len(case.stems)
                    or metadata["checkpoint_sha256"] != report["checkpoint_sha256"]
                    or metadata["terminal_position_noise"]):
                raise ValueError("Candidate VAL protocol or data identity differs")
        fixed, locked = checked_test_subset(cases["test"], .15, 20261110, manifest)
        count = sum(len(c.stems) for c in fixed)
        pixels = sum(len(c.stems) * math.prod(c.lr_shape) * c.scale**2 for c in fixed)
        if (len(fixed), count, pixels) != (84, 14991, 3929800704):
            raise ValueError("Historical XY x4 test cohort mismatch")
        kwargs = dict(predictor=report["predictor"], steps=report["steps"], seed=report["seed"],
                      samples=report["samples"], inference_variant="endpoint_safe",
                      inference_sigma=report["sampling_sigma"])
        if cfg["mode"] == "gsbm":
            kwargs.update(drift_sampler=report.get("drift_sampler", "forward"),
                          disable_guidance=not bool(report.get("geometry_guidance")))
        if cfg["mode"] == "haar_bridge":
            kwargs["readout"] = report["readout"]
        rows.append(dict(checkpoint=str(checkpoint), checkpoint_sha256=report["checkpoint_sha256"],
                         validation=str(result), summary_sha256=fingerprint(result / "summary.json"),
                         pooled_psnr=report["all_psnr_db"], ssim=report["mean_slice_ssim"], inference=kwargs))
    # A tie keeps the first supplied candidate.
    best = max(rows, key=lambda r: r["pooled_psnr"])
    record = dict(selection_metric="whole_val_pooled_psnr", candidates=rows, selected=best,
                  inventory_sha256=fingerprint(inventory), manifest_sha256=fingerprint(manifest))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        stage = Path(tmp) / "frozen"
        stage.mkdir()
        for source, name, expected in [(best["checkpoint"], "model.pt", best["checkpoint_sha256"]),
                                       (inventory, "inventory.json.gz", record["inventory_sha256"]),
                                       (manifest, "manifest.json", record["manifest_sha256"])]:
            shutil.copyfile(source, stage / name)
            if fingerprint(stage / name) != expected:
                raise RuntimeError("Artifact changed during freeze")
        (stage / "freeze.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        os.rename(stage, destination)
    return record


def confirm(folder, output, device):
    folder = Path(folder)
    record = json.loads(read(folder / "freeze.json"))
    checkpoint, inventory, manifest = [folder / n for n in ("model.pt", "inventory.json.gz", "manifest.json")]
    if ([fingerprint(p) for p in (checkpoint, inventory, manifest)] !=
            [record["selected"]["checkpoint_sha256"], record["inventory_sha256"], record["manifest_sha256"]]):
        raise ValueError("Frozen model, data or manifest changed")
    cfg = load_checkpoint(checkpoint)["config"]
    roots = cfg["data"]["test"]
    result = infer_mri(checkpoint, project_path(roots["root_lq"]), output,
                       root_hq=project_path(roots["root_hq"]), split="test", case_inventory=inventory,
                       subset_manifest=manifest, subset_seed=20261110, test_fraction=.15,
                       save_volume_limit=0, device_name=device, resume_cases=True, **record["selected"]["inference"])
    summary = json.loads(read(result / "summary.json"))
    if (len(summary["cases"]), summary["total_slices"],
            sum(row["metrics"]["pixel_count"] for row in summary["cases"])) != (84, 14991, 3929800704):
        raise ValueError("Formal test output has an unexpected extent")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    start = commands.add_parser("freeze")
    start.add_argument("--candidate", nargs=2, action="append", required=True, metavar=("CHECKPOINT", "VAL_DIR"))
    for name in ("inventory", "manifest", "folder"):
        start.add_argument("--" + name, required=True)
    test = commands.add_parser("confirm")
    for name in ("folder", "output"):
        test.add_argument("--" + name, required=True)
    test.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.action == "freeze":
        print(json.dumps(freeze(args.candidate, args.inventory, args.manifest, args.folder), indent=2))
    else:
        confirm(args.folder, args.output, args.device)


if __name__ == "__main__":
    main()
