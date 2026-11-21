"""Stagewise Haar MRI ablations selected exclusively by complete whole-case VAL."""
import argparse
from dataclasses import asdict, dataclass, replace
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from .runtime import atomic_save, fingerprint, load_checkpoint


@dataclass(frozen=True)
class Arm:
    variant: str
    dim: int = 2
    query: str = "endpoint"
    path: int = 0
    reference: str = "identity"
    shuffle: int = 0
    seed: int = 42
    readout: str = "endpoint"

    def run_id(self, study):
        # Both readouts of the same supervised B share one training run.
        train = asdict(self)
        train.pop("readout")
        key = hashlib.sha256(json.dumps(train, sort_keys=True).encode()).hexdigest()[:12]
        return f"{study}_{self.variant}_{key}"


STAGES = ("backbone", "dimensions", "controls", "reference", "replicate")


def plan(stage, previous):
    if stage == "backbone":
        return [Arm("direct"), Arm("bridge"), Arm("bridge", readout="integral"),
                Arm("drift_only", readout="integral")]
    if stage == "dimensions":
        return [Arm("mean")] + [Arm(method, dim=d) for method in ("pca", "mls") for d in (2, 4, 6, 8)]
    if stage == "controls":
        selected = Arm(**previous["dimensions"]["winners"]["mls"]["arm"])
        return [selected, replace(selected, path=1), replace(selected, query="state"),
                replace(selected, query="state", path=1), replace(selected, shuffle=1)]
    if stage == "reference":
        selected = Arm(**previous["controls"]["winners"]["physical_mls"]["arm"])
        return [replace(selected, reference=ref) for ref in ("identity", "geometry", "shuffled")]
    selected = [Arm(**previous[name]["winners"][key]["arm"]) for name, key in
                (("backbone", "direct"), ("backbone", "bridge"), ("dimensions", "mean"),
                 ("dimensions", "pca"), ("controls", "physical_mls"), ("reference", "geometry"),
                 ("reference", "shuffled"))]
    return [replace(arm, seed=seed) for arm in selected for seed in (42, 43, 44)]


def write_plan(stage, folder, study, previous):
    arms = plan(stage, previous)
    with (folder / f"{stage}.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        for arm in arms:
            writer.writerow([arm.run_id(study), *asdict(arm).values()])


def evaluate(arm, study):
    rid = arm.run_id(study)
    run = Path("runs/mri") / rid
    dest = Path("outputs/mri") / (rid + ("_integral_val" if arm.variant == "bridge" and arm.readout == "integral" else "_val"))
    summary = json.loads((dest / "summary.json").read_text(encoding="utf8"))
    checkpoint = run / "bridge" / "best.pt"
    saved = load_checkpoint(checkpoint)
    cfg = saved["config"]
    if (summary["split"] != "val" or summary["subset"]["fraction"] != 1
            or summary["checkpoint_sha256"] != fingerprint(checkpoint) or not math.isfinite(summary["all_psnr_db"])
            or cfg["train"]["seed"] != arm.seed or summary.get("readout") != arm.readout
            or cfg["train"]["selection_metric"] != "pooled_psnr" or cfg["data"].get("debug_case_limit") is not None):
        raise ValueError(f"Invalid VAL record for {rid}: {dest}")
    spatial = cfg.get("spatial_atlas", {})
    if arm.variant in {"pca", "mean", "mls"}:
        if (spatial.get("method"), spatial.get("dim"), spatial.get("query"), spatial.get("path_geometry"),
            spatial.get("reference"), spatial.get("shuffle_routing")) != (arm.variant, arm.dim, arm.query,
                                                                          bool(arm.path), arm.reference, bool(arm.shuffle)):
            raise ValueError(f"Spatial geometry does not match requested arm: {rid}")
    elif cfg["mode"] != ("haar_direct" if arm.variant == "direct" else "haar_bridge") or spatial:
        raise ValueError(f"Non-spatial model mismatch: {rid}")
    if arm.variant == "drift_only" and cfg["train"]["endpoint_weight"] != 0:
        raise ValueError(f"Unsupervised endpoint loss differs: {rid}")
    ids = [row["id"] for row in saved["data_identity"] if row["split"] == "val"]
    identity = {row["id"]: row["fingerprint"] for row in saved["data_identity"] if row["split"] == "val"}
    if (summary["subset"]["case_ids"] != ids or {row["case_id"]: row["input_fingerprint"]
                                                  for row in summary["cases"]} != identity):
        raise ValueError(f"VAL cohort differs from checkpoint: {rid}")
    return dict(arm=asdict(arm), run_id=rid, checkpoint=str(checkpoint), result=str(dest),
                summary_sha256=fingerprint(dest / "summary.json"), pooled_psnr=summary["all_psnr_db"],
                slice_ssim=summary["mean_slice_ssim"], edge_mse=summary["edge_mse"], pixels=summary["pixel_count"],
                val_identity=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest())


def select(stage, folder, study, previous):
    arms = plan(stage, previous)
    rows = [evaluate(arm, study) for arm in arms]
    identities = {(row["val_identity"], row["pixels"]) for row in rows}
    for earlier in previous.values():
        identities.update((r["val_identity"], r["pixels"]) for r in earlier["rows"])
    if len(identities) != 1:
        raise ValueError("Arms do not share the same whole-case VAL cohort")
    if stage == "controls":
        # Only the deliberately broken LR--HR correspondence is a negative control.
        groups = {"physical_mls": [r for r in rows if not r["arm"]["shuffle"]]}
    elif stage == "reference":
        groups = {r["arm"]["reference"]: [r] for r in rows}
    else:
        groups = {}
        for r in rows:
            key = "bridge" if r["arm"]["variant"] == "drift_only" else r["arm"]["variant"]
            groups.setdefault(key, []).append(r)
    winners = {key: max(group, key=lambda r: r["pooled_psnr"]) for key, group in groups.items()}
    report = {"stage": stage, "selection_metric": "whole-VAL pooled PSNR, dB", "rows": rows, "winners": winners}
    if stage == "replicate":
        # Only complete three-seed averages enter this ranking.
        by_method = {}
        for row in rows:
            key = json.dumps({k: v for k, v in row["arm"].items() if k != "seed"}, sort_keys=True)
            by_method.setdefault(key, []).append(row)
        ranking = []
        for cfg, group in by_method.items():
            if sorted(r["arm"]["seed"] for r in group) != [42, 43, 44]:
                raise ValueError("Incomplete three-seed validation")
            ranking.append({"config": json.loads(cfg), "mean_pooled_psnr": mean(r["pooled_psnr"] for r in group),
                            "mean_slice_ssim": mean(r["slice_ssim"] for r in group), "runs": group})
        report.pop("winners")
        report["ranking"] = sorted(ranking, key=lambda r: -r["mean_pooled_psnr"])
    atomic_save(report, folder / f"{stage}.json", json_file=True)
    previous[stage] = report
    i = STAGES.index(stage) + 1
    if i < len(STAGES):
        write_plan(STAGES[i], folder, study, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["init", "select"])
    parser.add_argument("--study", required=True)
    parser.add_argument("--stage", choices=STAGES, default="backbone")
    args = parser.parse_args()
    folder = Path("reports/mri/studies") / args.study
    folder.mkdir(parents=True, exist_ok=True)
    previous = {stage: json.loads((folder / f"{stage}.json").read_text(encoding="utf8"))
                for stage in STAGES[:STAGES.index(args.stage)]}
    if args.action == "init":
        write_plan("backbone", folder, args.study, {})
    else:
        select(args.stage, folder, args.study, previous)
        if args.stage == "backbone":
            (folder / "descriptor.txt").write_text(previous["backbone"]["winners"]["direct"]["checkpoint"] + "\n")


if __name__ == "__main__":
    main()
