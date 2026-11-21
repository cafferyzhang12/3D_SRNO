"""Freeze every finalist from the same completed VAL study before releasing TEST jobs."""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import torch
from vemgsbm.mri_compare import paired_comparison
from vemgsbm.mri_freeze import freeze
from vemgsbm.runtime import fingerprint

ARCHIVES = {
    "gsbm": "vem_gsbm/runs/mri/mri_x4_gsbm_20261111_001/bridge/best.pt",
    "momentum": "vem_gsbm/runs/mri/mri_x4_parallel_20261111_001_momentum/momentum/best.pt",
    "unguided_mls": "vem_gsbm_mls_20261113_inventory_fix/runs/mri/mri_x4_gsbm_mls_inventory_fix_20261113/bridge/best.pt",
    "momentum_mls": "vem_gsbm_mls_v2_20261120_f08f30129d6a/runs/mri/mri_x4_momentum_mls_v2_20261120/momentum/best.pt",
    "tdm": "vem_gsbm_momentum_tdm_20261116/runs/mri/mri_x4_momentum_tdm_20261116/momentum/best.pt",
}


def matched_geometry_validation(dimensions):
    """Compare D/E on the same seed, dimension, readout, routing and path on VAL."""
    rows = json.loads(dimensions.read_text(encoding="utf8"))["rows"]
    result = []
    for dim in (2, 4, 6, 8):
        summaries, evidence = [], {}
        for variant in ("pca", "mls"):
            match, = [r for r in rows if r["arm"]["dim"] == dim and r["arm"]["variant"] == variant]
            arm = match["arm"]
            path = Path(match["result"]) / "summary.json"
            s = json.loads(path.read_text(encoding="utf8"))
            if (s["split"] != "val" or s["subset"]["fraction"] != 1
                    or fingerprint(path) != match["summary_sha256"] or arm["seed"] != 42
                    or arm["path"] or arm["shuffle"] or arm["query"] != "endpoint"
                    or arm["readout"] != "endpoint" or arm["reference"] != "identity"):
                raise ValueError(f"Invalid matched {variant} validation evidence at dimension {dim}")
            summaries.append(s)
            evidence[variant + "_summary"] = str(path.resolve())
            evidence[variant + "_sha256"] = match["summary_sha256"]
        paired = paired_comparison(*summaries)
        result.append(dict(dim=dim, seed=42, **evidence, **{k: paired[k] for k in
                           ("pooled_psnr_db", "pooled_mse", "mean_slice_ssim")}))
    return result


def new_arms(study):
    """Freeze each checkpoint/readout once, preserving all of its validation roles."""
    eligible = [m for m in study["ranking"] if m["config"]["reference"] != "shuffled"]
    champion = max(eligible, key=lambda m: m["mean_pooled_psnr"])["config"]
    matched_models = study.get("matched_models", [])
    if sorted(m["config"]["variant"] for m in matched_models) != ["mls", "pca"]:
        raise ValueError("Matched TEST comparison needs one PCA and one MLS configuration")
    configs = {m["config"]["variant"]: m["config"] for m in matched_models}
    pca, mls = configs["pca"], configs["mls"]
    if (pca != {**mls, "variant": "pca"} or pca["dim"] not in (2, 4, 6, 8)
            or pca["query"] != "endpoint" or pca["path"] or pca["shuffle"]
            or pca["readout"] != "endpoint" or pca["reference"] != "identity"):
        raise ValueError("Frozen isolation controls must differ only in local surface order")
    by_source = {}
    for models, is_finalist in ((study["ranking"], True), (matched_models, False)):
        for model in models:
            seeds = sorted(r["arm"]["seed"] for r in model["runs"])
            if seeds != [42, 43, 44]:
                raise ValueError("Expected three prespecified training seeds per configuration")
            for result in model["runs"]:
                arm = result["arm"]
                if {k: v for k, v in arm.items() if k != "seed"} != model["config"]:
                    raise ValueError("Reported replicate is not the selected arm")
                method = "bridge" if arm["variant"] == "drift_only" else arm["variant"]
                if arm["reference"] != "identity":
                    method += "_" + arm["reference"]
                role = "finalist" if is_finalist else "matched_" + arm["variant"]
                key = (str(Path(result["checkpoint"]).resolve()), str(Path(result["result"]).resolve()))
                if key in by_source:
                    prior = by_source[key]
                    if prior["arm"] != arm or role in prior["roles"] or prior["validation_sha256"] != result["summary_sha256"]:
                        raise ValueError("Conflicting roles or validation records for one checkpoint/readout")
                    prior["roles"].append(role)
                else:
                    if not is_finalist:
                        method = role
                    by_source[key] = dict(method=method, training_seed=arm["seed"], checkpoint=key[0],
                                          label=f"{method}_s{arm['seed']}", validation=key[1],
                                          group="new" if is_finalist else "isolation", roles=[role], arm=arm,
                                          validation_sha256=result["summary_sha256"], validation_psnr=result["pooled_psnr"],
                                          confirm_archive=is_finalist and model["config"] == champion)
    return list(by_source.values())


def development_evidence(folder):
    """Compact selection-stage results and diagnostics from the chosen training epoch."""
    result = []
    keys = ("effective_neighbors", "quadratic_design_condition", "quadratic_fallback_fraction",
            "pca_to_mean_fallback_fraction", "path_relative_correction_rms", "path_accepted_steps")
    for stage in ("backbone", "dimensions", "controls", "reference"):
        path = folder / f"{stage}.json"
        for run in json.loads(path.read_text(encoding="utf8"))["rows"]:
            saved = torch.load(run["checkpoint"], map_location="cpu", weights_only=True, mmap=True)
            epoch = saved["epoch"]
            history = saved.get("epoch_history")
            if history is None:
                log = Path(run["checkpoint"]).parent / "metrics.jsonl"
                history = [json.loads(line) for line in log.read_text(encoding="utf8").splitlines()]
            selected, = [row for row in history if row["epoch"] == epoch]
            result.append(dict(stage=stage, run_id=run["run_id"], **run["arm"], selected_epoch=epoch,
                               val_psnr=run["pooled_psnr"], val_ssim=run["slice_ssim"], val_edge_mse=run["edge_mse"],
                               **{key: selected.get(key) for key in keys}, summary_sha256=run["summary_sha256"],
                               summary=str(Path(run["result"]).resolve() / "summary.json")))
            del saved
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ("study", "stage1", "legacy-results", "external-root", "archive-root", "history-root", "folder"):
        parser.add_argument("--" + k, required=True, type=Path)
    parser.add_argument("--external-prefix", required=True)
    args = parser.parse_args()
    folder = args.folder.resolve()
    snapshot = args.history_root
    history = snapshot / "runs/mri/mri_x4_gsbm_mls_v2_20261120"
    inventory = history / "mri_case_inventory.json.gz"
    manifest = snapshot / "outputs/mri/mri_x4_gsbm_mls_v2_20261120_test15/subset_manifest.json"
    study = json.loads(args.study.read_text())
    decision = json.loads((args.stage1 / "stage1_decision.json").read_text())
    dimensions = args.study.parent / "dimensions.json"
    runs = []

    def add(label, method, seed, group, checkpoint, validation):
        runs.append(dict(label=label, method=method, training_seed=seed, group=group,
                         checkpoint=str(Path(checkpoint).resolve()), validation=str(Path(validation).resolve()),
                         freeze_dir=str(folder / label), output=str(folder.parent / "test" / label)))

    for row in new_arms(study):
        add(row["label"], row["method"], row["training_seed"], row["group"], row["checkpoint"], row["validation"])
        runs[-1].update(row)
    if len({r["label"] for r in runs}) != len(runs):
        raise ValueError("Duplicate models in frozen selection")
    for method, path in ARCHIVES.items():
        add(method, method, None, "historical", args.archive_root / path, args.legacy_results / (args.stage1.name+"_"+method+"_val"))
    historical_weights = history / "bridge/best.pt"
    add("guided_epoch38", "guided_epoch38", None, "historical", historical_weights, args.stage1 / "guided")
    winner = decision["ranking"][0]["arm"]
    if winner != "guided":
        add("epoch38_" + winner, "epoch38_" + winner, None, "historical", historical_weights, args.stage1 / winner)
    add("bilinear", "bilinear", None, "bilinear", historical_weights, args.stage1 / "bilinear")
    for method in ("swinir", "res_srdiff"):
        run_id = args.external_prefix + "_" + method
        add(method, method, None, "external", args.external_root / "runs/mri" / run_id / "model.pt",
            args.external_root / "outputs/mri" / (run_id+"_val"))
    folder.parent.mkdir(parents=True, exist_ok=True)
    if folder.exists():
        existing = json.loads((folder / "matrix.json").read_text())
        if (existing.get("dimensions_sha256") != fingerprint(dimensions)
                or existing["replicate_sha256"] != fingerprint(args.study) or existing["stage1_sha256"] !=
                fingerprint(args.stage1 / "stage1_decision.json") or [r["label"] for r in existing["runs"]] !=
                [r["label"] for r in runs]):
            raise ValueError("Existing freeze belongs to a different study")
        return
    with tempfile.TemporaryDirectory(dir=folder.parent) as tmp:
        staged = Path(tmp) / "freeze"
        staged.mkdir()
        for run in runs:
            selection = freeze([(run["checkpoint"], run["validation"])], inventory, manifest, staged / run["label"])
            selected = selection["selected"]
            if run["group"] in {"new", "isolation"} and (run["validation_sha256"] != selected["summary_sha256"]
                    or not math.isclose(run["validation_psnr"], selected["pooled_psnr"], rel_tol=1e-12)):
                raise ValueError(f"Replicate VAL changed before freeze: {run['label']}")
            run["validation_psnr"] = selected["pooled_psnr"]
            run["validation_sha256"] = selected["summary_sha256"]
            run["freeze_sha256"] = fingerprint(staged / run["label"] / "freeze.json")
        record = dict(runs=runs, replicate_sha256=fingerprint(args.study),
                      champion_method=next(r["method"] for r in runs if r.get("confirm_archive")),
                      matched_validation=matched_geometry_validation(dimensions),
                      development=development_evidence(args.study.parent), sampler_validation=decision,
                      dimensions_sha256=fingerprint(dimensions),
                      stage1_sha256=fingerprint(args.stage1 / "stage1_decision.json"))
        (staged / "matrix.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        with (staged / "tests.tsv").open("w", encoding="utf-8") as f:
            for r in runs:
                f.write("\t".join(str(r[k]) for k in ("label","freeze_dir","output")) + "\n")
        with (staged / "archive.tsv").open("w", encoding="utf-8") as f:
            for r in runs:
                if r.get("confirm_archive"):
                    f.write(r["label"] + "\t" + r["freeze_dir"] + "\n")
        os.rename(staged, folder)


if __name__ == "__main__":
    main()
