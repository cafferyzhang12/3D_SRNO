"""Freeze every finalist from the same completed VAL study before releasing TEST jobs."""
import argparse
import json
import os
from pathlib import Path
import tempfile

from vemgsbm.mri_freeze import freeze
from vemgsbm.runtime import fingerprint

ARCHIVES = {
    "gsbm": "vem_gsbm/runs/mri/mri_x4_gsbm_20261111_001/bridge/best.pt",
    "momentum": "vem_gsbm/runs/mri/mri_x4_parallel_20261111_001_momentum/momentum/best.pt",
    "unguided_mls": "vem_gsbm_mls_20261113_inventory_fix/runs/mri/mri_x4_gsbm_mls_inventory_fix_20261113/bridge/best.pt",
    "momentum_mls": "vem_gsbm_mls_v2_20261120_f08f30129d6a/runs/mri/mri_x4_momentum_mls_v2_20261120/momentum/best.pt",
    "tdm": "vem_gsbm_momentum_tdm_20261116/runs/mri/mri_x4_momentum_tdm_20261116/momentum/best.pt",
}


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
    runs = []

    def add(label, method, seed, group, checkpoint, validation):
        runs.append(dict(label=label, method=method, training_seed=seed, group=group,
                         checkpoint=str(Path(checkpoint).resolve()), validation=str(Path(validation).resolve()),
                         freeze_dir=str(folder / label), output=str(folder.parent / "test" / label)))

    champion = next(m["config"] for m in study["ranking"] if m["config"]["reference"] != "shuffled")
    for model in study["ranking"]:
        seeds = sorted(r["arm"]["seed"] for r in model["runs"])
        if seeds != [42, 43, 44]:
            raise ValueError("Expected three prespecified training seeds per finalist")
        for r in model["runs"]:
            arm = r["arm"]
            method = "bridge" if arm["variant"] == "drift_only" else arm["variant"]
            if arm["reference"] != "identity":
                method += "_" + arm["reference"]
            seed = arm["seed"]
            add(f"{method}_s{seed}", method, seed, "new", r["checkpoint"], r["result"])
            runs[-1].update(arm=arm, confirm_archive=model["config"] == champion)
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
        if (existing["replicate_sha256"] != fingerprint(args.study) or existing["stage1_sha256"] !=
                fingerprint(args.stage1 / "stage1_decision.json") or [r["label"] for r in existing["runs"]] !=
                [r["label"] for r in runs]):
            raise ValueError("Existing freeze belongs to a different study")
        return
    with tempfile.TemporaryDirectory(dir=folder.parent) as tmp:
        staged = Path(tmp) / "freeze"
        staged.mkdir()
        for run in runs:
            selection = freeze([(run["checkpoint"], run["validation"])], inventory, manifest, staged / run["label"])
            run["validation_psnr"] = selection["selected"]["pooled_psnr"]
        record = dict(runs=runs, replicate_sha256=fingerprint(args.study),
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
