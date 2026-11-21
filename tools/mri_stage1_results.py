"""Check the five paired VAL arms and quantify the effect of the historical online update."""
import argparse
import csv
import json
import math
from pathlib import Path

from vemgsbm.mri_compare import paired_comparison
from vemgsbm.mri_stage1 import ARMS
from vemgsbm.runtime import atomic_save


def summarize(folder):
    summaries = {arm: json.loads((folder / arm / "summary.json").read_text()) for arm in ARMS}
    reference = summaries["plain"]
    diagnostics = json.loads((folder / "diagnostics/summary.json").read_text())
    if (diagnostics["checkpoint_sha256"] != reference["checkpoint_sha256"]
            or diagnostics["seed"] != reference["seed"] or diagnostics["split"] != "val"
            or [r["case_id"] for r in diagnostics["cases"]] != reference["subset"]["case_ids"]):
        raise ValueError("Diagnostic and inference cohorts differ")
    rows = []
    for name, summary in summaries.items():
        sampler, eta, steps = ARMS[name]
        if (summary["split"] != "val" or summary["subset"]["fraction"] != 1
                or summary["checkpoint_sha256"] != reference["checkpoint_sha256"]
                or summary["seed"] != reference["seed"] or summary["sampling_sigma"] != eta
                or summary["drift_sampler"] != sampler or summary["steps"] != steps
                or summary["inference_variant"] != "endpoint_safe"):
            raise ValueError(f"Inconsistent sampler experiment {name}")
        paired = paired_comparison(reference, summary)
        rows.append(dict(arm=name, nfe=summary["drift_network_calls_per_tile_sample"],
                         mse=summary["all_mse"], psnr=summary["all_psnr_db"], ssim=summary["mean_slice_ssim"],
                         delta_psnr=paired["pooled_psnr_db"]["difference"],
                         delta_ssim=paired["mean_slice_ssim"]["difference"],
                         psnr_ci95=paired["pooled_psnr_db"]["ci95"], ssim_ci95=paired["mean_slice_ssim"]["ci95"]))
    delta = summaries["guided"]["all_mse"] - reference["all_mse"]
    whole = diagnostics["correction"]["whole"]
    if not math.isclose(delta, whole["delta_mse"], abs_tol=1.e-10, rel_tol=1.e-5):
        raise ValueError("Voxelwise correction decomposition disagrees with the full VAL summary")
    chart = diagnostics["chart"]
    regions = {}
    for name, stats in diagnostics["correction"].items():
        status = "无有效修正"
        if stats["change_sq"] > 0:
            if stats["dot"] <= 0:
                status = "与真实误差内积非正"
            elif stats["delta_mse"] > 0:
                status = "方向有正内积，当前幅度使误差增加"
            else:
                status = "当前幅度降低误差"
        regions[name] = dict(**stats, interpretation=status)
    return dict(selection="pooled_psnr", validation_cases=reference["subset"]["selected_cases"],
                validation_slices=reference["total_slices"], validation_pixels=reference["pixel_count"],
                historical_checkpoint_sha256=reference["checkpoint_sha256"],
                ranking=sorted(rows, key=lambda r: -r["psnr"]), chart=chart, correction=regions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    report = summarize(args.folder)
    atomic_save(report, args.folder / "stage1_decision.json", json_file=True)
    with (args.folder / "stage1_ranking.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["ranking"][0]))
        writer.writeheader()
        writer.writerows(report["ranking"])
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
