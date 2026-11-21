"""Rank whole-case results on one locked cohort and bootstrap paired differences."""
import argparse
import csv
import json
from pathlib import Path

from vemgsbm.mri_compare import paired_comparison
from vemgsbm.runtime import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", nargs=2, metavar=("LABEL", "SUMMARY"), required=True)
    parser.add_argument("--reference", required=True, help="Label to use for paired confidence intervals")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sources = dict(args.result)
    if len(sources) != len(args.result) or args.reference not in sources:
        parser.error("Use distinct labels and include the reference in --result")
    data = {key: json.loads(Path(path).read_text(encoding="utf-8")) for key, path in sources.items()}
    metrics = ("all_psnr_db", "mean_slice_ssim", "all_mse", "foreground_psnr_db", "hfen", "edge_mse",
               "z_difference_mse", "total_slices", "pixel_count")
    result = []
    for key, summary in data.items():
        comparison = paired_comparison(data[args.reference], summary)
        nfe = summary.get("network_calls_per_tile_sample", summary.get("drift_network_calls_per_tile_sample"))
        if nfe is None and "momentum" in summary.get("inference_variant", ""):
            nfe = summary["steps"]
        seconds = [r.get("evaluation_seconds") for r in summary["cases"]]
        row = dict(method=key, split=summary.get("split", "archived"), cases=len(summary["cases"]),
                   **{name: summary.get(name) for name in metrics}, reference=args.reference,
                   input_dimensionality=summary.get("input_dimensionality"),
                   pretraining=summary.get("pretraining", summary.get("external", {}).get("pretraining")),
                   nfe_per_tile=nfe, seed=summary.get("seed"),
                   case_evaluation_seconds=sum(seconds) if all(s is not None for s in seconds) else None,
                   checkpoint_sha256=summary.get("checkpoint_sha256"), summary_sha256=fingerprint(sources[key]),
                   source=sources[key])
        for name in ("pooled_psnr_db", "pooled_mse", "mean_slice_ssim"):
            stats = comparison[name]
            row["delta_" + name] = stats["difference"]
            lo, hi = stats["ci95"] or (None, None)
            row[name + "_low95"], row[name + "_high95"] = lo, hi
        result.append(row)
    result.sort(key=lambda row: -row["all_psnr_db"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["rank", *result[0]])
        writer.writeheader()
        writer.writerows(dict(rank=i, **row) for i, row in enumerate(result, 1))


if __name__ == "__main__":
    main()
