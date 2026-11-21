"""Compare the two MLS MRI models on a locked held-out cohort."""

import argparse
import json
import math
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_test(folder, manifest):
    summary = read_json(folder / "summary.json")
    if summary["subset"] != manifest or read_json(folder / "subset_manifest.json") != manifest:
        raise ValueError(f"Test cohort differs from the locked manifest in {folder}")
    cases = summary["cases"]
    if [row["case_id"] for row in cases] != manifest["case_ids"]:
        raise ValueError(f"Result cases differ from the locked manifest in {folder}")
    signature, case_ssim, checkpoints = [], [], set()
    for row in cases:
        metadata = read_json(folder / row["case_id"] / "metadata.json")
        metrics = row["metrics"]
        if (metadata["case_id"] != row["case_id"] or metadata["scale_xy"] != 4 or metadata["scale_z"] != 1
                or metadata["shape_zyx"][0] != row["slices"]
                or metrics["ssim_slice_count"] != row["slices"]
                or not math.isfinite(metrics["mean_slice_ssim"])):
            raise ValueError(f"Incomplete X4 paired MRI metrics for {row['case_id']} in {folder}")
        checkpoints.add(metadata["checkpoint_sha256"])
        case_ssim.append(metrics["mean_slice_ssim"])
        signature.append((metadata["input_fingerprint"], metadata["shape_zyx"], metadata["original_indices"],
                          metadata["normalization_scale"], metadata["normalization_offset"],
                          metadata["steps"], metadata["samples"], metadata["seed"]))
    if len(checkpoints) != 1:
        raise ValueError(f"Expected one evaluation checkpoint in {folder}")
    scores = {key: summary[key] for key in ("all_mse", "all_psnr_db", "mean_slice_ssim")}
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError(f"Non-finite evaluation summary in {folder}")
    return signature, case_ssim, {
        "result": str(folder), "checkpoint_sha256": checkpoints.pop(),
        "sampling_sigma": summary["sampling_sigma"], **scores,
        "mean_case_ssim": sum(case_ssim) / len(case_ssim),
    }


def compare(gsbm, momentum, manifest):
    g_signature, g_cases, g_scores = read_test(gsbm, manifest)
    m_signature, m_cases, m_scores = read_test(momentum, manifest)
    if g_signature != m_signature or g_scores["sampling_sigma"] != m_scores["sampling_sigma"]:
        raise ValueError("The MRI test inputs or sampling protocols differ")
    directions = [right - left for left, right in zip(g_cases, m_cases, strict=True)]
    return {
        "test_cases": manifest["selected_cases"], "test_slices": sum(row[1][0] for row in g_signature),
        "gsbm_mls": g_scores, "momentum_mls": m_scores,
        "momentum_minus_gsbm": {
            **{key: m_scores[key] - g_scores[key] for key in
               ("all_mse", "all_psnr_db", "mean_slice_ssim", "mean_case_ssim")},
            "cases_higher_ssim": sum(value > 0 for value in directions),
            "cases_lower_ssim": sum(value < 0 for value in directions),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsbm", type=Path, required=True)
    parser.add_argument("--momentum", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.gsbm, args.momentum, read_json(args.manifest))
    serial = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(serial)
    print(serial, end="")


if __name__ == "__main__":
    main()
