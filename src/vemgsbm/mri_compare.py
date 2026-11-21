"""Paired whole-case bootstrap for pooled PSNR, MSE and slice-weighted SSIM differences."""

import argparse
import json
from pathlib import Path

import numpy as np

from .runtime import atomic_save


def _paired_case_totals(reference, candidate):
    if reference["subset"] != candidate["subset"]:
        raise ValueError("Case manifests differ")
    if "split" in reference and "split" in candidate and reference["split"] != candidate["split"]:
        raise ValueError("Evaluation splits differ")
    if ("metric_protocol" in reference and "metric_protocol" in candidate
            and reference["metric_protocol"] != candidate["metric_protocol"]):
        raise ValueError("Metric protocols differ")
    keys = ("pixel_count", "ssim_slice_count", "evaluated_slices")
    metrics = []
    fingerprints_checked = True
    for summary in (reference, candidate):
        by_id = {case["case_id"]: case for case in summary["cases"]}
        ids = reference["subset"]["case_ids"]
        if len(by_id) != len(summary["cases"]) or len(set(ids)) != len(ids) or set(by_id) != set(ids):
            raise ValueError("Invalid case list for the locked cohort")
        metrics.append([by_id[case_id] for case_id in ids])
    for left, right in zip(*metrics):
        if any(left["metrics"][k] != right["metrics"][k] for k in keys):
            raise ValueError("Different slices or voxel counts in paired cases")
        fingerprint_a, fingerprint_b = left.get("input_fingerprint"), right.get("input_fingerprint")
        if fingerprint_a and fingerprint_b:
            if fingerprint_a != fingerprint_b:
                raise ValueError("Paired case input fingerprints differ")
        else:
            fingerprints_checked = False
    totals = []
    for arm in metrics:
        rows = []
        for case in arm:
            m = case["metrics"]
            ssim_sum = (m["mean_slice_ssim"] or 0) * m["ssim_slice_count"]
            rows.append([m["squared_error_sum"], m["pixel_count"], ssim_sum, m["ssim_slice_count"]])
        totals.append(np.asarray(rows, dtype=np.float64))
    if not len(totals[0]) or not all(np.isfinite(a).all() for a in totals):
        raise ValueError("Nonempty, finite paired case totals required")
    return np.stack(totals), fingerprints_checked


def paired_seed_comparison(pairs, seed=42, resamples=10000):
    """Pair patients across methods and seeds, then average the per-seed estimands."""
    if resamples < 2 or not pairs:
        raise ValueError("At least one compared pair and two bootstrap draws required")
    totals, checked = [], []
    for reference, candidate in pairs:
        if reference["subset"] != pairs[0][0]["subset"]:
            raise ValueError("Different case manifests across paired seeds")
        data, validated = _paired_case_totals(reference, candidate)
        totals.append(data)
        checked.append(validated)

    def differences(index):
        values = []
        with np.errstate(divide="ignore", invalid="ignore"):
            for pair in totals:
                ref, cand = (arm[index].sum(axis=1).T for arm in pair)
                values.append(np.stack((10 * np.log10(ref[0] / cand[0]), (cand[0] - ref[0]) / ref[1],
                                        (cand[2] - ref[2]) / ref[3]), axis=-1))
        return np.mean(values, axis=0)

    count = totals[0].shape[1]
    observed = differences(np.arange(count)[None])[0]
    rng = np.random.default_rng(seed)
    draws = np.concatenate([differences(rng.integers(count, size=(min(128, resamples - i), count)))
                            for i in range(0, resamples, 128)])
    result = {"number_of_cases": count, "seed": seed, "resamples": resamples, "level": .95,
              "difference": "candidate minus reference", "input_fingerprints_checked": all(checked),
              "paired_runs": len(pairs)}
    for i, name in enumerate(("pooled_psnr_db", "pooled_mse", "mean_slice_ssim")):
        ci = np.percentile(draws[:, i], [2.5, 97.5]).tolist() if np.isfinite(draws[:, i]).all() else None
        result[name] = {"difference": float(observed[i]) if np.isfinite(observed[i]) else None, "ci95": ci}
    return result


def paired_comparison(reference, candidate, seed=42, resamples=10000):
    return paired_seed_comparison([(reference, candidate)], seed=seed, resamples=resamples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Reference arm summary.json")
    parser.add_argument("candidate", type=Path, help="Candidate arm summary.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = paired_comparison(json.loads(args.reference.read_text(encoding="utf-8")),
                               json.loads(args.candidate.read_text(encoding="utf-8")), args.seed, args.resamples)
    if args.output:
        atomic_save(result, args.output, json_file=True)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
