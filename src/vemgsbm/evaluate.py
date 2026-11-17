"""Evaluate normalized image arrays, including missing slices separately."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity, adapted_rand_error, variation_of_information

from .data import load_volume
from .runtime import atomic_save


def image_metrics(prediction, reference, scale):
    if prediction.shape != reference.shape or scale < 2:
        raise ValueError("Prediction and reference must have matching ZYX shapes, scale >= 2")
    if (reference.shape[0] - 1) % scale:
        raise ValueError("Reference must end at an observed layer; crop to (Z-1)%scale=0")
    totals = {"all": [0.0, 0], "missing": [0.0, 0]}
    observed_max, missing_ssim = 0.0, []
    for z in range(reference.shape[0]):
        pred = np.asarray(prediction[z], dtype="float64")
        ref = np.asarray(reference[z], dtype="float64")
        if not np.isfinite(pred).all() or not np.isfinite(ref).all():
            raise ValueError("Evaluation images contain nonfinite intensities")
        if min(pred.min(), ref.min()) < 0 or max(pred.max(), ref.max()) > 1:
            raise ValueError("Evaluation requires matching normalized [0,1] images")
        error = np.square(pred - ref)
        totals["all"][0] += float(error.sum())
        totals["all"][1] += error.size
        if z % scale:
            totals["missing"][0] += float(error.sum())
            totals["missing"][1] += error.size
            width = min(7, min(ref.shape))
            width -= 1 - width % 2
            if width >= 3:
                missing_ssim.append(float(structural_similarity(ref, pred, data_range=1.0, win_size=width)))
        else:
            observed_max = max(observed_max, float(np.abs(pred - ref).max()))
    result = {"observed_max_abs_error": observed_max,
              "missing_xy_ssim": float(np.mean(missing_ssim)) if missing_ssim else None}
    for name, (squared_error, count) in totals.items():
        mse = squared_error / count if count else None
        result[name + "_mse"] = mse
        result[name + "_psnr_db"] = -10 * math.log10(mse) if mse and mse > 0 else None
        result[name + "_perfect_match"] = mse == 0
    # Orthogonal slice SSIM exposes cross-plane discontinuities.
    for axis, label in ((1, "xz"), (2, "yz")):
        scores = []
        for index in range(reference.shape[axis]):
            key = [slice(None)] * 3
            key[axis] = index
            ref = np.asarray(reference[tuple(key)], dtype="float64")
            pred = np.asarray(prediction[tuple(key)], dtype="float64")
            width = min(7, min(ref.shape))
            width -= 1 - width % 2
            if width >= 3:
                scores.append(float(structural_similarity(ref, pred, data_range=1, win_size=width)))
        result[label + "_ssim"] = float(np.mean(scores)) if scores else None
    return result


def label_metrics(prediction_path, reference_path, expected_shape=None):
    pred = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    ref = np.load(reference_path, mmap_mode="r", allow_pickle=False)
    if pred.shape != ref.shape or pred.ndim != 3 or pred.dtype.kind not in "ui" or ref.dtype.kind not in "ui":
        raise ValueError("Instance labels must be matching integer ZYX NPY volumes (0 = background)")
    if expected_shape is not None and tuple(ref.shape) != tuple(expected_shape):
        raise ValueError("Instance labels must use the reconstruction image grid and crop")
    if not np.any(ref > 0) or np.any(ref < 0) or np.any(pred < 0):
        raise ValueError("Labels must be nonnegative, with nonempty reference foreground")
    are, precision, recall = adapted_rand_error(ref, pred, ignore_labels=(0,))
    split, merge = variation_of_information(ref, pred, ignore_labels=(0,))
    return {"adapted_rand_error": float(are), "adapted_rand_precision": float(precision),
            "adapted_rand_recall": float(recall), "vi_split": float(split), "vi_merge": float(merge)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prediction-labels")
    parser.add_argument("--reference-labels")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("Choose a new evaluation output path")
    prediction, reference = load_volume(args.prediction), load_volume(args.reference)
    result = image_metrics(prediction, reference, args.scale)
    if bool(args.prediction_labels) != bool(args.reference_labels):
        parser.error("Provide both label arrays together")
    if args.prediction_labels:
        result["independent_instance_segmentation"] = label_metrics(
            args.prediction_labels, args.reference_labels, expected_shape=reference.shape)
    atomic_save(result, args.output, json_file=True)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
