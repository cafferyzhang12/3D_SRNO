"""Download an explicit public N5/Zarr ROI to a local raw ZYX NPY file."""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


def _open_source(source: str, driver: str):
    try:
        import tensorstore as ts
    except ImportError as error:
        raise ImportError('ROI download requires the optional dependency: python -m pip install -e ".[data]"') from error
    return ts.open(
        {
            "driver": driver,
            "kvstore": {"driver": "http", "base_url": source.rstrip("/") + "/"},
            "fill_missing_data_reads": False,
        },
        open=True,
        read=True,
        write=False,
        context={"cache_pool": {"total_bytes_limit": 64 * 1024 * 1024}},
    ).result()


def download_roi(
    source: str,
    output_path: str | Path,
    axes: str,
    roi: tuple[int, int, int, int, int, int],
    driver: str = "zarr",
    max_voxels: int = 32_000_000,
) -> dict:
    """Read only intersecting source chunks. ROI bounds are half-open ZYX.

    Store unnormalized source intensities in ZYX order. Run vemgsbm.prepare
    afterwards to apply the same validated scaling used for local inputs.
    """
    output_path = Path(output_path).expanduser().resolve()
    axes = axes.upper()
    if driver not in {"zarr", "n5"}:
        raise ValueError("driver must be zarr (v2) or n5")
    if urlparse(source).scheme not in {"https", "http"}:
        raise ValueError("source must be a public HTTP(S) array URL, including its scale path")
    if len(axes) != 3 or set(axes) != set("ZYX"):
        raise ValueError("axes must describe source storage dimensions as a permutation of ZYX")
    if len(roi) != 6 or any(not isinstance(value, int) for value in roi):
        raise ValueError("roi must be six integers: Z0 Z1 Y0 Y1 X0 X1")
    bounds = dict(zip("ZYX", zip(roi[::2], roi[1::2])))
    shape = tuple(stop - start for start, stop in bounds.values())
    if any(start < 0 or stop <= start for start, stop in bounds.values()):
        raise ValueError("ROI bounds must be nonnegative, nonempty, half-open intervals")
    if max_voxels < 1 or math.prod(shape) > max_voxels:
        raise ValueError("Requested ROI exceeds --max-voxels. Choose a smaller explicit crop.")
    if output_path.suffix.lower() != ".npy":
        raise ValueError("Output must end in .npy")
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    array = _open_source(source, driver)
    if len(array.shape) != 3 or any(array.domain.inclusive_min):
        raise ValueError("Source must be a zero-origin 3D grayscale array. Select one channel/time point first.")
    if any(bounds[axis][1] > size for axis, size in zip(axes, array.shape)):
        raise ValueError(f"ROI is outside source storage shape {array.shape} in {axes} order")
    dtype = np.dtype(array.dtype.numpy_dtype)
    if dtype.kind not in "uif":
        raise ValueError(f"Source must contain real intensity values, got {dtype}")
    labels = tuple(str(label).upper() for label in array.domain.labels)
    if all(labels) and labels != tuple(axes):
        raise ValueError(f"Declared axes {axes} disagree with source dimension labels {labels}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp.npy")
    output = None
    try:
        output = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=shape)
        permutation = tuple(axes.index(axis) for axis in "ZYX")
        for output_z, source_z in enumerate(range(*bounds["Z"])):
            slices = [slice(*bounds[axis]) for axis in axes]
            slices[axes.index("Z")] = slice(source_z, source_z + 1)
            plane = np.asarray(array[tuple(slices)].read().result()).transpose(permutation)[0]
            if plane.shape != shape[1:] or not np.isfinite(plane).all():
                raise ValueError(f"Invalid source intensities or shape at Z={source_z}")
            output[output_z] = plane
        output.flush()
        output._mmap.close()
        output = None
        os.link(temporary, output_path)
        temporary.unlink()
    finally:
        if output is not None:
            output._mmap.close()
        temporary.unlink(missing_ok=True)
    return {"source": source, "source_axes": axes, "roi_zyx_half_open": list(roi),
            "output": str(output_path), "axes": "ZYX", "shape": shape,
            "dtype": str(dtype), "normalized": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Public HTTP(S) URL of an individual raw array, e.g. .../s0/")
    parser.add_argument("--driver", choices=["zarr", "n5"], default="zarr", help="zarr means Zarr v2")
    parser.add_argument("--axes", required=True, help="Storage axis order, verified from metadata")
    parser.add_argument("--roi", type=int, nargs=6, required=True, metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"))
    parser.add_argument("--max-voxels", type=int, default=32_000_000)
    parser.add_argument("--output", required=True, help="Raw ZYX NPY file, normalized separately with prepare")
    args = parser.parse_args()
    print(json.dumps(download_roi(args.source, args.output, args.axes, tuple(args.roi), args.driver, args.max_voxels), indent=2))


if __name__ == "__main__":
    main()
