"""Convert a local microscopy volume to normalized float32 ZYX NPY."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np


@dataclass
class _Source:
    shape: tuple[int, ...]
    dtype: np.dtype
    read_plane: Callable[[int], np.ndarray]


def _natural_key(path: Path) -> list:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)]


@contextmanager
def _open_source(path: Path, key: str | None) -> Iterator[_Source]:
    with ExitStack() as stack:
        suffix = path.suffix.lower()
        if path.is_dir() or suffix in {".tif", ".tiff"}:
            try:
                import tifffile
            except ImportError as error:
                raise ImportError("TIFF preparation requires tifffile. Install the project's dependencies.") from error
            if path.is_dir():
                files = sorted(
                    (item for item in path.iterdir() if item.is_file() and item.suffix.lower() in {".tif", ".tiff"}),
                    key=_natural_key,
                )
                if not files:
                    raise ValueError(f"No .tif/.tiff planes found in {path}")
                first = tifffile.imread(files[0])
                if first.ndim != 2:
                    raise ValueError("A TIFF directory must contain one grayscale 2D plane per file")
                source_shape = (len(files), *first.shape)
                source_dtype = first.dtype

                def read_plane(index: int) -> np.ndarray:
                    plane = tifffile.imread(files[index])
                    if plane.shape != first.shape or plane.dtype != source_dtype:
                        raise ValueError(f"Inconsistent shape or dtype in {files[index]}")
                    return plane

                yield _Source(source_shape, source_dtype, read_plane)
                return
            handle = stack.enter_context(tifffile.TiffFile(path))
            if len(handle.series) != 1:
                raise ValueError("TIFF must contain one 3D grayscale series. Export the intended series first.")
            series = handle.series[0]
            if len(series.shape) != 3 or len(series.pages) != series.shape[0]:
                raise ValueError("TIFF must be a stack of 2D grayscale pages. Export channels/time points separately.")
            if series.pages[0].samplesperpixel != 1:
                raise ValueError("RGB/RGBA TIFF is unsupported. Export the EM intensity channel separately.")
            yield _Source(tuple(series.shape), series.dtype, lambda index: series.pages[index].asarray())
            return
        if suffix == ".npy":
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            stack.callback(array._mmap.close)
            yield _Source(tuple(array.shape), array.dtype, lambda index: array[index])
            return
        if suffix in {".h5", ".hdf5", ".hdf"}:
            if not key:
                raise ValueError("HDF5 input requires --key pointing to the raw intensity dataset")
            try:
                import h5py
            except ImportError as error:
                raise ImportError("HDF5 preparation requires h5py. Install the project's dependencies.") from error
            handle = stack.enter_context(h5py.File(path, "r"))
            if key not in handle or not isinstance(handle[key], h5py.Dataset):
                raise ValueError(f"HDF5 dataset {key!r} does not exist in {path}")
            array = handle[key]
            yield _Source(tuple(array.shape), array.dtype, lambda index: array[index])
            return
        raise ValueError("Input must be NPY, HDF5, a multipage TIFF, or a directory of TIFF planes")


def prepare_volume(
    input_path: str | Path,
    output_path: str | Path,
    axes: str,
    key: str | None = None,
    intensity_min: float | None = None,
    intensity_max: float | None = None,
    overwrite: bool = False,
) -> dict:
    """Stream source planes into a temporary file, publish only after validation.

    Explicit intensity bounds clip outliers. uint8/uint16 default to their full
    integer ranges. Floating-point inputs require both explicit bounds.
    """
    input_path = Path(input_path).expanduser().resolve(strict=True)
    output_path = Path(output_path).expanduser().resolve()
    axes = axes.upper()
    if len(axes) != 3 or set(axes) != set("ZYX"):
        raise ValueError("axes must be a permutation of ZYX describing source array dimensions")
    if output_path.suffix.lower() != ".npy":
        raise ValueError("Output path must end in .npy")
    if input_path == output_path:
        raise ValueError("Input and output must be different paths")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")
    if (intensity_min is None) != (intensity_max is None):
        raise ValueError("Specify --intensity-min and --intensity-max together")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp.npy")
    output = None
    try:
        with _open_source(input_path, key) as source:
            if len(source.shape) != 3 or min(source.shape) < 1:
                raise ValueError(f"Source must be nonempty 3D, got shape {source.shape}")
            if source.dtype.kind not in "uif":
                raise ValueError(f"Source must contain real intensity values, got dtype {source.dtype}")
            if intensity_min is None:
                if source.dtype.kind != "u" or source.dtype.itemsize not in (1, 2):
                    raise ValueError("Only uint8/uint16 have default scaling. Float and other inputs require explicit intensity bounds.")
                intensity_min = 0.0
                intensity_max = float(np.iinfo(source.dtype).max)
            if not math.isfinite(intensity_min) or not math.isfinite(intensity_max) or intensity_max <= intensity_min:
                raise ValueError("Intensity bounds must be finite with max > min")
            output_shape = tuple(source.shape[axes.index(axis)] for axis in "ZYX")
            output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=output_shape)
            output_axis = "ZYX".index(axes[0])
            plane_axes = [axis for axis in "ZYX" if axis != axes[0]]
            permutation = tuple(axes[1:].index(axis) for axis in plane_axes)
            clipped = 0
            for index in range(source.shape[0]):
                plane = np.array(source.read_plane(index), dtype=np.float64, copy=True)
                if plane.shape != source.shape[1:] or not np.isfinite(plane).all():
                    raise ValueError(f"Source plane {index} has invalid shape or nonfinite intensities")
                clipped += int(np.count_nonzero((plane < intensity_min) | (plane > intensity_max)))
                np.clip(plane, intensity_min, intensity_max, out=plane)
                plane -= intensity_min
                plane /= intensity_max - intensity_min
                target = [slice(None), slice(None), slice(None)]
                target[output_axis] = index
                output[tuple(target)] = plane.transpose(permutation)
            output.flush()
            output._mmap.close()
            output = None
        if overwrite:
            os.replace(temporary, output_path)
        else:
            # Hard-link creation fails atomically if another writer won the race.
            os.link(temporary, output_path)
            temporary.unlink()
        return {
            "input": str(input_path),
            "output": str(output_path),
            "source_axes": axes,
            "axes": "ZYX",
            "shape": output_shape,
            "dtype": "float32",
            "intensity_min": intensity_min,
            "intensity_max": intensity_max,
            "clipped_voxels": clipped,
        }
    finally:
        if output is not None:
            output._mmap.close()
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Local NPY/HDF5/TIFF or directory of 2D TIFF planes")
    parser.add_argument("--output", required=True, help="Prepared float32 ZYX .npy destination")
    parser.add_argument("--axes", required=True, help="Source array axis order, e.g. ZYX or XYZ")
    parser.add_argument("--key", help="Raw intensity dataset key for HDF5")
    parser.add_argument("--intensity-min", type=float)
    parser.add_argument("--intensity-max", type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = prepare_volume(args.input, args.output, args.axes, args.key, args.intensity_min, args.intensity_max, args.overwrite)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
