"""Create a reproducible synthetic sparse test volume and its cropped reference."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import tempfile

import numpy as np

from .data import load_volume
from .runtime import atomic_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prepared isotropic HR NPY from the held-out test split")
    parser.add_argument("--scale", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.scale < 2:
        parser.error("scale must be >= 2")
    volume = load_volume(args.input)
    depth = (volume.shape[0] - 1) // args.scale * args.scale + 1
    if depth <= args.scale:
        parser.error("Need at least two measured slices")
    destination = Path(args.output_dir)
    if destination.exists():
        raise FileExistsError("Choose a new output directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shape = (depth, *volume.shape[1:])
    lock_path = destination.with_name(destination.name + ".lock")
    lock_file = lock_path.open("x")
    try:
        with lock_file, tempfile.TemporaryDirectory(prefix="vem-degrade-", dir=destination.parent) as temporary, ExitStack() as maps:
            temporary = Path(temporary)
            reference = np.lib.format.open_memmap(temporary / "reference.npy", mode="w+", dtype="float32", shape=shape)
            maps.callback(reference._mmap.close)
            sparse = np.lib.format.open_memmap(temporary / "sparse.npy", mode="w+", dtype="float32",
                                             shape=((depth - 1) // args.scale + 1, *shape[1:]))
            maps.callback(sparse._mmap.close)
            for z in range(depth):
                plane = np.asarray(volume[z])
                if not np.isfinite(plane).all() or plane.min() < 0 or plane.max() > 1:
                    raise ValueError("Input must be finite and normalized [0,1]")
                reference[z] = plane
                if z % args.scale == 0:
                    sparse[z // args.scale] = plane
            reference.flush()
            sparse.flush()
            maps.close()
            atomic_save({"source": str(Path(args.input).resolve()), "original_shape_zyx": list(volume.shape),
                         "reference_shape_zyx": list(shape), "scale_z": args.scale,
                         "operator": "noiseless_slice_selection", "first_observed_index": 0,
                         "last_observed_index": depth - 1}, temporary / "degradation.json", json_file=True)
            if destination.exists():
                raise FileExistsError("Output directory appeared during processing")
            os.rename(temporary, destination)
    finally:
        lock_path.unlink(missing_ok=True)
    print(destination.resolve())


if __name__ == "__main__":
    main()
