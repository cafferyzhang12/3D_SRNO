"""Explicit specimen splits and reproducible axial super-resolution patches."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class VolumeRecord:
    id: str
    path: Path
    split: str
    specimen_id: str
    spacing: tuple[float, float, float]


def read_manifest(
    manifest: str | Path,
    split: str | None = None,
    require_isotropic: bool = True,
) -> list[VolumeRecord]:
    """Validate the complete manifest before selecting a requested split.

    Relative paths are resolved against the CSV's directory. Spacing is ZYX,
    expressed in the same physical unit for every record, conventionally nm.
    """
    manifest = Path(manifest).expanduser().resolve(strict=True)
    required = {"id", "path", "split", "specimen_id", "spacing_z", "spacing_y", "spacing_x"}
    records = []
    identifiers: set[str] = set()
    paths: set[Path] = set()
    specimen_splits: dict[str, str] = {}
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Manifest must have columns {sorted(required)}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Manifest contains duplicate column names")
        for line, row in enumerate(reader, start=2):
            if None in row or any(not (row.get(key) or "").strip() for key in required):
                raise ValueError(f"Manifest line {line} has missing fields or extra values")
            if any(key.startswith("roi_") and value for key, value in row.items()):
                raise ValueError("ROI columns are unsupported. Export separate crops with their source specimen_id.")
            identifier = row["id"].strip()
            split_name = row["split"].strip()
            if split_name not in {"train", "val", "test"}:
                raise ValueError(f"split must be train, val, or test at line {line}")
            specimen = row["specimen_id"].strip()
            path = Path(row["path"].strip()).expanduser()
            if not path.is_absolute():
                path = manifest.parent / path
            path = path.resolve(strict=True)
            if not path.is_file() or path.suffix.lower() != ".npy":
                raise ValueError(f"Prepared volume must be a .npy file: {path}")
            if identifier in identifiers:
                raise ValueError(f"Duplicate volume id: {identifier}")
            if path in paths:
                raise ValueError(f"Duplicate volume path: {path}")
            if specimen in specimen_splits and specimen_splits[specimen] != split_name:
                raise ValueError(f"Specimen {specimen!r} appears in multiple splits")
            try:
                spacing = tuple(float(row[f"spacing_{axis}"]) for axis in "zyx")
            except ValueError as error:
                raise ValueError(f"Invalid spacing at manifest line {line}") from error
            if not all(math.isfinite(value) and value > 0 for value in spacing):
                raise ValueError(f"Spacing must be finite and positive at line {line}")
            if require_isotropic and not all(math.isclose(value, spacing[0], rel_tol=1e-6) for value in spacing):
                raise ValueError(f"Synthetic HR training requires isotropic spacing: {identifier} {spacing}")
            identifiers.add(identifier)
            paths.add(path)
            specimen_splits[specimen] = split_name
            records.append(VolumeRecord(identifier, path, split_name, specimen, spacing))
    if not records:
        raise ValueError("Manifest contains no volumes")
    if split is not None:
        records = [record for record in records if record.split == split]
        if not records:
            raise ValueError(f"Manifest contains no volumes for split {split!r}")
    return records


def load_volume(path: str | Path) -> np.memmap:
    """Open a prepared float32 ZYX volume without materializing it in RAM.

    Validate its header here. Pixel finiteness and range are checked on each
    extracted patch, and on every voxel during the preparation command.
    """
    path = Path(path)
    if path.suffix.lower() != ".npy":
        raise ValueError(f"Expected a prepared .npy volume: {path}")
    volume = np.load(path, mmap_mode="r", allow_pickle=False)
    if volume.ndim != 3 or min(volume.shape) < 1:
        raise ValueError(f"Volume must be nonempty 3D ZYX, got {volume.shape}: {path}")
    if volume.dtype != np.dtype("float32"):
        raise ValueError(f"Volume must have float32 dtype, got {volume.dtype}: {path}")
    return volume


def make_observation(hr: torch.Tensor, scale: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep z=0,scale,...,Z-1 and linearly interpolate their exact coordinates.

    Accept ZYX, CZYX, or NCZYX tensors. Both outputs have the input shape.
    The mask is one at measured planes and zero at missing planes.
    """
    if not isinstance(scale, int) or isinstance(scale, bool) or scale < 1:
        raise ValueError("scale must be a positive integer")
    if hr.ndim not in (3, 4, 5) or not hr.is_floating_point():
        raise ValueError("hr must be a floating-point ZYX, CZYX, or NCZYX tensor")
    depth, height, width = hr.shape[-3:]
    if min(depth, height, width) < 1 or (depth - 1) % scale:
        raise ValueError("HR depth must satisfy (Z - 1) % scale == 0, with nonempty axes")
    anchors = hr[..., ::scale, :, :]
    base = F.interpolate(
        anchors.reshape(-1, 1, anchors.shape[-3], height, width),
        size=(depth, height, width),
        mode="trilinear",
        align_corners=True,
    ).reshape(hr.shape)
    # Preserve measured planes exactly, including floating-point roundoff.
    base[..., ::scale, :, :] = anchors
    mask = torch.zeros_like(hr)
    mask[..., ::scale, :, :] = 1
    return base, mask


class PatchDataset(Dataset):
    """Fixed samples per volume with independent, reproducible spatial crops."""

    def __init__(
        self,
        manifest: str | Path,
        split: str,
        patch_size: tuple[int, int, int] = (17, 64, 64),
        scale: int = 4,
        samples_per_volume: int = 32,
        seed: int = 42,
    ) -> None:
        if len(patch_size) != 3 or any(not isinstance(value, int) or value < 1 for value in patch_size):
            raise ValueError("patch_size must contain three positive integers in ZYX order")
        if not isinstance(scale, int) or isinstance(scale, bool) or scale < 1 or (patch_size[0] - 1) % scale:
            raise ValueError("scale must be positive and patch depth must satisfy (Z - 1) % scale == 0")
        if not isinstance(samples_per_volume, int) or samples_per_volume < 1:
            raise ValueError("samples_per_volume must be a positive integer")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.records = read_manifest(manifest, split=split)
        self.split = split
        self.patch_size = tuple(patch_size)
        self.scale = scale
        self.samples_per_volume = samples_per_volume
        self.seed = seed
        self.epoch = 0
        self._volumes: dict[Path, np.memmap] = {}
        for record in self.records:
            volume = load_volume(record.path)
            if any(actual < needed for actual, needed in zip(volume.shape, self.patch_size)):
                raise ValueError(f"Volume {record.id} shape {volume.shape} is smaller than patch {self.patch_size}")

    def __len__(self) -> int:
        return len(self.records) * self.samples_per_volume

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch if self.split == "train" else 0

    def __getstate__(self) -> dict:
        # Spawned workers reopen files instead of serializing the mapped data.
        state = self.__dict__.copy()
        state["_volumes"] = {}
        return state

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record = self.records[index // self.samples_per_volume]
        if record.path not in self._volumes:
            self._volumes[record.path] = load_volume(record.path)
        volume = self._volumes[record.path]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        starts = [int(rng.integers(actual - needed + 1)) for actual, needed in zip(volume.shape, self.patch_size)]
        slices = tuple(slice(start, start + size) for start, size in zip(starts, self.patch_size))
        patch = np.array(volume[slices], dtype=np.float32, order="C", copy=True)
        if not np.isfinite(patch).all() or patch.min() < 0 or patch.max() > 1:
            raise ValueError(f"Volume {record.id} contains nonfinite values or intensities outside [0,1]")
        hr = torch.from_numpy(patch).unsqueeze(0)
        base, mask = make_observation(hr, self.scale)
        return {"hr": hr, "base": base, "mask": mask, "id": record.id}
