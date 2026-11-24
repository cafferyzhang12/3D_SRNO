"""Paired MRI PNG sequences with in-plane scaling and preserved slice positions."""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset


SIDECAR = ".mri_sequence_positions.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _haar_statistics(value, mean, std):
    if value.ndim != 5 or value.shape[1] != 16:
        raise ValueError("Haar whitening expects (B,16,Z,Y,X)")
    mean, std = (torch.as_tensor(v, device=value.device, dtype=value.dtype) for v in (mean, std))
    if mean.numel() != 16 or std.numel() != 16 or not (mean.isfinite().all() and std.isfinite().all()) or (std < 0).any():
        raise ValueError("Expected 16 finite means and nonnegative standard deviations")
    return mean.reshape(1,16,1,1,1), std.reshape(1,16,1,1,1)


def normalize_haar(residual, mean, std):
    mean, std = _haar_statistics(residual, mean, std)
    return (residual - mean) / (std + 1.e-6)


def denormalize_haar(residual, mean, std):
    mean, std = _haar_statistics(residual, mean, std)
    return residual * (std + 1.e-6) + mean


def natural_key(value):
    text = str(value)
    tokens = tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                   for part in re.split(r"(\d+)", text))
    return tokens, text.casefold(), text


@dataclass(frozen=True)
class MRICase:
    id: str
    stems: tuple[str, ...]
    lq: tuple[Path, ...]
    hq: tuple[Path, ...] | None
    indices: tuple[int, ...]
    lr_shape: tuple[int, int]
    bits: int
    normalization_scale: float
    normalization_offset: float
    channel_policy: str
    scale: int
    metadata: dict = field(default_factory=dict)
    fingerprint: str = ""

    @property
    def runs(self) -> tuple[tuple[int, int], ...]:
        """Half-open sequence-array intervals with consecutive original indices."""
        starts = [0]
        starts.extend(i for i in range(1, len(self.indices)) if self.indices[i] != self.indices[i - 1] + 1)
        return tuple(zip(starts, starts[1:] + [len(self.indices)]))


def _case_directories(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"MRI root does not exist: {root}")
    cases = {}
    folded = set()
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.is_symlink():
                raise ValueError(f"MRI symlinks are unsupported: {entry.path}")
            if entry.is_dir(follow_symlinks=False):
                if entry.name.casefold() in folded:
                    raise ValueError(f"Case IDs differing only by case are ambiguous: {entry.path}")
                folded.add(entry.name.casefold())
                cases[entry.name] = Path(entry.path)
            elif Path(entry.name).suffix.lower() in IMAGE_SUFFIXES:
                raise ValueError(f"Expected root/<case_id>/<slice>.png, found image directly in root: {entry.path}")
    if not cases:
        raise ValueError(f"No MRI case directories in {root}")
    return cases


def _frames(directory: Path, executor: ThreadPoolExecutor | None = None) -> tuple[list[Path], dict[str, tuple[int, int]]]:
    """Inspect one directory using the caller's pool, preserving result order."""
    def inspect_entry(entry):
        # is_dir may itself issue a stat on filesystems without directory type
        # hints, so keep both type inspection and the explicit stat in workers.
        if entry.is_symlink():
            raise ValueError(f"MRI symlinks are unsupported: {entry.path}")
        if entry.is_dir(follow_symlinks=False):
            raise ValueError(f"Nested directory inside MRI case: {entry.path}")
        suffix = Path(entry.name).suffix.lower()
        if suffix in IMAGE_SUFFIXES and suffix != ".png":
            raise ValueError(f"Expected prepared PNG slices, found {entry.path}")
        if suffix != ".png":
            return None
        stat = entry.stat(follow_symlinks=False)
        return Path(entry.path), (stat.st_size, stat.st_mtime_ns)

    paths = []
    stamps = {}
    folded = set()
    with os.scandir(directory) as entries:
        # Queue at most one case directory. This function is called only from
        # the indexing thread, before that same pool is used for PNG headers.
        results = executor.map(inspect_entry, entries) if executor is not None else map(inspect_entry, entries)
        for result in results:
            if result is None:
                continue
            path, stamp = result
            if path.stem.casefold() in folded:
                raise ValueError(f"Duplicate or ambiguous slice stem in {directory}: {path.stem}")
            folded.add(path.stem.casefold())
            stamps[str(path)] = stamp
            paths.append(path)
    return sorted(paths, key=lambda path: natural_key(path.name)), stamps


def _filename_indices(paths: Sequence[Path]) -> tuple[int, ...]:
    """Preserve numeric ordinal gaps when every stem shares one naming pattern."""
    matches = [re.fullmatch(r"(.*?)(\d+)(\D*)", path.stem) for path in paths]
    if matches and all(matches) and len({(match[1], match[3]) for match in matches}) == 1:
        values = tuple(int(match[2]) for match in matches)
        if len(set(values)) != len(values):
            raise ValueError("Ambiguous MRI numeric slice positions, e.g. slice1 and slice01")
        return tuple(value - values[0] for value in values)
    return tuple(range(len(paths)))


def _png_header(path: Path):
    with path.open("rb") as handle:
        header = handle.read(26)
    if len(header) != 26 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG header: {path}")
    bits = header[24]
    if bits not in {8, 16}:
        raise ValueError(f"MRI requires encoded uint8/uint16 PNG, got {bits} bits: {path}")
    return bits


def _read_raw(path: Path, channel_policy: str) -> tuple[np.ndarray, int]:
    bits = _png_header(path)
    with Image.open(path) as image:
        if image.mode not in {"L", "RGB", "I", "I;16", "I;16B"}:
            raise ValueError(f"Unsupported MRI PNG mode {image.mode!r}: {path}")
        if image.mode == "RGB" and bits != 8:
            raise ValueError(f"RGB 16-bit PNG cannot be decoded losslessly with Pillow: {path}")
        raw = np.array(image, copy=True)
    if raw.ndim == 3:
        if channel_policy != "identical_rgb_to_gray" or raw.shape[-1] != 3:
            raise ValueError(f"Expected a single grayscale MRI channel: {path}")
        if not (np.array_equal(raw[..., 0], raw[..., 1]) and np.array_equal(raw[..., 0], raw[..., 2])):
            raise ValueError(f"MRI RGB channels differ: {path}")
        raw = raw[..., 0]
    if raw.ndim != 2 or raw.dtype.kind not in "ui" or min(raw.shape) < 1:
        raise ValueError(f"Expected a nonempty integer grayscale plane: {path}")
    if raw.min() < 0 or raw.max() > 2 ** bits - 1:
        raise ValueError(f"Pixel values exceed encoded bit depth: {path}")
    return raw, bits


def read_plane(path, normalization_scale=None, normalization_offset=0.0,
               channel_policy="identical_rgb_to_gray", expected_bits=None) -> np.ndarray:
    """Decode one lossless grayscale plane with a fixed intensity transform."""
    if channel_policy not in {"native", "identical_rgb_to_gray"}:
        raise ValueError(f"Unknown channel policy {channel_policy!r}")
    raw, bits = _read_raw(Path(path), channel_policy)
    if expected_bits is not None and bits != expected_bits:
        raise ValueError(f"Image bit depth changed after indexing: {path}")
    scale = float(2 ** bits - 1 if normalization_scale is None else normalization_scale)
    offset = float(normalization_offset)
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(offset):
        raise ValueError("Normalization scale must be finite positive and offset must be finite")
    plane = (raw.astype(np.float32) - offset) / scale
    if not np.isfinite(plane).all() or plane.min() < 0 or plane.max() > 1:
        raise ValueError(f"Fixed normalization produced pixels outside [0,1]: {path}")
    return plane


def _preserved_indices(root_lq: Path, root_hq: Path | None):
    path = root_lq / SIDECAR
    shared_hq = False
    if not path.exists():
        if root_hq is not None and (root_hq / SIDECAR).exists():
            path = root_hq / SIDECAR
            shared_hq = True
        else:
            return None, {}
    if path.is_symlink():
        raise ValueError(f"MRI sidecar symlink is unsupported: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("version") != 1 or payload.get("kind") != "mri_zero_pair_quarantine"
            or (not shared_hq and Path(payload.get("root_lq", "")).resolve() != root_lq)
            or not isinstance(payload.get("patients"), dict)):
        raise ValueError(f"Invalid MRI sequence sidecar: {path}")
    if root_hq is not None:
        counterpart = root_hq / SIDECAR
        if (Path(payload.get("root_hq", "")).resolve() != root_hq or counterpart.is_symlink()
                or not counterpart.is_file() or json.loads(counterpart.read_text(encoding="utf-8")) != payload):
            raise ValueError(f"Unpaired or inconsistent MRI sidecars: {path}, {counterpart}")
    for case_id, record in payload["patients"].items():
        indices = record.get("indices")
        filenames = record.get("filenames")
        removed = record.get("removed_stems")
        if (not isinstance(indices, dict) or not indices or not isinstance(filenames, dict)
                or set(filenames) != set(indices) or not isinstance(removed, list)
                or len(set(removed)) != len(removed) or not set(removed) <= set(indices)
                or any(not isinstance(value, int) or isinstance(value, bool) for value in indices.values())
                or sorted(indices.values()) != list(range(len(indices)))
                or any(Path(name).name != name or Path(name).stem != stem for stem, name in filenames.items())):
            raise ValueError(f"Invalid preserved MRI slice positions: {path}, case {case_id}")
    return payload["patients"], {"sidecar": str(path), "shared_hq_sidecar": shared_hq,
                                  "sidecar_lq_root": payload["root_lq"]}


def index_cases(root_lq, root_hq=None, scale=4, cache_path=None,
                channel_policy="identical_rgb_to_gray", normalization_scale=None,
                normalization_offset=0.0, scan_workers=4, case_limit=None, progress=False,
                legacy_cache_path=None) -> list[MRICase]:
    """Check exact pairing and headers, reusing size/mtime-validated signatures.

    All cases share one bit depth and normalization. A case_limit explicitly
    limits full header checks after complete LQ/HQ directory-set validation.
    """
    if not isinstance(scale, int) or isinstance(scale, bool) or scale < 1:
        raise ValueError("MRI xy scale must be a positive integer")
    if channel_policy not in {"native", "identical_rgb_to_gray"}:
        raise ValueError(f"Unknown channel policy {channel_policy!r}")
    if not isinstance(scan_workers, int) or scan_workers < 1:
        raise ValueError("scan_workers must be a positive integer")
    if case_limit is not None and (not isinstance(case_limit, int) or case_limit < 1):
        raise ValueError("case_limit must be a positive integer")
    if (normalization_scale is not None and (not math.isfinite(normalization_scale) or normalization_scale <= 0)
            or not math.isfinite(normalization_offset)):
        raise ValueError("Invalid fixed intensity normalization")
    root_lq = Path(root_lq).expanduser().resolve()
    root_hq = Path(root_hq).expanduser().resolve() if root_hq is not None else None
    lq_dirs = _case_directories(root_lq)
    hq_dirs = _case_directories(root_hq) if root_hq is not None else None
    if hq_dirs is not None and set(lq_dirs) != set(hq_dirs):
        raise ValueError(f"MRI case mismatch between {root_lq} and {root_hq}")
    preserved, preservation_metadata = _preserved_indices(root_lq, root_hq)
    if preserved is not None and set(preserved) != set(lq_dirs):
        raise ValueError("MRI case inventory changed since zero-pair quarantine")
    cache = {}
    cache_path = Path(cache_path) if cache_path is not None else None
    identity = {"root_lq": str(root_lq), "root_hq": str(root_hq) if root_hq is not None else None,
                "channel_policy": channel_policy}
    if cache_path is not None and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("version") != 1 or payload.get("identity") != identity or not isinstance(payload.get("images"), dict):
            raise ValueError(f"MRI scan cache identity/schema mismatch: {cache_path}")
        cache = payload["images"]
    legacy_cache = {}
    if legacy_cache_path is not None and Path(legacy_cache_path).exists():
        payload = json.loads(Path(legacy_cache_path).read_text(encoding="utf-8"))
        if payload.get("version") != 1 or not isinstance(payload.get("entries"), dict):
            raise ValueError(f"Legacy MRI cache schema mismatch: {legacy_cache_path}")
        legacy_cache = payload["entries"]
    updated = {}

    def inspect(item):
        path, stamp = item
        previous = cache.get(str(path))
        legacy = legacy_cache.get(str(path))
        if (previous is None and legacy is not None and legacy.get("size") == stamp[0]
                and legacy.get("mtime_ns") == stamp[1] and legacy.get("channel_policy") == channel_policy):
            signature = legacy.get("signature")
            if (not isinstance(signature, list) or len(signature) != 2
                    or not isinstance(signature[0], list) or len(signature[0]) != 3 or signature[0][0] != 1):
                raise ValueError(f"Invalid legacy grayscale MRI signature: {path}")
            previous = {"stamp": list(stamp), "shape": signature[0][1:], "bits": signature[1]}
        if previous is not None and previous.get("stamp") == list(stamp):
            shape, bits = previous.get("shape"), previous.get("bits")
            if (not isinstance(shape, list) or len(shape) != 2 or bits not in {8, 16}
                    or any(not isinstance(value, int) or value < 1 for value in shape)):
                raise ValueError(f"Malformed cached MRI signature: {path}")
            return path, previous
        bits = _png_header(path)
        with Image.open(path) as image:
            mode, (width, height) = image.mode, image.size
        if mode == "RGB":
            raw, bits = _read_raw(path, channel_policy)
            height, width = raw.shape
        elif mode not in {"L", "I", "I;16", "I;16B"}:
            raise ValueError(f"Unsupported MRI PNG mode {mode!r}: {path}")
        return path, {"stamp": list(stamp), "shape": [height, width], "bits": bits}

    cases = []
    global_bits = None
    case_ids = sorted(lq_dirs, key=natural_key)
    if case_limit is not None:
        case_ids = case_ids[:case_limit]
    with ThreadPoolExecutor(max_workers=scan_workers) as pool:
        for number, case_id in enumerate(case_ids, start=1):
            lq, stamps = _frames(lq_dirs[case_id], executor=pool)
            ignored_lq = 0
            if preserved is not None and preservation_metadata["shared_hq_sidecar"]:
                removed = set(preserved[case_id]["removed_stems"])
                ignored_lq = sum(path.stem in removed for path in lq)
                lq = [path for path in lq if path.stem not in removed]
            hq = None
            if hq_dirs is not None:
                hq_files, hq_stamps = _frames(hq_dirs[case_id], executor=pool)
                hq_by_stem = {path.stem: path for path in hq_files}
                if {path.stem for path in lq} != set(hq_by_stem):
                    raise ValueError(f"MRI slice stem mismatch in case {case_id}")
                hq = [hq_by_stem[path.stem] for path in lq]
                stamps.update(hq_stamps)
            indices = _filename_indices(lq) if preserved is None else tuple(range(len(lq)))
            if preserved is not None:
                record = preserved[case_id]
                if {path.stem for path in lq} != set(record["indices"]) - set(record["removed_stems"]):
                    raise ValueError(f"MRI slice inventory changed since quarantine: {case_id}")
                if any(path.name != record["filenames"][path.stem] for path in lq):
                    raise ValueError(f"MRI slice filenames changed since quarantine: {case_id}")
                indices = tuple(record["indices"][path.stem] for path in lq)
                if any(a >= b for a, b in zip(indices, indices[1:])):
                    raise ValueError(f"MRI natural order disagrees with preserved slice positions: {case_id}")
            if not lq:
                if preserved is not None and set(preserved[case_id]["removed_stems"]) == set(preserved[case_id]["indices"]):
                    continue
                raise ValueError(f"No MRI PNG slices in {lq_dirs[case_id]}")
            paths = lq + (hq or [])
            signatures = list(pool.map(inspect, ((path, stamps[str(path)]) for path in paths)))
            for path, signature in signatures:
                updated[str(path)] = signature
            first = updated[str(lq[0])]
            shape, bits = tuple(first["shape"]), first["bits"]
            if global_bits is not None and bits != global_bits:
                raise ValueError("MRI cases have mixed bit depths. Use one documented common intensity representation.")
            global_bits = bits
            for path in lq:
                signature = updated[str(path)]
                if tuple(signature["shape"]) != shape or signature["bits"] != bits:
                    raise ValueError(f"Inconsistent MRI LQ shape/bit depth: {path}")
            for path in hq or []:
                signature = updated[str(path)]
                if tuple(signature["shape"]) != (shape[0] * scale, shape[1] * scale) or signature["bits"] != bits:
                    raise ValueError(f"MRI HQ must match LQ bit depth and exact xy scale {scale}: {path}")
            norm_scale = float(2 ** bits - 1 if normalization_scale is None else normalization_scale)
            identity_record = {"id": case_id, "indices": indices, "bits": bits, "scale": scale,
                               "normalization": [norm_scale, normalization_offset], "channel_policy": channel_policy,
                               "images": [[str(path), *stamps[str(path)]] for path in paths]}
            digest = hashlib.sha256(json.dumps(identity_record, sort_keys=True).encode("utf-8")).hexdigest()
            cases.append(MRICase(case_id, tuple(path.stem for path in lq), tuple(lq),
                                 tuple(hq) if hq is not None else None, indices, shape, bits,
                                 norm_scale, float(normalization_offset), channel_policy, scale,
                                 {**preservation_metadata, "ignored_lq_removed_count": ignored_lq,
                                  "indexed_case_limit": case_limit,
                                  "index_source": "quarantine_original_index" if preserved is not None
                                  else "filename_numeric_relative_or_sorted_index"}, digest))
            if progress and (number == 1 or number % 100 == 0 or number == len(case_ids)):
                print(f"MRI index {root_lq}: {number}/{len(case_ids)} cases, {len(updated)} image signatures", flush=True)
    if not cases:
        raise ValueError(f"No active MRI cases in {root_lq}")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            cache.update(updated)
            temporary.write_text(json.dumps({"version": 1, "identity": identity, "images": cache}), encoding="utf-8")
            os.replace(temporary, cache_path)
        finally:
            temporary.unlink(missing_ok=True)
    return cases


def data_fingerprint(cases: Sequence[MRICase]) -> str:
    """Hash indexed paths, stats, sequence positions, and intensity calibration.

    This reuses indexing metadata, with no additional filesystem reads. It is
    a reproducibility check rather than a cryptographic hash of image pixels.
    """
    if not cases or any(not case.fingerprint for case in cases):
        raise ValueError("Data fingerprint requires nonempty cases created by index_cases")
    payload = [(case.id, case.fingerprint) for case in sorted(cases, key=lambda case: natural_key(case.id))]
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()


def validate_case_splits(splits: Mapping[str, Sequence[MRICase]]) -> dict:
    seen = {}
    counts = {}
    bits = set()
    for split, cases in splits.items():
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown MRI split {split!r}")
        counts[split] = len(cases)
        for case in cases:
            key = case.id.casefold()
            if key in seen:
                raise ValueError(f"MRI patient leakage or duplicate ID: {case.id} in {seen[key]} and {split}")
            seen[key] = split
            bits.add((case.bits, case.normalization_scale, case.normalization_offset))
    if len(bits) != 1:
        raise ValueError("All MRI splits must use one bit depth and fixed normalization")
    return {"case_counts": counts, "independent_test": "test" in splits}


def load_case(case: MRICase) -> tuple[np.ndarray, np.ndarray | None]:
    """Load kept slices in recorded order. Use case.runs to avoid crossing gaps."""
    def load(paths, expected_shape):
        planes = []
        for path in paths:
            plane = read_plane(path, case.normalization_scale, case.normalization_offset,
                               case.channel_policy, case.bits)
            if plane.shape != expected_shape:
                raise ValueError(f"MRI image shape changed after indexing: {path}")
            planes.append(plane)
        return np.stack(planes)
    lr = load(case.lq, case.lr_shape)
    hr = load(case.hq, tuple(size * case.scale for size in case.lr_shape)) if case.hq is not None else None
    return lr, hr


def upsample_xy_patch(plane: np.ndarray, y0: int, x0: int, height: int, width: int, scale: int) -> torch.Tensor:
    """Bilinear HR patch with the same boundary values as full-plane resizing."""
    if (plane.ndim != 2 or min(height, width, scale) < 1 or min(y0, x0) < 0
            or y0 + height > plane.shape[0] or x0 + width > plane.shape[1]):
        raise ValueError("Requested XY patch must lie inside a 2D source plane")
    top, left = max(0, y0 - 1), max(0, x0 - 1)
    bottom, right = min(plane.shape[0], y0 + height + 1), min(plane.shape[1], x0 + width + 1)
    halo = torch.from_numpy(np.ascontiguousarray(plane[top:bottom, left:right])).unsqueeze(0).unsqueeze(0)
    enlarged = F.interpolate(halo, size=((bottom - top) * scale, (right - left) * scale),
                             mode="bilinear", align_corners=False)[0, 0]
    y, x = (y0 - top) * scale, (x0 - left) * scale
    return enlarged[y:y + height * scale, x:x + width * scale]


class PairedMRIPatchDataset(Dataset):
    def __init__(self, root_lq, root_hq, split, scale=4, patch_size=(5, 128, 128),
                 samples_per_volume=32, seed=42, cases=None, samples_per_epoch=None, **index_options):
        if split not in {"train", "val", "test"}:
            raise ValueError("MRI split must be train, val, or test")
        if (len(patch_size) != 3 or any(not isinstance(size, int) or size < 1 for size in patch_size)
                or not isinstance(scale, int) or scale < 1 or any(size % scale for size in patch_size[1:])):
            raise ValueError("MRI HR patch must be positive ZYX with Y/X divisible by scale")
        if not isinstance(samples_per_volume, int) or samples_per_volume < 1 or not isinstance(seed, int) or seed < 0:
            raise ValueError("samples_per_volume must be positive and seed must be nonnegative")
        if samples_per_epoch is not None and (not isinstance(samples_per_epoch, int) or samples_per_epoch < 1):
            raise ValueError("samples_per_epoch must be a positive integer")
        self.all_cases = list(cases) if cases is not None else index_cases(root_lq, root_hq, scale, **index_options)
        self.cases = []
        self.skipped_cases = []
        self.split, self.scale, self.patch_size = split, scale, tuple(patch_size)
        self.samples_per_volume, self.seed, self.epoch = samples_per_volume, seed, 0
        self.samples_per_epoch = samples_per_epoch
        self._windows = []
        depth, height, width = self.patch_size
        for case in self.all_cases:
            if case.hq is None or case.scale != scale:
                raise ValueError("MRI training requires paired cases with matching xy scale")
            if case.lr_shape[0] < height // scale or case.lr_shape[1] < width // scale:
                raise ValueError(f"MRI case {case.id} is smaller than the configured XY patch")
            starts = [start for low, high in case.runs for start in range(low, high - depth + 1)]
            if not starts:
                self.skipped_cases.append({"id": case.id, "reason": "short_contiguous_run",
                                           "required_depth": depth,
                                           "longest_run": max(high - low for low, high in case.runs)})
                continue
            self.cases.append(case)
            self._windows.append(starts)
        if not self.cases:
            raise ValueError(f"No MRI cases have a contiguous run of {depth} slices")
        if self.skipped_cases:
            print(f"MRI {split}: omitted {len(self.skipped_cases)} cases with short runs; see dataset.skipped_cases / preflight report", flush=True)
        self.set_epoch(0)

    def __len__(self):
        if self.samples_per_epoch is not None:
            return self.samples_per_epoch
        return len(self.cases) * self.samples_per_volume

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch if self.split == "train" else 0
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 919]))
        self._case_order = rng.permutation(len(self.cases))

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        case_index = (int(self._case_order[index % len(self.cases)]) if self.samples_per_epoch is not None
                      else index // self.samples_per_volume)
        case = self.cases[case_index]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        z0 = int(rng.choice(self._windows[case_index]))
        depth, height, width = self.patch_size
        lr_height, lr_width = height // self.scale, width // self.scale
        y0 = int(rng.integers(case.lr_shape[0] - lr_height + 1))
        x0 = int(rng.integers(case.lr_shape[1] - lr_width + 1))
        low_frames, high_frames, base_frames = [], [], []
        for position in range(z0, z0 + depth):
            low = read_plane(case.lq[position], case.normalization_scale, case.normalization_offset,
                             case.channel_policy, case.bits)
            high = read_plane(case.hq[position], case.normalization_scale, case.normalization_offset,
                              case.channel_policy, case.bits)
            if low.shape != case.lr_shape or high.shape != tuple(size * self.scale for size in case.lr_shape):
                raise ValueError(f"MRI image shape changed after indexing: {case.lq[position]}, {case.hq[position]}")
            low_frames.append(low[y0:y0 + lr_height, x0:x0 + lr_width])
            high_frames.append(high[y0 * self.scale:y0 * self.scale + height, x0 * self.scale:x0 * self.scale + width])
            base_frames.append(upsample_xy_patch(low, y0, x0, lr_height, lr_width, self.scale))
        lr = torch.from_numpy(np.stack(low_frames)).unsqueeze(0)
        hr = torch.from_numpy(np.stack(high_frames)).unsqueeze(0)
        base = torch.stack(base_frames).unsqueeze(0)
        return {"hr": hr, "lr": lr, "base": base, "mask": torch.zeros_like(hr), "id": case.id}
