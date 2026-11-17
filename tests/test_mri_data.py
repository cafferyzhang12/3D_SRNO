import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from vemgsbm.mri_data import (SIDECAR, PairedMRIPatchDataset, data_fingerprint,
                              index_cases, load_case, read_plane, upsample_xy_patch, validate_case_splits)
from vemgsbm.mri_preflight import preflight


def make_case(tmp_path, name="case1", count=7, scale=2, indices=None, bits=8, rgb=False):
    lq, hq = tmp_path / "LQ", tmp_path / "HQ"
    (lq / name).mkdir(parents=True, exist_ok=True)
    (hq / name).mkdir(parents=True, exist_ok=True)
    dtype = np.uint8 if bits == 8 else np.uint16
    for index in range(count) if indices is None else indices:
        low = np.full((8, 9), 20 + index, dtype=dtype)
        high = np.full((8 * scale, 9 * scale), 120 + index, dtype=dtype)
        if rgb:
            low, high = np.repeat(low[..., None], 3, axis=2), np.repeat(high[..., None], 3, axis=2)
        Image.fromarray(low).save(lq / name / f"slice{index}.png")
        Image.fromarray(high).save(hq / name / f"slice{index}.png")
    return lq, hq


def sidecar(lq, hq, case_id, original_count, removed, paired=True, recorded_lq=None):
    record = {"indices": {f"slice{i}": i for i in range(original_count)},
              "filenames": {f"slice{i}": f"slice{i}.png" for i in range(original_count)},
              "removed_stems": [f"slice{i}" for i in removed]}
    payload = {"version": 1, "kind": "mri_zero_pair_quarantine", "root_lq": str(recorded_lq or lq),
               "root_hq": str(hq), "patients": {case_id: record}}
    (hq / SIDECAR).write_text(json.dumps(payload), encoding="utf-8")
    if paired:
        (lq / SIDECAR).write_text(json.dumps(payload), encoding="utf-8")


def test_actual_lr_is_preserved_and_xy_only_base(tmp_path):
    lq, hq = make_case(tmp_path, indices=[1, 2, 3, 10, 11, 12])
    cases = index_cases(lq, hq, scale=2)
    assert cases[0].stems == ("slice1", "slice2", "slice3", "slice10", "slice11", "slice12")
    assert cases[0].runs == ((0, 3), (3, 6))
    lr, hr = load_case(cases[0])
    assert lr.shape == (6, 8, 9)
    assert hr.shape == (6, 16, 18)
    dataset = PairedMRIPatchDataset(lq, hq, "train", scale=2, patch_size=(3, 8, 10), samples_per_volume=4)
    sample = dataset[0]
    assert sample["hr"].shape == sample["base"].shape == (1, 3, 8, 10)
    assert sample["lr"].shape == (1, 3, 4, 5)
    assert torch.count_nonzero(sample["mask"]) == 0
    torch.testing.assert_close(sample["hr"] - sample["base"], torch.full_like(sample["hr"], 100 / 255))
    assert not torch.equal(sample["lr"], sample["hr"][..., ::2, ::2])
    torch.testing.assert_close(dataset[0]["hr"], sample["hr"], rtol=0, atol=0)


def test_paired_xy_crop_alignment_on_nonconstant_pixels(tmp_path):
    lq, hq = make_case(tmp_path, count=3)
    low = np.arange(8 * 9, dtype=np.uint8).reshape(8, 9)
    for index in range(3):
        Image.fromarray(low).save(lq / "case1" / f"slice{index}.png")
        Image.fromarray(np.repeat(np.repeat(low, 2, axis=0), 2, axis=1)).save(hq / "case1" / f"slice{index}.png")
    dataset = PairedMRIPatchDataset(lq, hq, "train", scale=2, patch_size=(3, 8, 10))
    sample = dataset[5]
    torch.testing.assert_close(sample["hr"][..., ::2, ::2], sample["lr"], rtol=0, atol=0)


@pytest.mark.parametrize("problem", ["missing_case", "missing_slice", "bad_scale", "different_bits", "ambiguous_stem"])
def test_pairing_failures_are_explicit(tmp_path, problem):
    lq, hq = make_case(tmp_path)
    if problem == "missing_case":
        (lq / "extra").mkdir()
    elif problem == "missing_slice":
        (hq / "case1" / "slice2.png").unlink()
    elif problem == "bad_scale":
        Image.fromarray(np.zeros((15, 18), dtype=np.uint8)).save(hq / "case1" / "slice2.png")
    elif problem == "different_bits":
        Image.fromarray(np.zeros((16, 18), dtype=np.uint16)).save(hq / "case1" / "slice2.png")
    else:
        Image.fromarray(np.zeros((8, 9), dtype=np.uint8)).save(lq / "case1" / "slice02.png")
        Image.fromarray(np.zeros((16, 18), dtype=np.uint8)).save(hq / "case1" / "slice02.png")
    with pytest.raises(ValueError):
        index_cases(lq, hq, scale=2)


def test_quarantine_preserves_gaps_and_excludes_short_case(tmp_path):
    lq, hq = make_case(tmp_path, indices=[0, 1, 3, 4, 5, 6])
    sidecar(lq, hq, "case1", 7, [2])
    case = index_cases(lq, hq, scale=2)[0]
    assert case.indices == (0, 1, 3, 4, 5, 6)
    assert case.runs == ((0, 2), (2, 6))
    dataset = PairedMRIPatchDataset(lq, hq, "train", scale=2, patch_size=(3, 8, 8), samples_per_volume=8)
    for index in range(len(dataset)):
        assert dataset[index]["lr"][0, 0, 0, 0] >= 23 / 255 - 1e-7
    (lq / "case1" / "slice3.png").unlink()
    (hq / "case1" / "slice3.png").unlink()
    with pytest.raises(ValueError, match="inventory"):
        index_cases(lq, hq, scale=2)


def test_shared_hq_sidecar_filters_only_documented_x2_extra_slices(tmp_path):
    lq, hq = make_case(tmp_path)
    (hq / "case1" / "slice2.png").unlink()
    sidecar(lq, hq, "case1", 7, [2], paired=False, recorded_lq=tmp_path / "original_X4_LQ")
    case = index_cases(lq, hq, scale=2)[0]
    assert case.indices == (0, 1, 3, 4, 5, 6)
    assert case.metadata["shared_hq_sidecar"] is True
    assert case.metadata["ignored_lq_removed_count"] == 1
    assert (lq / "case1" / "slice2.png").exists()
    Image.fromarray(np.zeros((8, 9), dtype=np.uint8)).save(lq / "case1" / "slice100.png")
    with pytest.raises(ValueError, match="stem mismatch"):
        index_cases(lq, hq, scale=2)


def test_reading_redundant_rgb_and_fixed_uint16_range(tmp_path):
    lq, hq = make_case(tmp_path / "rgb", rgb=True)
    case = index_cases(lq, hq, scale=2)[0]
    assert read_plane(case.lq[0]).shape == (8, 9)
    image = np.full((8, 9, 3), 20, dtype=np.uint8)
    image[0, 0, 2] = 21
    Image.fromarray(image).save(case.lq[0])
    with pytest.raises(ValueError, match="channels differ"):
        read_plane(case.lq[0])
    low16, high16 = make_case(tmp_path / "uint16", bits=16)
    case16 = index_cases(low16, high16, scale=2)[0]
    assert case16.normalization_scale == 65535
    assert np.isclose(read_plane(case16.lq[0])[0, 0], 20 / 65535)
    with pytest.raises(ValueError, match="outside"):
        read_plane(case16.lq[0], normalization_scale=10)


def test_legacy_cache_is_read_only_and_new_cache_is_invalidated(tmp_path, monkeypatch):
    from vemgsbm import mri_data
    lq, hq = make_case(tmp_path, count=3)
    entries = {}
    for root, shape in [(lq, [1, 8, 9]), (hq, [1, 16, 18])]:
        for path in (root / "case1").glob("*.png"):
            stat = path.stat()
            entries[str(path)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                  "channel_policy": "identical_rgb_to_gray", "signature": [shape, 8]}
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"version": 1, "entries": entries}), encoding="utf-8")
    before = legacy.read_bytes()
    original_header = mri_data._png_header
    monkeypatch.setattr(mri_data, "_png_header", lambda path: pytest.fail("Unchanged cached PNG should not be reopened"))
    cache = tmp_path / "new.json"
    first = index_cases(lq, hq, scale=2, cache_path=cache, legacy_cache_path=legacy)
    assert before == legacy.read_bytes()
    assert data_fingerprint(first) == data_fingerprint(index_cases(lq, hq, scale=2, cache_path=cache))
    monkeypatch.setattr(mri_data, "_png_header", original_header)
    path = lq / "case1" / "slice0.png"
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
    assert data_fingerprint(index_cases(lq, hq, scale=2, cache_path=cache)) != data_fingerprint(first)


def test_global_budget_balances_cases_validation_fixed_and_short_reported(tmp_path):
    lq, hq = make_case(tmp_path, "case1")
    make_case(tmp_path, "case2")
    make_case(tmp_path, "short", count=1)
    cases = index_cases(lq, hq, scale=2)
    dataset = PairedMRIPatchDataset(lq, hq, "val", scale=2, patch_size=(3, 8, 8),
                                    cases=cases, samples_per_epoch=5)
    assert len(dataset) == 5
    ids = [dataset[index]["id"] for index in range(len(dataset))]
    assert sorted([ids.count("case1"), ids.count("case2")]) == [2, 3]
    assert dataset.skipped_cases[0]["id"] == "short"
    before = dataset[0]["hr"]
    dataset.set_epoch(10)
    torch.testing.assert_close(before, dataset[0]["hr"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="leakage"):
        validate_case_splits({"train": cases, "val": cases})


def test_preflight_checks_complete_splits_and_rejects_debug_subset(tmp_path):
    train_lq, train_hq = make_case(tmp_path / "train", name="patient_a")
    val_lq, val_hq = make_case(tmp_path / "val", name="patient_b")
    config = {"data": {"scale": 2, "patch_size": [3, 8, 8],
                       "train": {"root_lq": str(train_lq), "root_hq": str(train_hq)},
                       "val": {"root_lq": str(val_lq), "root_hq": str(val_hq)}}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    report = preflight(path, tmp_path / "report.json", scan_workers=1)
    assert report["validation"]["independent_test"] is False
    assert report["splits"]["train"]["slice_count"] == 7
    config["data"]["debug_case_limit"] = 1
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="forbids debug"):
        preflight(path, tmp_path / "debug.json")


@pytest.mark.parametrize("crop", [(0, 0, 4, 5), (2, 3, 4, 5), (4, 4, 4, 5)])
def test_halo_bilinear_matches_full_slice_crop(crop):
    rng = np.random.default_rng(14)
    plane = rng.random((8, 9), dtype=np.float32)
    y0, x0, height, width = crop
    for scale in (2, 4):
        full = torch.nn.functional.interpolate(torch.from_numpy(plane)[None, None], scale_factor=scale,
                                               mode="bilinear", align_corners=False)[0, 0]
        expected = full[y0 * scale:(y0 + height) * scale, x0 * scale:(x0 + width) * scale]
        torch.testing.assert_close(upsample_xy_patch(plane, *crop, scale), expected, rtol=0, atol=1e-7)


def test_parallel_metadata_scan_reuses_one_pool_and_preserves_fingerprint(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager
    from threading import Lock
    import time
    from vemgsbm import mri_data

    lq, hq = make_case(tmp_path, "case10", indices=[1, 2, 3, 10, 11, 12])
    make_case(tmp_path, "case2", count=5)
    serial = index_cases(lq, hq, scale=2, scan_workers=1)
    original_scandir = os.scandir
    lock = Lock()
    active = 0
    concurrent_peak = 0
    pools_created = []

    class TrackedEntry:
        def __init__(self, entry):
            self.entry = entry
            self.name, self.path = entry.name, entry.path

        def is_symlink(self):
            return self.entry.is_symlink()

        def is_dir(self, *, follow_symlinks=True):
            return self.entry.is_dir(follow_symlinks=follow_symlinks)

        def stat(self, *, follow_symlinks=True):
            nonlocal active, concurrent_peak
            with lock:
                active += 1
                concurrent_peak = max(concurrent_peak, active)
            try:
                # Model a filesystem metadata wait without relying on wall-time
                # speed assertions, which are unreliable on loaded test hosts.
                time.sleep(0.01)
                return self.entry.stat(follow_symlinks=follow_symlinks)
            finally:
                with lock:
                    active -= 1

    @contextmanager
    def tracked_scandir(path):
        with original_scandir(path) as entries:
            yield (TrackedEntry(entry) for entry in entries)

    def tracked_pool(*args, **kwargs):
        pools_created.append(kwargs.get("max_workers"))
        return ThreadPoolExecutor(*args, **kwargs)

    monkeypatch.setattr(mri_data.os, "scandir", tracked_scandir)
    monkeypatch.setattr(mri_data, "ThreadPoolExecutor", tracked_pool)
    parallel = index_cases(lq, hq, scale=2, scan_workers=4)
    assert pools_created == [4]
    assert concurrent_peak > 1
    assert serial == parallel
    assert data_fingerprint(serial) == data_fingerprint(parallel)


@pytest.mark.parametrize("problem", ["nested_directory", "unexpected_format"])
def test_parallel_directory_validation_remains_strict(tmp_path, problem):
    lq, hq = make_case(tmp_path)
    if problem == "nested_directory":
        (lq / "case1" / "nested").mkdir()
        message = "Nested directory"
    else:
        Image.fromarray(np.zeros((8, 9), dtype=np.uint8)).save(lq / "case1" / "preview.jpg")
        message = "Expected prepared PNG"
    with pytest.raises(ValueError, match=message):
        index_cases(lq, hq, scale=2, scan_workers=4)
