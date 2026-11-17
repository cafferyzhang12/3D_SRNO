import csv
import itertools

import h5py
import numpy as np
import pytest
import tifffile
import torch

from vemgsbm.data import PatchDataset, load_volume, make_observation, read_manifest
from vemgsbm.prepare import prepare_volume
from vemgsbm.download_roi import download_roi


def write_manifest(tmp_path, rows):
    path = tmp_path / "volumes.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "path", "split", "specimen_id", "spacing_z", "spacing_y", "spacing_x"])
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("axes", ["".join(axes) for axes in itertools.permutations("ZYX")])
def test_prepare_axis_permutations_stream_source_planes(tmp_path, axes):
    expected = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    source = expected.transpose(tuple("ZYX".index(axis) for axis in axes))
    np.save(tmp_path / "raw.npy", source)
    result = prepare_volume(tmp_path / "raw.npy", tmp_path / "prepared.npy", axes)
    assert result["shape"] == (3, 4, 5)
    prepared = load_volume(tmp_path / "prepared.npy")
    assert isinstance(prepared, np.memmap)
    np.testing.assert_allclose(prepared, expected.astype(np.float32) / 255, atol=1e-7)


def test_compressed_tiff_and_hdf5_match_npy(tmp_path):
    source = np.arange(5 * 8 * 9, dtype=np.uint16).reshape(5, 8, 9)
    tifffile.imwrite(tmp_path / "raw.tif", source, photometric="minisblack", compression="deflate")
    with h5py.File(tmp_path / "raw.h5", "w") as handle:
        handle.create_dataset("volumes/raw", data=source, chunks=(1, 8, 9))
    for filename, key in [("raw.tif", None), ("raw.h5", "volumes/raw")]:
        output = tmp_path / f"{filename}.npy"
        prepare_volume(tmp_path / filename, output, "ZYX", key=key)
        np.testing.assert_allclose(np.load(output), source.astype(np.float32) / 65535, atol=1e-7)
    with pytest.raises(ValueError, match="requires --key"):
        prepare_volume(tmp_path / "raw.h5", tmp_path / "missing.npy", "ZYX")


def test_tiff_directory_uses_natural_numeric_order(tmp_path):
    directory = tmp_path / "planes"
    directory.mkdir()
    for index in [10, 2, 1]:
        tifffile.imwrite(directory / f"slice{index}.tif", np.full((4, 5), index, dtype=np.uint8))
    prepare_volume(directory, tmp_path / "prepared.npy", "ZYX")
    np.testing.assert_allclose(np.load(tmp_path / "prepared.npy")[:, 0, 0], np.array([1, 2, 10]) / 255)


def test_float_range_required_and_source_is_readonly(tmp_path):
    source = np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5)
    np.save(tmp_path / "raw.npy", source)
    with pytest.raises(ValueError, match="explicit intensity bounds"):
        prepare_volume(tmp_path / "raw.npy", tmp_path / "prepared.npy", "ZYX")
    result = prepare_volume(tmp_path / "raw.npy", tmp_path / "prepared.npy", "ZYX", intensity_min=10, intensity_max=40)
    np.testing.assert_array_equal(np.load(tmp_path / "raw.npy"), source)
    np.testing.assert_allclose(np.load(tmp_path / "prepared.npy"), np.clip((source - 10) / 30, 0, 1), atol=1e-7)
    assert result["clipped_voxels"] == 29


def test_failed_preparation_preserves_existing_output_and_cleans_temp(tmp_path):
    np.save(tmp_path / "prepared.npy", np.zeros((3, 4, 5), dtype=np.float32))
    previous = (tmp_path / "prepared.npy").read_bytes()
    broken = np.zeros((3, 4, 5), dtype=np.float32)
    broken[2, 0, 0] = np.nan
    np.save(tmp_path / "raw.npy", broken)
    with pytest.raises(FileExistsError):
        prepare_volume(tmp_path / "raw.npy", tmp_path / "prepared.npy", "ZYX")
    with pytest.raises(ValueError, match="nonfinite"):
        prepare_volume(tmp_path / "raw.npy", tmp_path / "prepared.npy", "ZYX", intensity_min=0, intensity_max=1, overwrite=True)
    assert (tmp_path / "prepared.npy").read_bytes() == previous
    assert not list(tmp_path.glob(".*.tmp.npy"))


@pytest.mark.parametrize("extra_dimensions", [(), (1,), (2, 1)])
def test_observation_interpolates_actual_anchor_coordinates(extra_dimensions):
    depth = torch.arange(17, dtype=torch.float32).reshape(17, 1, 1)
    linear = depth.expand(*extra_dimensions, 17, 4, 5) / 16
    base, mask = make_observation(linear, 4)
    torch.testing.assert_close(base, linear, rtol=0, atol=0)
    assert torch.equal(base[..., ::4, :, :], linear[..., ::4, :, :])
    assert torch.all(mask[..., ::4, :, :] == 1)
    assert mask[..., 1, :, :].count_nonzero() == 0
    with pytest.raises(ValueError, match="Z - 1"):
        make_observation(linear[..., :16, :, :], 4)


def test_manifest_checks_all_splits_and_resolves_relative_paths(tmp_path):
    volume = np.zeros((9, 12, 13), dtype=np.float32)
    np.save(tmp_path / "one.npy", volume)
    np.save(tmp_path / "two.npy", volume)
    manifest = write_manifest(tmp_path, [
        ["one", "one.npy", "train", "specimen_A", 5, 5, 5],
        ["two", "two.npy", "val", "specimen_A", 5, 5, 5],
    ])
    with pytest.raises(ValueError, match="multiple splits"):
        read_manifest(manifest, split="train")
    manifest = write_manifest(tmp_path, [
        ["one", "one.npy", "train", "specimen_A", 5, 5, 5],
        ["two", "one.npy", "train", "specimen_A", 5, 5, 5],
    ])
    with pytest.raises(ValueError, match="Duplicate volume path"):
        read_manifest(manifest)
    manifest = write_manifest(tmp_path, [["one", "one.npy", "train", "specimen_A", 40, 4, 4]])
    with pytest.raises(ValueError, match="isotropic"):
        read_manifest(manifest)
    records = read_manifest(manifest, require_isotropic=False)
    assert records[0].path == tmp_path / "one.npy"
    assert records[0].spacing == (40, 4, 4)


def test_patch_reproducibility_epoch_and_validation(tmp_path):
    rng = np.random.default_rng(21)
    for split in ["train", "val"]:
        np.save(tmp_path / f"{split}.npy", rng.random((17, 20, 21), dtype=np.float32))
    manifest = write_manifest(tmp_path, [
        ["a", "train.npy", "train", "specimen_A", 5, 5, 5],
        ["b", "val.npy", "val", "specimen_B", 5, 5, 5],
    ])
    options = dict(manifest=manifest, patch_size=(9, 8, 8), scale=4, samples_per_volume=3)
    training = PatchDataset(split="train", **options)
    first = training[1]
    torch.testing.assert_close(first["hr"], training[1]["hr"], rtol=0, atol=0)
    assert first["hr"].shape == (1, 9, 8, 8)
    assert first["id"] == "a"
    assert torch.equal(first["base"] * first["mask"], first["hr"] * first["mask"])
    training.set_epoch(1)
    assert not torch.equal(first["hr"], training[1]["hr"])
    training.set_epoch(0)
    assert torch.equal(first["hr"], training[1]["hr"])
    validation = PatchDataset(split="val", **options)
    before = validation[1]["hr"]
    validation.set_epoch(10)
    assert torch.equal(before, validation[1]["hr"])
    with pytest.raises(IndexError):
        training[len(training)]


def test_invalid_volume_header_and_patch_intensity(tmp_path):
    path = tmp_path / "volume.npy"
    np.save(path, np.zeros((9, 8, 8), dtype=np.uint16))
    with pytest.raises(ValueError, match="float32"):
        load_volume(path)
    np.save(path, np.zeros((9, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="3D ZYX"):
        load_volume(path)
    np.save(path, np.full((9, 8, 8), 2, dtype=np.float32))
    manifest = write_manifest(tmp_path, [["bad", "volume.npy", "train", "one", 5, 5, 5]])
    dataset = PatchDataset(manifest, "train", patch_size=(9, 8, 8), scale=4)
    with pytest.raises(ValueError, match="outside"):
        dataset[0]


@pytest.mark.parametrize("axes", ["".join(axes) for axes in itertools.permutations("ZYX")])
def test_roi_downloader_streaming_and_axes_without_network(tmp_path, monkeypatch, axes):
    from types import SimpleNamespace
    from vemgsbm import download_roi as module

    expected = np.arange(5 * 6 * 7, dtype=np.uint16).reshape(5, 6, 7)
    storage = expected.transpose(tuple("ZYX".index(axis) for axis in axes))
    reads = []

    class FakeArray:
        shape = storage.shape
        domain = SimpleNamespace(inclusive_min=(0, 0, 0), labels=tuple(axes))
        dtype = SimpleNamespace(numpy_dtype=np.dtype("uint16"))

        def __getitem__(self, slices):
            reads.append(slices)
            return SimpleNamespace(read=lambda: SimpleNamespace(result=lambda: storage[slices]))

    monkeypatch.setattr(module, "_open_source", lambda source, driver: FakeArray())
    output = tmp_path / "raw.npy"
    result = download_roi("https://example.test/raw/s0/", output, axes, (1, 4, 2, 6, 1, 6))
    np.testing.assert_array_equal(np.load(output), expected[1:4, 2:6, 1:6])
    assert len(reads) == 3
    assert result["normalized"] is False
    assert result["shape"] == (3, 4, 5)
    with pytest.raises(FileExistsError):
        download_roi("https://example.test/raw/s0/", output, axes, (1, 4, 2, 6, 1, 6))
    with pytest.raises(ValueError, match="outside source"):
        download_roi("https://example.test/raw/s0/", tmp_path / "outside.npy", axes, (0, 100, 0, 1, 0, 1))


def test_roi_download_limit_before_opening_remote_source(tmp_path, monkeypatch):
    from vemgsbm import download_roi as module

    def forbidden_open(*args):
        raise AssertionError("Invalid ROI must be rejected before any network access")

    monkeypatch.setattr(module, "_open_source", forbidden_open)
    with pytest.raises(ValueError, match="max-voxels"):
        download_roi("https://example.test/s0", tmp_path / "large.npy", "ZYX", (0, 1000, 0, 1000, 0, 1000))
    with pytest.raises(ValueError, match="nonempty"):
        download_roi("https://example.test/s0", tmp_path / "empty.npy", "ZYX", (0, 0, 0, 5, 0, 5))


def test_roi_failure_leaves_no_partial_published_file(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from vemgsbm import download_roi as module

    class BrokenArray:
        shape = (5, 6, 7)
        domain = SimpleNamespace(inclusive_min=(0, 0, 0), labels=("", "", ""))
        dtype = SimpleNamespace(numpy_dtype=np.dtype("uint8"))

        def __getitem__(self, slices):
            def result():
                if slices[0].start == 1:
                    raise IOError("Simulated missing remote chunk")
                return np.ones((1, 4, 5), dtype=np.uint8)
            return SimpleNamespace(read=lambda: SimpleNamespace(result=result))

    monkeypatch.setattr(module, "_open_source", lambda source, driver: BrokenArray())
    with pytest.raises(IOError, match="missing remote chunk"):
        download_roi("https://example.test/s0", tmp_path / "partial.npy", "ZYX", (0, 3, 0, 4, 0, 5))
    assert not (tmp_path / "partial.npy").exists()
    assert not list(tmp_path.glob(".*.tmp.npy"))


def test_manifest_split_enum(tmp_path):
    np.save(tmp_path / "data.npy", np.zeros((9, 8, 8), dtype=np.float32))
    manifest = write_manifest(tmp_path, [["a", "data.npy", "validation", "specimen_a", 5, 5, 5]])
    with pytest.raises(ValueError, match="train, val, or test"):
        read_manifest(manifest)
