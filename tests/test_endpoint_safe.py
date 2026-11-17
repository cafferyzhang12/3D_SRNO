"""Regression checks for terminal noise and clipping order."""
import numpy as np
from PIL import Image
import torch

from vemgsbm import mri_infer
from vemgsbm.mri_data import index_cases
from vemgsbm.network import integrate


def test_terminal_noise_switch_preserves_prefix_and_rng():
    x = torch.zeros(1, 1, 3, 4, 4)
    trace = []

    def drift(state, base, mask, time, direction):
        trace.append(state.clone())
        return torch.full_like(state, 0.2)

    generator = torch.Generator().manual_seed(123)
    original = integrate(drift, x, x, x, steps=4, generator=generator)
    old_rng = generator.get_state().clone()
    old_trace = trace.copy()
    trace.clear()
    generator.manual_seed(123)
    fixed = integrate(drift, x, x, x, steps=4, generator=generator, terminal_noise=False)
    assert all(torch.equal(a, b) for a, b in zip(old_trace, trace))
    assert torch.equal(old_rng, generator.get_state())
    torch.testing.assert_close(fixed, old_trace[-1] + .05)
    assert not torch.equal(original, fixed)


def test_clip_happens_after_overlapping_tiles_are_blended(tmp_path, monkeypatch):
    lq = tmp_path / "lq"
    (lq / "case").mkdir(parents=True)
    for i in range(3):
        Image.fromarray(np.zeros((2, 3), dtype=np.uint8)).save(lq / "case" / f"{i}.png")
    case = index_cases(lq, scale=4, scan_workers=1)[0]
    config = {"data": {"scale": 4, "patch_size": [3, 8, 8]}, "sampling": {"sigma": .1}, "mode": "gsbm"}

    def run(variant):
        values = iter((-0.8, 0.6))

        def tile(_, base, *args, **kwargs):
            assert kwargs["terminal_noise"] == (variant == "legacy")
            return torch.full_like(base, next(values))

        monkeypatch.setattr(mri_infer, "reconstruct", tile)
        folder = tmp_path / variant
        mri_infer._infer_case(case, folder, None, None, {**config, "inference_variant": variant},
                              "cpu", 42, 20, 1)
        return np.load(folder / "mean.npy")

    old, fixed = run("legacy"), run("endpoint_safe")
    hann = np.hanning(10)[1:-1]
    old_expected = .6 * hann[0] / (hann[4] + hann[0])
    np.testing.assert_allclose(old[:, :, 4], old_expected, atol=1e-7)
    np.testing.assert_array_equal(fixed[:, :, 4], 0.0)
    np.testing.assert_allclose(fixed[:, :, 8:], .6, atol=1e-7)
