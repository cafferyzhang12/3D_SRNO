import pytest
import torch

from vemgsbm.haar_residual import LRConditionEncoder
from vemgsbm.paths import optimize_path
from vemgsbm.spatial_atlas import SpatialAtlas


@pytest.fixture()
def payload():
    torch.manual_seed(19)
    encoder = LRConditionEncoder([4, 8, 8])
    per_case = 80
    descriptors = .1 * torch.randn(4 * per_case, 4)
    x, y = torch.randn(2, len(descriptors))
    residuals = torch.zeros(len(x), 16)
    residuals[:, :3] = torch.stack((x, y, .15*x.square() + .1*x*y), dim=1)
    return {"version": 1, "train_case_ids": ["a", "b", "c", "d"], "widths": [4, 8, 8],
            "descriptors": descriptors, "residuals": residuals, "encoder": encoder.state_dict(),
            "case_indices": torch.arange(4).repeat_interleave(per_case)}


def module(payload, **changes):
    return SpatialAtlas(payload, {"dim": 2, "folds": 2, "centers": 4, "candidates": 2, "neighbors": 80,
                                  "query_chunk": 7, **changes})


def test_case_fold_excludes_fit_and_retrieval(payload):
    a = module(payload)
    changed = dict(payload, residuals=payload["residuals"].clone())
    changed["residuals"][payload["case_indices"] == 0] += 20
    b = module(changed)
    lr = torch.randn(1, 1, 2, 4, 4)
    u, v = a.bind(lr, ["a"]), b.bind(lr, ["a"])
    raw = torch.randn(1, 16, *lr.shape[2:]) * .1
    torch.testing.assert_close(u.features(raw), v.features(raw))
    assert u.features(raw).shape == (1, 65, *lr.shape[2:])
    with pytest.raises(ValueError, match="TRAIN queries"):
        a.bind(lr, ["outside"])
    assert u.whiten(raw) is raw and u.unwhiten(raw) is raw
    raw.requires_grad_()
    u.features(raw).sum().backward()
    assert torch.isfinite(raw.grad).all()


def test_lr_reference_roundtrip_power_and_direction_control(payload):
    real, shuffled = (module(payload, reference=ref) for ref in ("geometry", "shuffled"))
    lr = torch.rand(1,1,2,4,4)
    u, v = real.bind(lr), shuffled.bind(lr)
    raw = torch.randn(1, 16, *lr.shape[2:])
    torch.testing.assert_close(u.unwhiten(u.whiten(raw)), raw, atol=2.e-6, rtol=2.e-6)
    covariance = u.factor @ u.factor.mT
    other = v.factor @ v.factor.mT
    spectrum = torch.linalg.eigvalsh(covariance)
    assert spectrum.min() >= .1 - 1.e-5 and spectrum.max() <= 4 + 1.e-5
    torch.testing.assert_close(spectrum.sum(-1), torch.full((32,), 16.))
    torch.testing.assert_close(torch.linalg.eigvalsh(other), spectrum, atol=2.e-5, rtol=2.e-5)
    assert not torch.allclose(other, covariance)
    noise = torch.randn(50000, 16) @ u.factor[0].T
    torch.testing.assert_close(torch.cov(noise.T), covariance[0], atol=.06, rtol=.06)


def test_moving_surface_subtracts_expansion_velocity(payload):
    atlas = module(payload, centers=1, candidates=1)
    atlas.mean.zero_()
    atlas.mean[:, 3] = .7
    atlas.basis.zero_()
    atlas.basis[:, :2] = torch.eye(2)
    atlas.bandwidth.fill_(1.)
    atlas.form.zero_()
    atlas.offset.zero_()
    atlas.linear.zero_()
    atlas.form[:, 0, 0, 2] = 1.
    bound = atlas.bind(torch.zeros(1,1,1,2,2))
    t = .4
    # m(t) = t*[q(t), 0, q(t)^2, .7, ...] with q(t)=t.
    mean = torch.zeros(1,16,1,2,2)
    velocity = torch.zeros_like(mean)
    mean[:,0], mean[:,2], mean[:,3] = t*t, t**3, .7*t
    velocity[:,0], velocity[:,2], velocity[:,3] = 2*t, 3*t*t, .7
    assert bound.mean_state_cost(mean,t).item() < 1.e-10
    dt = .01
    def cost(speed):
        return bound.kinetic_time(mean-speed*dt/2, mean+speed*dt/2, dt, t).item()
    assert cost(velocity) < 1.e-10
    velocity[:, 3] += 1.
    assert cost(velocity) == pytest.approx(1/16, abs=1.e-5)
    path, diagnostics = optimize_path(torch.zeros_like(mean), mean, torch.zeros_like(mean), bound,
                                      {"iterations":1,"quadrature":2,"backtrack_steps":2,
                                       "state_weight":.1,"metric_weight":.1})
    assert diagnostics["objective_after"] <= diagnostics["objective_before"]
    assert torch.isfinite(path.mean_knots).all()


def test_mean_path_cannot_cancel_opposite_chart_errors(payload):
    atlas = module(payload, centers=2, candidates=2, method="mean")
    atlas.mean.zero_()
    atlas.mean[::2, 3] = 1.
    atlas.mean[1::2, 3] = -1.
    bound = atlas.bind(torch.zeros(1,1,1,2,2))
    bound.weights.fill_(.5)
    mean = torch.zeros(1,16,1,2,2)
    assert bound.mean_state_cost(mean, .5).item() == pytest.approx(.25 / 16)


def test_unsupported_quadratic_reduces_to_pca(payload):
    mls = module(payload, neighbors=2)
    pca = module(payload, neighbors=2, method="pca")
    lr = torch.rand(1,1,2,4,4)
    a,b = mls.bind(lr), pca.bind(lr)
    x = torch.randn(1,16,2,4,4)
    assert a.diagnostics()["quadratic_fallback_fraction"].item() == pytest.approx(1.)
    torch.testing.assert_close(a.features(x)[:, :64], b.features(x)[:, :64])


def test_spatial_training_checkpoint_and_whole_case_deployment(tmp_path):
    import json
    from test_training import _config, _save_config
    from vemgsbm.mri_infer import infer_mri
    from vemgsbm.runtime import load_checkpoint
    from vemgsbm.train import run_training

    cfg = _config(tmp_path)
    cfg.pop("representation")
    cfg.update(mode="haar_direct", model={"widths": [4, 6, 8]})
    cfg["train"].update(selection_metric="pooled_psnr")
    path = tmp_path / "config.yaml"
    _save_config(cfg, path)
    direct = run_training(path, stage="bridge", device_name="cpu")
    cfg.update(mode="haar_bridge", run_dir=str(tmp_path / "geometry"))
    cfg["spatial_atlas"] = {"direct_checkpoint": str(direct / "best.pt"), "method": "mls",
                            "dim": 2, "path_geometry": True, "reference": "geometry", "folds": 2,
                            "sites_per_case": 12, "patches_per_case": 1, "centers": 3, "neighbors": 12}
    cfg["sampling"]["readout"] = "endpoint"
    _save_config(cfg, path)
    run = run_training(path, stage="bridge", device_name="cpu")
    saved = load_checkpoint(run / "best.pt")
    assert saved["format_version"] == 4
    assert sorted(saved["spatial_atlas"]["train_case_ids"]) == ["train_00","train_01"]
    rows = json.loads((run / "whole_val" / "epoch_0001.json").read_text())
    roots = cfg["data"]["val"]
    out = tmp_path / "deployment"
    infer_mri(run/"best.pt", roots["root_lq"], out, root_hq=roots["root_hq"], split="val", device_name="cpu")
    result = json.loads((out/"summary.json").read_text())
    assert result["all_psnr_db"] == pytest.approx(rows["val_pooled_psnr_db"])
    assert result["mean_slice_ssim"] == pytest.approx(rows["val_mean_slice_ssim"])
