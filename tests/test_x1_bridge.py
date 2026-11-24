"""Endpoint math, actual TRAIN statistics, EMA resume, and deployed x1 reconstruction."""
import copy
import json

import numpy as np
import pytest
import torch
import yaml

from vemgsbm.bridge import endpoint_velocity, integrate_endpoint, sample_gaussian_bridge, sample_induced_coupling
from vemgsbm.haar_residual import HaarObjective, X1HaarResidualNetwork, XYMultiScaleResidual
from vemgsbm.haar_stats import ensure_haar_statistics
from vemgsbm.mri_data import normalize_haar, denormalize_haar, index_cases
from vemgsbm.momentum.model import HaarXY
from vemgsbm.runtime import read_config, sampling_noise, build_models, load_checkpoint
from vemgsbm.training_state import ModelEMA, warmup_cosine
from vemgsbm.train import run_training
from vemgsbm.mri_infer import infer_mri, _infer_case
from vemgsbm.spatial_atlas import TrainingMLSAtlas
from test_training import _config, _write_case


def _x1_ddp_worker(rank,rendezvous,config_path):
    from datetime import timedelta
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo",init_method=rendezvous,rank=rank,world_size=4,
                                        timeout=timedelta(seconds=120))
    try:
        run_training(config_path,device_name="cpu")
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(),reason="Gloo unavailable")
def test_four_rank_x1_whitening_and_ema(tmp_path):
    cfg = x1_config(tmp_path)
    cfg["train"].update(samples_per_epoch=8,batch_size=1)
    path = tmp_path/"x1_ddp.yaml"
    path.write_text(yaml.safe_dump(cfg))
    torch.multiprocessing.spawn(_x1_ddp_worker,
        args=((tmp_path/"store").resolve().as_uri(),str(path)),nprocs=4)
    saved = load_checkpoint(tmp_path/"run/bridge/last.pt")
    assert saved["world_size"] == 4 and saved["ema_updates"] == 2
    assert saved["band_statistics"]["slice_count"] == 6
    assert saved["epoch_history"][0]["train_items"] == 8


def test_whitening_roundtrip_and_black_bands():
    torch.manual_seed(13)
    codec = HaarXY()
    image = torch.randn(2,1,5,32,32)*.2
    residual = codec.encode(image)
    mean, std = torch.linspace(-.1,.1,16), torch.linspace(0.,.15,16)
    recovered = denormalize_haar(normalize_haar(residual,mean,std),mean,std)
    assert (codec.decode(recovered)-image).abs().max() < 1.e-6
    assert torch.isfinite(normalize_haar(torch.zeros_like(residual),torch.zeros(16),torch.zeros(16))).all()


@pytest.mark.parametrize("steps",[1,2,4,8,20])
@pytest.mark.parametrize("solver",["euler","heun"])
def test_endpoint_readout_safe_and_exact_calls(steps,solver):
    initial = torch.randn(2,16,2,3,3)
    truth = torch.randn_like(initial)
    times = []
    def predict(x,t):
        times.append(t)
        assert 0 <= t < 1
        return truth
    rng = torch.Generator().manual_seed(41)
    before = rng.get_state().clone()
    actual = integrate_endpoint(predict,initial,steps,solver=solver,sigma_infer=999,generator=rng)
    assert torch.equal(actual,truth)
    assert len(times) == (steps if solver == "euler" else 2*steps-1)
    assert times[0] == 0
    assert torch.equal(before,rng.get_state())  # ODE draws no integration noise.


def test_sde_noise_and_safe_training_boundaries():
    target = torch.randn(2,16,2,3,3)
    for t in (torch.tensor([1.e-4,1-1.e-4]),):
        state,t,initial = sample_gaussian_bridge(target,time=t)
        assert torch.isfinite(endpoint_velocity(state,target,t)).all()
        assert initial.std() > 0
    for t in (0.,1.):
        with pytest.raises(ValueError,match="safe interval"):
            sample_gaussian_bridge(target,time=t)
    with pytest.raises(ValueError,match="t < 1"):
        endpoint_velocity(target,target,1.)
    with pytest.raises(ValueError,match="sigma"):
        integrate_endpoint(lambda x,t:x,target,sampler_type="sde",sigma_bridge=.08,sigma_infer=.05)
    cfg = {"path":{"sigma":.08},"sampling":{"sigma":.08,"sampler_type":"sde","inference_sigma":.05}}
    with pytest.raises(ValueError,match="sigma_bridge"):
        sampling_noise(cfg)
    cfg["sampling"]["inference_sigma"] = .08
    assert sampling_noise(cfg) == .08
    cfg["sampling"]["sampler_type"] = "ode"
    assert sampling_noise(cfg) == 0
    rng = torch.Generator().manual_seed(42)
    before = rng.get_state().clone()
    integrate_endpoint(lambda x,t:x,target,4,sampler_type="sde",sigma_bridge=.08,generator=rng)
    assert not torch.equal(before,rng.get_state())


def test_xy_context_preserves_independent_z_planes():
    block = XYMultiScaleResidual(4)
    value = torch.randn(1,4,5,9,9)
    changed = value.clone()
    changed[:,:,0] += 10
    torch.testing.assert_close(block(value)[:,:,1:],block(changed)[:,:,1:])
    convolutions = [m for m in block.modules() if isinstance(m,torch.nn.Conv3d) and m.kernel_size != (1,1,1)]
    assert [m.dilation for m in convolutions] == [(1,1,1),(1,2,2),(1,4,4)]


def x1_config(tmp_path):
    cfg = _config(tmp_path)
    cfg.pop("representation")
    cfg.update(mode="haar_bridge",predict_target="x1",use_band_whitening=True,
               sigma_0=.03,sigma_bridge=.08,infer_sampler="ode",infer_steps=2,mls_reg_weight=0.,
               band_stats_cache=str(tmp_path/"stats.pt"),model={"widths":[4,6,8],"multiscale_xy":True})
    cfg["path"]["sigma"] = .08
    cfg["sampling"].update(sigma=.08,inference_sigma=0.,sampler_type="ode",readout="endpoint")
    cfg["train"].update(matching_time_min=1.e-4,rollout_steps=2,selection_metric="pooled_psnr",
                        warmup_epochs=1,schedule_epochs=4,ema_decay=.9,checkpoint_epochs=[1,2])
    return cfg


def test_statistics_fit_all_train_slices_and_cache_identity(tmp_path):
    cfg = x1_config(tmp_path)
    cases = index_cases(cfg["data"]["train"]["root_lq"],cfg["data"]["train"]["root_hq"],scale=4)
    stats = ensure_haar_statistics(cases,tmp_path/"stats.pt")
    assert stats["slice_count"] == 6 and stats["case_count"] == 2
    assert stats["coefficient_count_per_band"] == 24
    torch.testing.assert_close(stats["mean"],ensure_haar_statistics(cases,tmp_path/"stats.pt")["mean"])
    with pytest.raises(ValueError,match="TRAIN identity"):
        ensure_haar_statistics(cases[:1],tmp_path/"stats.pt")
    with pytest.raises(ValueError,match="only.*TRAIN"):
        ensure_haar_statistics(cases,tmp_path/"val.pt",split="val")


def test_x1_objective_prefix_detach_and_direction_removal():
    model = X1HaarResidualNetwork([4,6,8],use_band_whitening=False)
    assert not any(k.startswith(("direction.","drift.")) for k in model.state_dict())
    lr = torch.rand(16,1,3,4,4)
    base,hr = torch.zeros(16,1,3,16,16),torch.rand(16,1,3,16,16)
    calls = []
    model.register_forward_pre_hook(lambda m,args: calls.append((len(args[0]),torch.is_grad_enabled(),args[0].requires_grad)))
    terms = HaarObjective(model,{"train":{"rollout_steps":3}})(lr,base,hr)
    assert terms["rollout_items"] == 2
    assert calls == [(16,True,False),(2,False,False),(2,False,False),(2,True,False)]
    assert "backward_matching" not in terms
    terms["loss"].backward()
    assert model.endpoint.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    a,b = sample_induced_coupling(model,lr[:1],base[:1],steps=2)
    assert a.shape == b.shape and not b.requires_grad


@pytest.mark.parametrize("sampler",["ode","sde"])
def test_x1_tile_batch_seed_invariance(tmp_path,sampler):
    cfg = x1_config(tmp_path)
    source = tmp_path/"tiled"
    _write_case(source,"one",30,size=32,textured=True)
    case, = index_cases(source/"LQ/X4_png",source/"HQ_png",scale=4)
    cfg["data"]["patch_size"] = [3,16,16]
    cfg["sampling"].update(sampler_type=sampler,inference_sigma=0. if sampler == "ode" else .08)
    cfg["inference_variant"] = "endpoint_safe"
    model = X1HaarResidualNetwork([4,6,8],use_band_whitening=False).eval()
    torch.nn.init.normal_(model.endpoint.weight,std=.002)
    outputs = []
    for batch in (1,4):
        cfg["sampling"]["tile_batch"] = batch
        dest = tmp_path/f"tiles_{batch}"
        _infer_case(case,dest,model,None,cfg,torch.device("cpu"),17,3,2)
        outputs.append(np.load(dest/"mean.npy"))
    np.testing.assert_allclose(*outputs,atol=1.e-6,rtol=1.e-5)


def test_x1_ema_resume_and_whole_val(tmp_path):
    cfg = x1_config(tmp_path)
    path = tmp_path/"x1.yaml"
    path.write_text(yaml.safe_dump(cfg))
    run = run_training(path,device_name="cpu")
    first = load_checkpoint(run/"last.pt")
    assert first["format_version"] == 5 and first["ema_updates"] == 2
    model,_ = build_models(first,torch.device("cpu"))
    torch.testing.assert_close(model.endpoint.weight,first["ema_model"]["endpoint.weight"])
    roots = cfg["data"]["val"]
    infer_mri(run/"last.pt",roots["root_lq"],tmp_path/"val",roots["root_hq"],device_name="cpu",split="val")
    summary = json.loads((tmp_path/"val/summary.json").read_text())
    whole = json.loads((run/"whole_val/epoch_0001.json").read_text())
    assert summary["all_psnr_db"] == pytest.approx(whole["val_pooled_psnr_db"],abs=1.e-6)
    cfg["train"]["bridge_epochs"] = 2
    path.write_text(yaml.safe_dump(cfg))
    run_training(path,device_name="cpu",resume=run/"last.pt")
    resumed = load_checkpoint(run/"last.pt")
    assert resumed["ema_updates"] == 4 and resumed["scheduler"]["last_epoch"] == 4
    assert (run/"epoch_0002.pt").is_file()
    cfg["run_dir"] = str(tmp_path/"uninterrupted")
    path.write_text(yaml.safe_dump(cfg))
    uninterrupted = load_checkpoint(run_training(path,device_name="cpu")/"last.pt")
    for section in ("model","ema_model"):
        for key,value in resumed[section].items():
            torch.testing.assert_close(value,uninterrupted[section][key],rtol=0,atol=0)


def test_train_mls_quadratic_normals_and_gradients():
    torch.manual_seed(37)
    xy = torch.randn(400,2)*.3
    bank = torch.zeros(400,16)
    bank[:,:2] = xy
    bank[:,2] = .4*xy[:,0]**2+.2*xy[:,1]**2
    payload = dict(residuals=bank,case_indices=torch.arange(4).repeat_interleave(100),train_case_ids=list("abcd"))
    options = dict(dim=2,folds=2,centers=8,neighbors=80,query_chunk=16,loss_sites_per_patch=24)
    atlas = TrainingMLSAtlas(payload,options)
    truth = bank[:24].T.reshape(1,16,1,4,6)
    pred = (truth+torch.randn_like(truth)*.02).requires_grad_()
    loss = atlas.normal_loss(pred,truth,["a"])
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert pred.grad.abs().sum() > 0 and torch.isfinite(pred.grad).all()
    assert atlas.normal_loss(truth,truth,["a"]) == 0
    changed = copy.deepcopy(payload)
    changed["residuals"][payload["case_indices"] == 0] += 10
    second = TrainingMLSAtlas(changed,options)
    fold = atlas.folds["a"]
    lo,hi = atlas.offsets[fold:fold+2]
    torch.testing.assert_close(atlas.mean[lo:hi],second.mean[lo:hi])


def test_ema_and_lr_schedule_are_optimizer_step_based():
    model = torch.nn.Linear(1,1,bias=False)
    model.weight.data.fill_(0)
    ema = ModelEMA(model,.9)
    model.weight.data.fill_(1)
    ema.update(model)
    torch.testing.assert_close(ema.model.weight,torch.full_like(model.weight,.1))
    opt = torch.optim.SGD(model.parameters(),lr=.1)
    schedule = warmup_cosine(opt,dict(epochs=4,warmup_epochs=1,min_lr_ratio=.1),2)
    rates = []
    for _ in range(8):
        rates.append(opt.param_groups[0]["lr"])
        opt.step()
        schedule.step()
    assert rates[:3] == pytest.approx([.05,.1,.1])
    assert rates[-1] == pytest.approx(.01)


def test_mls_and_straightening_training_checkpoint_contract(tmp_path):
    cfg = x1_config(tmp_path)
    path = tmp_path/"teacher.yaml"
    path.write_text(yaml.safe_dump(cfg))
    teacher_path = run_training(path,device_name="cpu")/"best.pt"
    cfg["run_dir"] = str(tmp_path/"mls")
    cfg["mls_reg_weight"] = .05
    cfg["mls_regularizer"] = dict(bank=str(tmp_path/"bank.pt"),dim=2,folds=2,centers=2,
                                  neighbors=8,sites_per_case=16,patches_per_case=2,
                                  loss_sites_per_patch=8,query_chunk=4)
    path.write_text(yaml.safe_dump(cfg))
    saved = load_checkpoint(run_training(path,device_name="cpu")/"last.pt")
    assert "mls_normal_loss" in saved["epoch_history"][0]
    assert saved["training_mls"]["train_case_ids"] == ["train_00","train_01"]
    _,geometry = build_models(saved,torch.device("cpu"))
    assert geometry is None  # A regularized checkpoint never routes at deployment.
    cfg.update(run_dir=str(tmp_path/"straight"),mls_reg_weight=0.,
               straightening=dict(teacher_checkpoint=str(teacher_path),teacher_steps=2))
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match="Straightening trains straight paths"):
        read_config(path)
    cfg["sigma_bridge"] = cfg["path"]["sigma"] = cfg["sampling"]["sigma"] = 0.
    path.write_text(yaml.safe_dump(cfg))
    straight = load_checkpoint(run_training(path,device_name="cpu")/"last.pt")
    assert straight["epoch_history"][0]["loss"] > 0
    assert straight["config"]["sigma_bridge"] == 0
