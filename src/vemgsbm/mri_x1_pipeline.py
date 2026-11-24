"""Compute-node preparation and evidence gates for the x1 MRI study."""
import argparse
import copy
import json
import math
from pathlib import Path
import re

import yaml

from .runtime import atomic_save, load_checkpoint, project_path, read_config


def late_stability(rows, window=5, max_drop=.15):
    """Only late regressions are unstable; early positive learning is expected."""
    recent = rows[-window:]
    values = [r["val_pooled_psnr_db"] for r in recent]
    drops = [max(0., a-b) for a,b in zip(values,values[1:])]
    finite = len(recent) == window and all(math.isfinite(v) for v in values)
    return finite and max(drops,default=0.) < max_drop, max(drops,default=0.)


def write_config(path,cfg):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    text = yaml.safe_dump(cfg,sort_keys=False)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Existing config differs: {path}")
    else:
        with path.open("x",encoding="utf-8") as stream:
            stream.write(text)


def prepare(template,legacy_checkpoint,inventory,ablation,run_id):
    from .mri_inventory import load_inventory
    cfg = read_config(template)
    old = load_checkpoint(legacy_checkpoint)
    if old["epoch"] != 24 or old["config"]["mode"] != "haar_bridge" or old["config"]["train"].get("endpoint_weight",0) <= 0:
        raise ValueError("The initial ablation must use supervised B epoch 24")
    evidence = json.loads(Path(ablation).read_text())
    if [r["nfe"] for r in evidence["rows"]] != [20,8,4,2,1] or any(
            r["cases"] != 10 or r["slices"] != 595 or not math.isfinite(r["psnr"]) for r in evidence["rows"]):
        raise ValueError("Incomplete legacy ODE ablation")
    cfg["data"] = copy.deepcopy(old["config"]["data"])
    cases,_ = load_inventory(inventory,cfg["data"],("train","val"))
    if (len(cases["train"]),sum(len(c.lq) for c in cases["train"]),len(cases["val"]),sum(len(c.lq) for c in cases["val"])) != (3919,292657,10,595):
        raise ValueError("Expected the complete historical TRAIN and VAL cohorts")
    steps = evidence["suggested_steps"]
    if steps not in (4,8):
        raise ValueError("Invalid suggested inference budget")
    cfg["infer_steps"] = cfg["sampling"]["steps"] = steps
    cfg["run_dir"] = f"runs/mri/{run_id}"
    cfg["band_stats_cache"] = f"reports/mri/{run_id}/train_haar_statistics.pt"
    cfg["mls_regularizer"]["bank"] = f"reports/mri/{run_id}/train_mls_residuals.pt"
    cfg["mls_reg_weight"] = 0.
    cfg["train"]["bridge_epochs"] = cfg["train"]["epochs"]
    path = project_path(cfg["run_dir"])/"experiment.yaml"
    write_config(path,cfg)
    smoke = copy.deepcopy(cfg)
    smoke["run_dir"] += "_smoke"
    smoke["train"].update(epochs=2,bridge_epochs=2,samples_per_epoch=4096,val_samples_per_epoch=256,
                          minimum_epochs=2,whole_val_patience=2,checkpoint_epochs=[1,2])
    write_config(path.with_name("smoke.yaml"),smoke)
    print(path,flush=True)


def check_smoke(config):
    cfg = read_config(config)
    folder = project_path(cfg["run_dir"])/"bridge"
    rows = [json.loads(line) for line in (folder/"metrics.jsonl").read_text().splitlines()]
    if len(rows) != 2 or [r["epoch"] for r in rows] != [1,2]:
        raise ValueError("Two completed smoke epochs required")
    for row in rows:
        if row["world_size"] != 4 or row["global_batch_size"] != 256 or row["val_case_count"] != 10:
            raise ValueError("Smoke must use four ranks, batch 64/rank and all VAL cases")
        if not all(math.isfinite(row[k]) for k in ("loss","grad_norm","val_pooled_psnr_db")):
            raise ValueError("Nonfinite smoke training")
        if row["rollout_items"] != 8:
            raise ValueError("Expected one eighth rollout batch per GPU")
    atomic_save(dict(passed=True,epochs=rows),folder/"smoke_gate.json",json_file=True)


def report(config,validation):
    cfg = read_config(config)
    rows = [json.loads(line) for line in (project_path(cfg["run_dir"])/"bridge/metrics.jsonl").read_text().splitlines()]
    summary = json.loads((Path(validation)/"summary.json").read_text())
    values = [r["val_pooled_psnr_db"] for r in rows]
    changes = [abs(a-b) for a,b in zip(values,values[1:])]
    completed = bool(rows and rows[-1]["epoch"] >= 40)
    recent = rows[-3:]
    metric_gate = completed and all(r["val_pooled_psnr_db"] >= 32.10 and r["val_mean_slice_ssim"] >= .900 for r in recent)
    late_stable, late_drop = late_stability(rows)
    stable = completed and late_stable
    result = dict(completed_40_epochs=completed,metric_gate=metric_gate,stability_gate=stable,
                  eligible_for_mls=metric_gate and stable,max_adjacent_psnr_change=max(changes,default=0.),
                  late_window_epochs=5,max_late_psnr_drop=late_drop,
                  milestones=[r for r in rows if r["epoch"] in (15,24,40)],
                  whole_val_psnr=summary["all_psnr_db"],whole_val_ssim=summary["mean_slice_ssim"],
                  final_test_released=False,reason="Stage 4 also requires matched TRAIN-finetuned external baselines and frozen cohorts")
    atomic_save(result,project_path(cfg["run_dir"])/"core_gate.json",json_file=True)
    print(json.dumps(result),flush=True)


def prepare_sr(template,legacy_checkpoint,inventory,run_id,statistics_cache=None):
    """Resolve historical data paths, then freeze the SR experiment configuration."""
    from .haar_stats import statistics_identity
    from .mri_inventory import load_inventory
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}",run_id):
        raise ValueError("Invalid run ID")
    cfg = read_config(template)
    if cfg["train"].get("objective") != "sr_reconstruction" or cfg.get("mls_reg_weight",0) <= 0:
        raise ValueError("SR+MLS preparation requires the reconstruction objective and positive MLS weight")
    old = load_checkpoint(legacy_checkpoint)
    cfg["data"] = copy.deepcopy(old["config"]["data"])
    if cfg["data"].get("debug_case_limit") is not None:
        raise ValueError("Formal SR training cannot use a debug cohort")
    cases,_ = load_inventory(inventory,cfg["data"],("train","val"))
    extent = (len(cases["train"]),sum(len(c.lq) for c in cases["train"]),
              len(cases["val"]),sum(len(c.lq) for c in cases["val"]))
    if extent != (3919,292657,10,595):
        raise ValueError("Expected the complete historical TRAIN and VAL cohorts")
    cfg["run_dir"] = f"runs/mri/{run_id}"
    cfg["band_stats_cache"] = f"reports/mri/{run_id}/train_haar_statistics.pt"
    cfg["mls_regularizer"]["bank"] = f"reports/mri/{run_id}/train_mls_residuals.pt"
    source = statistics_cache or old["config"].get("band_stats_cache")
    if source and project_path(source).is_file():
        payload = load_checkpoint(project_path(source))
        if payload.get("identity") != statistics_identity(cases["train"]):
            if statistics_cache:
                raise ValueError("Requested statistics cache has a different full TRAIN identity")
        else:
            cfg["band_stats_cache"] = str(project_path(source))
            print(f"Reusing identity-verified full TRAIN statistics: {cfg['band_stats_cache']}",flush=True)
    elif statistics_cache:
        raise FileNotFoundError(statistics_cache)
    cfg["train"]["bridge_epochs"] = cfg["train"]["epochs"]
    path = project_path(cfg["run_dir"])/"experiment.yaml"
    write_config(path,cfg)
    smoke = copy.deepcopy(cfg)
    smoke["run_dir"] += "_smoke"
    smoke["train"].update(epochs=1,bridge_epochs=1,samples_per_epoch=1024,val_samples_per_epoch=256,
                          minimum_epochs=1,whole_val_patience=1,checkpoint_epochs=[1])
    write_config(path.with_name("smoke.yaml"),smoke)
    print(path,flush=True)


def prebuild_mls(config,inventory,device="cuda"):
    """Fit TRAIN statistics and MLS before any NCCL process group starts."""
    import time
    from .datasets import training_data
    from .haar_residual import make_haar_model
    from .haar_stats import ensure_haar_statistics
    from .runtime import setup
    from .spatial_atlas import TrainingMLSAtlas, prepare_training_mls
    cfg = read_config(config)
    begin = time.perf_counter()
    device = setup(cfg["train"]["seed"],device,cfg["train"].get("threads",4))
    train,_,_,_ = training_data(cfg,inventory)
    stats = ensure_haar_statistics(train.all_cases,project_path(cfg["band_stats_cache"]),device=device)
    model = make_haar_model(cfg).to(device)
    model.set_band_statistics(stats)
    payload = prepare_training_mls(train,model,cfg["mls_regularizer"])
    atlas = TrainingMLSAtlas(payload,cfg["mls_regularizer"])
    if not bool(atlas.quadratic_supported.any()):
        raise ValueError("TRAIN atlas has no supported quadratic MLS charts")
    record = dict(train_cases=len(train.all_cases),bank_sites=len(payload["residuals"]),
                  folds=len(atlas.offsets)-1,geometry_mode=cfg["mls_regularizer"]["geometry_mode"],
                  max_fit_sites_per_fold=cfg["mls_regularizer"]["max_fit_sites_per_fold"],
                  fit_sites_per_fold=atlas.fit_sites_per_fold,
                  pca_supported_fraction=float(atlas.pca_supported.float().mean()),
                  quadratic_supported_fraction=float(atlas.quadratic_supported.float().mean()),
                  effective_neighbors_mean=float(atlas.effective.mean()),
                  design_condition_mean=float(atlas.condition.mean()),
                  quadratic_form_squared_sum=float(atlas.form.square().sum()),
                  elapsed_seconds=time.perf_counter()-begin)
    if "base_coefficients" in payload:
        # These describe the sampled TRAIN bank, not full-population statistics.
        base = payload["base_coefficients"].float()
        residual = payload["residuals"].float()
        if model.use_band_whitening:
            residual = residual*(stats["std"].cpu()+1.e-6)+stats["mean"].cpu()
        high = base+residual
        record["sampled_train_bank_physical_haar"] = {
            "baseline_band_variance":base.var(0,unbiased=False).tolist(),
            "hr_band_variance":high.var(0,unbiased=False).tolist(),
            "baseline_high_frequency_energy":float(base[:,1:].square().mean()),
            "hr_high_frequency_energy":float(high[:,1:].square().mean()),
            "paired_residual_energy":float(residual.square().mean())}
    atomic_save(record,project_path(cfg["run_dir"])/"mls_prebuild.json",json_file=True)
    print(json.dumps(record),flush=True)


def check_fixed_batch(record):
    """A falling composite loss alone does not prove deployment SR improves."""
    for name in ("loss","rollout_pixel_mse"):
        values = record[name]
        if len(values) < 2 or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"Invalid fixed-batch {name}")
        if values[-1] >= values[0]:
            raise ValueError(f"Fixed-batch {name} did not improve: {values[0]} -> {values[-1]}")
    if record["rollout_steps"] != record["inference_steps"]:
        raise ValueError("Fixed-batch rollout and deployment budgets differ")


def fixed_batch_sr(config,inventory,directory,steps=24,device="cuda"):
    import torch
    from .datasets import training_data
    from .haar_residual import HaarObjective, make_haar_model
    from .runtime import setup
    from .spatial_atlas import TrainingMLSAtlas, prepare_training_mls
    if steps < 1:
        raise ValueError("Positive fixed-batch optimization budget required")
    cfg = read_config(config)
    device = setup(cfg["train"]["seed"],device,cfg["train"].get("threads",4))
    train,_,_,_ = training_data(cfg,inventory)
    model = make_haar_model(cfg).to(device)
    model.set_band_statistics(load_checkpoint(project_path(cfg["band_stats_cache"])))
    atlas = TrainingMLSAtlas(prepare_training_mls(train,model,cfg["mls_regularizer"]),cfg["mls_regularizer"]).to(device)
    objective = HaarObjective(model,cfg,atlas)
    selected = []
    for index in range(len(train)):
        row = train[index]
        if float((row["hr"]-row["base"]).square().mean()) > 1.e-8:
            selected.append(row)
        if len(selected) == 4:
            break
    if len(selected) != 4:
        raise ValueError("Four nontrivial TRAIN patches required for the fixed-batch SR gate")
    lr,base,hr = [torch.stack([row[key] for row in selected]).to(device) for key in ("lr","base","hr")]
    case_ids = [row["id"] for row in selected]
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg["train"]["learning_rate"])
    record = dict(loss=[],rollout_pixel_mse=[],rollout_steps=cfg["train"]["rollout_steps"],
                  inference_steps=cfg["sampling"]["steps"],case_ids=case_ids,optimizer_updates=steps)
    for iteration in range(steps+1):
        torch.manual_seed(20261110)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(lr,base,hr,case_ids)["loss"]
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite fixed-batch SR loss")
        prediction = model.restore(lr,base,cfg["sampling"]["steps"],0.,sampler_type="ode",solver=cfg["sampling"]["solver"])
        record["loss"].append(float(loss.detach()))
        record["rollout_pixel_mse"].append(float((prediction-hr).square().mean()))
        if iteration < steps:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["train"]["clip_grad_norm"],error_if_nonfinite=True)
            optimizer.step()
        print(json.dumps(dict(fixed_batch_update=iteration,loss=record["loss"][-1],
                              rollout_pixel_mse=record["rollout_pixel_mse"][-1])),flush=True)
    check_fixed_batch(record)
    Path(directory).mkdir(parents=True,exist_ok=True)
    atomic_save(dict(passed=True,**record),Path(directory)/"smoke.json",json_file=True)


def check_sr_smoke(config):
    cfg = read_config(config)
    folder = project_path(cfg["run_dir"])/"bridge"
    rows = [json.loads(line) for line in (folder/"metrics.jsonl").read_text().splitlines()]
    if len(rows) != 1 or rows[0]["epoch"] != 1:
        raise ValueError("One completed SR smoke epoch required")
    row = rows[0]
    if (row["world_size"],row["global_batch_size"],row["val_case_count"],row["rollout_items"]) != (4,256,10,8):
        raise ValueError("SR smoke needs four ranks, batch 64/rank, one eighth rollout, and all VAL cases")
    keys = ("loss","grad_norm","start_pixel_mse","rollout_pixel_mse","start_hf_mse","start_edge_mse",
            "start_mls_reconstruction_mse","rollout_mls_reconstruction_mse",
            "val_pooled_psnr_db","val_mean_slice_ssim")
    if not all(math.isfinite(row[k]) for k in keys):
        raise ValueError("Nonfinite SR smoke metrics")
    if row["grad_norm"] <= 0:
        raise ValueError("SR smoke must exercise optimization")
    atomic_save(dict(passed=True,epochs=rows),folder/"smoke_gate.json",json_file=True)


def report_sr(config,validation):
    cfg = read_config(config)
    rows = [json.loads(line) for line in (project_path(cfg["run_dir"])/"bridge/metrics.jsonl").read_text().splitlines()]
    summary = json.loads((Path(validation)/"summary.json").read_text())
    stable,drop = late_stability(rows)
    result = dict(completed_40_epochs=bool(rows and rows[-1]["epoch"] >= 40),
                  stability_gate=stable,max_late_psnr_drop=drop,late_window_epochs=5,
                  whole_val_psnr=summary["all_psnr_db"],whole_val_ssim=summary["mean_slice_ssim"],
                  final_test_authorized=True,test_selection="fixed historical 84 cases; checkpoint selected only on VAL",
                  metric_threshold_required=False)
    atomic_save(result,project_path(cfg["run_dir"])/"sr_report.json",json_file=True)
    print(json.dumps(result),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    prep = sub.add_parser("prepare")
    for key in ("template","legacy-checkpoint","inventory","ablation","run-id"):
        prep.add_argument("--"+key,required=True)
    smoke = sub.add_parser("check-smoke")
    smoke.add_argument("--config",required=True)
    summary = sub.add_parser("report")
    summary.add_argument("--config",required=True)
    summary.add_argument("--validation",required=True)
    prep_sr = sub.add_parser("prepare-sr")
    for key in ("template","legacy-checkpoint","inventory","run-id"):
        prep_sr.add_argument("--"+key,required=True)
    prep_sr.add_argument("--statistics-cache")
    bank = sub.add_parser("prebuild-mls")
    fixed = sub.add_parser("fixed-batch-sr")
    for command_parser in (bank,fixed):
        command_parser.add_argument("--config",required=True)
        command_parser.add_argument("--inventory",required=True)
        command_parser.add_argument("--device",default="cuda")
    fixed.add_argument("--directory",required=True)
    fixed.add_argument("--steps",type=int,default=24)
    sr_smoke = sub.add_parser("check-sr-smoke")
    sr_smoke.add_argument("--config",required=True)
    sr_summary = sub.add_parser("report-sr")
    sr_summary.add_argument("--config",required=True)
    sr_summary.add_argument("--validation",required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    {"prepare":prepare,"check-smoke":check_smoke,"report":report,"prepare-sr":prepare_sr,
     "prebuild-mls":prebuild_mls,"fixed-batch-sr":fixed_batch_sr,"check-sr-smoke":check_sr_smoke,
     "report-sr":report_sr}[command](**args)


if __name__ == "__main__":
    main()
