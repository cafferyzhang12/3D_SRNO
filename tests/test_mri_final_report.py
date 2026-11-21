import csv
import json
from pathlib import Path
import runpy

import pytest

from vemgsbm.mri_final_report import build, compute_profile
from vemgsbm.runtime import fingerprint


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf8")


def case_summary(psnr, ssim):
    cases = []
    mse = 10 ** (-psnr / 10)
    for i in range(84):
        slices = 178 + int(i < 39)
        pixels = slices * 512**2
        cases.append({"case_id": str(i), "input_fingerprint": str(i), "evaluation_seconds": 1.5,
                      "metrics": {"case_id": str(i), "pixel_count": pixels, "squared_error_sum": pixels * mse,
                                  "evaluated_slices": slices, "ssim_slice_count": slices,
                                  "all_mse": mse, "all_psnr_db": psnr, "mean_slice_ssim": ssim}})
    return {"cases": cases, "subset": {"case_ids": [r["case_id"] for r in cases], "fraction": .15},
            "all_psnr_db": psnr, "all_mse": mse, "mean_slice_ssim": ssim,
            "total_slices": 14991, "pixel_count": 3929800704}


def test_frozen_multiseed_test_report_selects_by_val_and_verifies_evidence(tmp_path):
    hist = tmp_path / "archive.json"
    save(hist, case_summary(30.6781, .907021))
    rows = []
    methods = ["direct", "bridge", "pca", "mls", "mls_geometry", "mls_shuffled", "swinir", "matched_mls"]
    for i, method in enumerate(methods):
        external = method == "swinir"
        for seed in ([None] if external else [42, 43, 44]):
            label = f"{method}-{seed}"
            result = tmp_path / label / "test"
            frozen = tmp_path / label / "freeze"
            opts = dict(predictor="model", steps=4, seed=42, samples=1)
            s = case_summary(30 + i / 2 + (seed or 42) % 3 / 10, .90 + i / 1000)
            s.update(split="test", checkpoint_sha256=label, inference_variant=method,
                     network_calls_per_tile_sample=4, **opts)
            if external:
                s["external"] = {"pretraining": "DF2K"}
            val = 50 if method == "direct" else i + 20
            save(result / "summary.json", s)
            save(frozen / "manifest.json", s["subset"])
            save(frozen / "freeze.json", {"selected": {"checkpoint_sha256": label, "pooled_psnr": val,
                                                       "inference": opts}})
            rows.append(dict(method=method, label=label, training_seed=seed, group="external" if external else "new",
                             output=str(result), freeze_dir=str(frozen), validation_psnr=val))
            if method in {"pca", "matched_mls"}:
                variant = "pca" if method == "pca" else "mls"
                rows[-1].update(roles=["matched_" + variant], arm=dict(variant=variant, seed=seed, path=0, dim=4,
                               shuffle=0, reference="identity", readout="endpoint", query="endpoint"))
                if method == "matched_mls":
                    rows[-1]["group"] = "isolation"
    matrix, folder = tmp_path / "matrix.json", tmp_path / "report"
    diagnostics = {"ranking": [dict(arm="plain", nfe=20, psnr=30., ssim=.9, delta_psnr=0., delta_ssim=0.)],
                   "chart": dict(oracle_recalled=.4, top3_mass=.5, lr_prior_weighted_gap=.01),
                   "correction": {"whole": dict(count=10, dot=1., delta_mse=-.001, optimal_change_multiplier=.8,
                                                interpretation="当前幅度降低误差")}}
    control = dict(stage="controls", query="endpoint", path=1, shuffle=0, path_relative_correction_rms=.001,
                   effective_neighbors=32., quadratic_fallback_fraction=.05, val_edge_mse=.008, val_psnr=29.3,
                   val_ssim=.89)
    save(matrix, {"runs": rows, "champion_method": "direct", "sampler_validation": diagnostics, "development": [control]})
    methods = build(matrix, hist, folder, resamples=20)
    assert methods[0]["method"] == "swinir"
    text = (folder / "report.md").read_text(encoding="utf8")
    assert "选定 **direct**" in text and "第 2/2" in text and "固定局部维数 4" in text
    assert "前三图概率质量为 0.5000" in text and "路径相对修正 RMS" in text
    assert not any(row["method"] == "matched_mls" for row in methods)
    paired = list(csv.DictReader((folder / "paired.csv").open(encoding="utf-8-sig")))
    same_seed = [r for r in paired if r["candidate"] == "bridge" and r["reference"] == "direct"]
    assert [r["training_seed"] for r in same_seed] == ["42", "43", "44", "all"]
    assert all(float(r["pooled_psnr_db_low95"]) == pytest.approx(.5) for r in same_seed)
    external = next(r for r in paired if r["candidate"] == "mls_geometry" and r["reference"] == "swinir")
    assert float(external["mean_slice_ssim"]) == pytest.approx(-.002)
    isolated = [r for r in paired if r["candidate"] == "matched_mls" and r["reference"] == "matched_pca"]
    assert [r["training_seed"] for r in isolated] == ["42", "43", "44", "all"]
    assert all(float(r["pooled_psnr_db_low95"]) == pytest.approx(2.5) for r in isolated)
    assert "ΔPSNR 95% CI" in text and "ΔSSIM 95% CI" in text
    assert len((folder / "cases.csv").read_text(encoding="utf8").splitlines()) == 84 * len(rows) + 1
    rows[0]["validation_psnr"] += 1
    save(matrix, {"runs": rows})
    with pytest.raises(ValueError, match="validation score"):
        build(matrix, hist, folder, resamples=20)


def test_call_counts_include_online_geometry_and_native_conditioning():
    base = dict(steps=20, drift_network_calls_per_tile_sample=20, inference_variant="endpoint_safe")
    g = compute_profile(dict(base, geometry_guidance={"start": .3, "normal_strength": .01}), "epoch38", "historical")
    # One LR route + two encoded states for each of 13 retained guided updates.
    assert (g["atlas_encoder_calls"], g["guidance_backward_calls"], g["total_neural_forwards_per_tile_sample"]) == (27, 13, 47)
    m = compute_profile(dict(steps=20, inference_variant="mls_guided_momentum_bridge",
                             momentum_geometry={"start": .25}), "momentum_mls", "historical")
    assert (m["atlas_encoder_calls"], m["atlas_encoder_inputs"], m["guidance_backward_calls"]) == (15, 43, 28)
    spatial = compute_profile(base, "mls_geometry", "new")
    assert spatial["total_neural_forwards_per_tile_sample"] == 22
    assert compute_profile(base, "bilinear", "bilinear")["total_neural_forwards_per_tile_sample"] == 0


def test_matched_dimension_validation_rejects_test_and_path_confound(tmp_path):
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/mri_freeze_study.py"))
    matched = module["matched_geometry_validation"]
    rows = []
    for dim in (2, 4, 6, 8):
        for variant in ("pca", "mls"):
            root = tmp_path / f"{variant}{dim}"
            s = case_summary(30 + (variant == "mls"), .9 + .01 * (variant == "mls"))
            # Small cohort suffices for the validation comparison.
            s["cases"] = s["cases"][:2]
            s.update(split="val", subset={"fraction": 1, "case_ids": ["0", "1"]})
            path = root / "summary.json"
            save(path, s)
            rows.append(dict(result=str(root), summary_sha256=fingerprint(path), arm=dict(
                dim=dim, variant=variant, query="endpoint", seed=42, path=0, shuffle=0, reference="identity",
                readout="endpoint")))
    dimensions = tmp_path / "dimensions.json"
    save(dimensions, {"rows": rows})
    checks = matched(dimensions)
    assert [r["dim"] for r in checks] == [2, 4, 6, 8]
    assert all(r["pooled_psnr_db"]["difference"] == pytest.approx(1.) for r in checks)
    rows[0]["arm"]["path"] = 1
    save(dimensions, {"rows": rows})
    with pytest.raises(ValueError, match="matched"):
        matched(dimensions)
    rows[0]["arm"]["path"] = 0
    p = Path(rows[0]["result"]) / "summary.json"
    s = json.loads(p.read_text())
    s["split"] = "test"
    save(p, s)
    rows[0]["summary_sha256"] = fingerprint(p)
    save(dimensions, {"rows": rows})
    with pytest.raises(ValueError, match="matched"):
        matched(dimensions)


def test_freeze_reuses_primary_weights_for_matched_comparisons(tmp_path):
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/mri_freeze_study.py"))
    def model(variant, path, psnr):
        cfg = dict(variant=variant, path=path, dim=4, query="endpoint", readout="endpoint",
                   reference="identity", shuffle=0)
        runs = []
        for seed in (42, 43, 44):
            root = tmp_path / f"{variant}_{path}_{seed}"
            runs.append(dict(arm={**cfg, "seed": seed}, checkpoint=root / "best.pt", result=root / "val",
                             pooled_psnr=psnr, summary_sha256=f"sha{root.name}"))
        return dict(config=cfg, mean_pooled_psnr=psnr, runs=runs)
    pca = model("pca", 0, 30)
    mls_path = model("mls", 1, 31)
    mls_control = model("mls", 0, 50)
    study = dict(ranking=[pca, mls_path], matched_models=[pca, mls_control])
    rows = module["new_arms"](study)
    assert len(rows) == 9 and sum(row["confirm_archive"] for row in rows) == 3
    assert {r["method"] for r in rows if r["confirm_archive"]} == {"mls"}
    assert all(r["roles"] == ["finalist", "matched_pca"] for r in rows if r["method"] == "pca")
    assert all(r["group"] == "isolation" for r in rows if r["method"] == "matched_mls")
    study["matched_models"] = [pca, mls_path]
    with pytest.raises(ValueError, match="surface order"):
        module["new_arms"](study)
