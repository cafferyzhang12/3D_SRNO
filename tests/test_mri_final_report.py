import csv
import json

import pytest

from vemgsbm.mri_final_report import build


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
    methods = ["direct", "bridge", "pca", "mls", "mls_geometry", "mls_shuffled", "swinir"]
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
    matrix, folder = tmp_path / "matrix.json", tmp_path / "report"
    save(matrix, {"runs": rows})
    methods = build(matrix, hist, folder, resamples=20)
    assert methods[0]["method"] == "swinir"
    text = (folder / "report.md").read_text(encoding="utf8")
    assert "选定 **direct**" in text and "第 2/2" in text
    paired = list(csv.DictReader((folder / "paired.csv").open(encoding="utf-8-sig")))
    same_seed = [r for r in paired if r["candidate"] == "bridge" and r["reference"] == "direct"]
    assert [r["training_seed"] for r in same_seed] == ["42", "43", "44", "all"]
    assert all(float(r["pooled_psnr_db_low95"]) == pytest.approx(.5) for r in same_seed)
    assert len((folder / "cases.csv").read_text(encoding="utf8").splitlines()) == 84 * len(rows) + 1
    rows[0]["validation_psnr"] += 1
    save(matrix, {"runs": rows})
    with pytest.raises(ValueError, match="validation score"):
        build(matrix, hist, folder, resamples=20)
