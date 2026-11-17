"""Validate all configured MRI PNG pairs and write a reusable header cache."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

import yaml

from .mri_data import PairedMRIPatchDataset, data_fingerprint, index_cases, validate_case_splits


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def preflight(config_path, output_path, scan_workers=4, overwrite=False):
    output_path = _project_path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"MRI report already exists: {output_path}")
    config_path = _project_path(config_path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = cfg["data"]
    if data.get("debug_case_limit") is not None:
        raise ValueError("Full MRI preflight forbids debug_case_limit. Use the complete formal config.")
    cache_paths = data.get("scan_cache", {})
    legacy_paths = data.get("legacy_scan_cache", {})
    splits = {}
    details = {}
    for split in ("train", "val", "test"):
        roots = data.get(split)
        if roots is None:
            continue
        cases = index_cases(
            root_lq=_project_path(roots["root_lq"]), root_hq=_project_path(roots["root_hq"]),
            scale=data["scale"], cache_path=_project_path(cache_paths[split]) if split in cache_paths else None,
            legacy_cache_path=_project_path(legacy_paths[split]) if split in legacy_paths else None,
            channel_policy=data.get("channel_policy", "identical_rgb_to_gray"),
            normalization_scale=data.get("normalization_scale"), normalization_offset=data.get("normalization_offset", 0.0),
            scan_workers=scan_workers, progress=True,
        )
        splits[split] = cases
        dataset = PairedMRIPatchDataset(roots["root_lq"], roots["root_hq"], split,
                                       scale=data["scale"], patch_size=tuple(data["patch_size"]),
                                       samples_per_volume=1, cases=cases)
        details[split] = {
            "root_lq": str(_project_path(roots["root_lq"]).resolve()),
            "root_hq": str(_project_path(roots["root_hq"]).resolve()),
            "case_count": len(cases), "slice_count": sum(len(case.stems) for case in cases),
            "patch_eligible_cases": len(dataset.cases), "skipped_patch_cases": dataset.skipped_cases,
            "fingerprint": data_fingerprint(cases),
            "cases": [{"id": case.id, "slices": len(case.stems), "lr_shape": case.lr_shape,
                       "bits": case.bits, "normalization_scale": case.normalization_scale,
                       "normalization_offset": case.normalization_offset, "runs": case.runs,
                       "metadata": case.metadata} for case in cases],
        }
    if "train" not in splits or "val" not in splits:
        raise ValueError("Formal MRI preflight requires separate train and val roots")
    report = {"config": str(config_path), "task": "paired_mri_xy", "scale_xy": data["scale"],
              "z_sampling": "original_indices_without_interpolation", "full_inventory": True,
              "validation": validate_case_splits(splits), "splits": details,
              "verification": "All pair inventories and size/mtime-cached headers. Pixel ranges checked on consumption."}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if overwrite:
            os.replace(temporary, output_path)
        else:
            os.link(temporary, output_path)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scan-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = preflight(args.config, args.output, args.scan_workers, args.overwrite)
    print(json.dumps(report["validation"], indent=2))


if __name__ == "__main__":
    main()
