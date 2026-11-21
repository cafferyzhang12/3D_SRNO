"""Discover configured MRI protocols and audit their cohorts without running models."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import yaml

from .mri_data import (_case_directories, _preserved_indices, index_cases, natural_key,
                       validate_case_splits)
from .mri_infer import checked_test_subset
from .mri_inventory import load_inventory
from .runtime import MODES, atomic_save, load_checkpoint, project_path


SPLITS = ("train", "val", "test")
ORIGINAL = {"mri_x4.yaml", "mri_momentum_x4.yaml", "mri_momentum_tdm_x4.yaml",
            "mri_momentum_mls_x4.yaml", "mri_gsbm_mls_x4.yaml", "mri_gsbm_mls_guided_x4.yaml"}
RESEARCH = {"mri_haar_direct_x4.yaml", "mri_haar_bridge_x4.yaml"}


class CohortMismatch(ValueError):
    """Aggregate mismatch safe to write to cohort reports."""


def protocol(data, data_root):
    if data.get("degradation") != "paired_xy" or type(data.get("scale")) is not int or data["scale"] not in (2, 4):
        raise ValueError("expected paired_xy with XY scale 2 or 4 and native Z")
    settings = {"xy_scale": data["scale"], "z_scale": 1,
                "channel_policy": data.get("channel_policy", "identical_rgb_to_gray"),
                "normalization_scale": data.get("normalization_scale"),
                "normalization_offset": data.get("normalization_offset", 0.0),
                "case_limit": data.get("debug_case_limit"), "roots": {}}
    for split in SPLITS:
        settings["roots"][split] = {}
        for role in ("root_lq", "root_hq"):
            raw = data.get(split, {}).get(role)
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"missing {split}.{role}")
            path = Path(os.path.expandvars(raw)).expanduser()
            if data_root and path.is_relative_to("data/MRI_SR"):
                path = Path(data_root) / path.relative_to("data/MRI_SR")
            settings["roots"][split][role] = str(project_path(path))
    return settings


def counts(cases):
    return {"cases": len(cases), "slices": sum(len(c.stems) for c in cases),
            "hr_voxels": sum(len(c.stems) * c.lr_shape[0] * c.lr_shape[1] * c.scale ** 2 for c in cases)}


def audit_names(roots, cases, case_limit):
    """Match every retained filename to disk, respecting the original zero-pair quarantine."""
    low, high = (Path(roots[r]) for r in ("root_lq", "root_hq"))
    lq_dirs, hq_dirs = _case_directories(low), _case_directories(high)
    if lq_dirs.keys() != hq_dirs.keys():
        raise CohortMismatch("LQ and HQ case directories differ")
    preserved, metadata = _preserved_indices(low, high)
    if preserved is not None and preserved.keys() != lq_dirs.keys():
        raise CohortMismatch("quarantine and live case directories differ")
    live = sorted(lq_dirs, key=natural_key)[:case_limit]
    expected = {c.id: c for c in cases}
    for name in live:
        case = expected.pop(name, None)
        record = preserved[name] if preserved is not None else None
        removed = set(record["removed_stems"]) if record is not None else set()
        if case is None and (record is None or set(record["indices"]) != removed):
            raise CohortMismatch("extra active case on disk")
        for role, folder in (("lq", lq_dirs[name]), ("hq", hq_dirs[name])):
            with os.scandir(folder) as listing:
                files = {item.name for item in listing if item.name.lower().endswith(".png")
                         and item.is_file(follow_symlinks=False)}
            if role == "lq" and metadata.get("shared_hq_sidecar"):
                files -= {record["filenames"][stem] for stem in removed}
            indexed = {p.name for p in getattr(case, role)} if case is not None else set()
            if files != indexed:
                raise CohortMismatch(f"retained {role.upper()} filenames differ from prepared inventory")
    if expected:
        raise CohortMismatch("indexed case absent from disk")


def discover(config_dir, data_root=None, checkpoint=None, inventory=None, manifest=None):
    if bool(checkpoint) != bool(inventory) or (manifest and not inventory):
        raise ValueError("Supply both protocol checkpoint and case inventory for verification")
    groups, skipped, unsupported = {}, [], []
    files = sorted(p for p in Path(config_dir).rglob("*") if p.is_file() and p.suffix.lower() in (".yml", ".yaml"))
    if not files:
        raise ValueError(f"No YAML configurations in {config_dir}")
    for path in files:
        name = path.relative_to(config_dir).as_posix()
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf8"))
            if not isinstance(cfg, dict) or not isinstance(cfg.get("data"), dict):
                unsupported.append({"config": name, "reason": "missing data mapping"})
                continue
            data = cfg["data"]
            if data.get("kind") != "paired_mri_png":
                skipped.append({"config": name, "reason": f"data.kind={data.get('kind', 'volume_npy')}"})
                continue
            settings = protocol(data, data_root)
            mode = cfg.get("mode")
            if mode not in MODES:
                unsupported.append({"config": name, "reason": f"unsupported model mode {mode!r}"})
            key = json.dumps(settings, sort_keys=True, separators=(",", ":"))
            group = groups.setdefault(key, {"protocol": settings, "configs": []})
            source = "original" if name in ORIGINAL else "research_template" if name in RESEARCH else "additional"
            test = cfg.get("test", {})
            group["configs"].append({"config": name, "origin": source, "mode": mode,
                                     "supported_mode": mode in MODES,
                                     "default_test_fraction": test.get("fraction", 0.15),
                                     "default_subset_seed": test.get("subset_seed", 20261110)})
        except (AttributeError, TypeError, ValueError, yaml.YAMLError) as exc:
            unsupported.append({"config": name, "reason": str(exc)})

    reference, prepared = None, None
    if checkpoint:
        archive = load_checkpoint(checkpoint)["config"]["data"]
        reference = protocol(archive, data_root)
        prepared, _ = load_inventory(inventory, archive, SPLITS)
        validate_case_splits(prepared)
    verified_reference = False
    for key, group in groups.items():
        settings = group["protocol"]
        roots = settings["roots"]
        group["id"] = hashlib.sha256(key.encode()).hexdigest()[:16]
        group["matches_checkpoint_protocol"] = settings == reference if reference else None
        group["missing_roots"] = [{"split": s, "role": r, "path": p} for s in SPLITS
                                  for r, p in roots[s].items() if not Path(p).is_dir()]
        if group["missing_roots"]:
            group["status"] = "missing_roots"
            continue
        group["status"] = "configured"
        if not checkpoint:
            continue
        try:
            if settings == reference:
                for split in SPLITS:
                    try:
                        audit_names(roots[split], prepared[split], settings["case_limit"])
                    except CohortMismatch as exc:
                        raise CohortMismatch(f"{split}: {exc}") from exc
                cases = prepared
                selected, _ = checked_test_subset(cases["test"], manifest_path=manifest)
                group["fixed_test_15pct"] = {**counts(selected), "manifest_checked": bool(manifest)}
                group["status"] = "verified_inventory_filenames"
                verified_reference = True
            else:
                cases = {s: index_cases(**roots[s], scale=settings["xy_scale"], case_limit=settings["case_limit"],
                                        channel_policy=settings["channel_policy"],
                                        normalization_scale=settings["normalization_scale"],
                                        normalization_offset=settings["normalization_offset"]) for s in SPLITS}
                validate_case_splits(cases)
                group["status"] = "verified_independent_png_index"
            group["splits"] = {s: counts(cases[s]) for s in SPLITS}
        except (ValueError, OSError, KeyError, TypeError) as exc:
            # Library exceptions can contain patient paths. Keep published artifacts aggregate-only.
            issue = str(exc) if isinstance(exc, CohortMismatch) else f"{type(exc).__name__} during cohort verification"
            group.update(status="verification_failed", issue=issue)
    return {"yaml_count": len(files), "protocol_count": len(groups), "verified_reference": verified_reference,
            "protocols": list(groups.values()), "skipped": skipped, "unsupported": unsupported}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-dir", type=Path, default=project_path("configs"))
    parser.add_argument("--data-root", default=os.getenv("MRI_DATA_ROOT"))
    parser.add_argument("--protocol-checkpoint")
    parser.add_argument("--case-inventory")
    parser.add_argument("--fixed-test-manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = discover(args.configs_dir, args.data_root, args.protocol_checkpoint, args.case_inventory,
                      args.fixed_test_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_save(report, args.output_dir / "inventory.json", json_file=True)
    fields = ["protocol_id", "status", "config", "origin", "mode", "default_test_fraction", "split",
              "root_lq", "root_hq", "xy_scale", "z_scale", "cases", "slices", "hr_voxels", "issue"]
    with (args.output_dir / "inventory.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for group in report["protocols"]:
            for config in group["configs"]:
                for split in SPLITS:
                    issue = ", ".join(r["role"] for r in group["missing_roots"] if r["split"] == split)
                    writer.writerow({**config, **group["protocol"]["roots"][split], **group["protocol"],
                                     **group.get("splits", {}).get(split, {}), "split": split,
                                     "protocol_id": group["id"], "status": group["status"],
                                     "issue": issue or group.get("issue", "")})
        for status in ("unsupported", "skipped"):
            for row in report[status]:
                writer.writerow({"config": row["config"], "status": status, "issue": row["reason"]})
    print(json.dumps({"yaml_count": report["yaml_count"], "protocol_count": report["protocol_count"],
                      "verified_reference": report["verified_reference"]}))
    if args.protocol_checkpoint and (not report["verified_reference"] or
                                     any(g["status"] == "verification_failed" for g in report["protocols"])):
        raise SystemExit("MRI protocol verification failed; see inventory.json")


if __name__ == "__main__":
    main()
