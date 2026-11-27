"""Unchanged MRI/EM dataset contract copied from vem_manifold; no image computation at import."""
import hashlib
import json
from .data import PatchDataset, read_manifest
from .runtime import project_path, fingerprint

def index_mri_data(data):
    """Audit all configured PNG splits once and return the case metadata."""
    from .mri_data import index_cases, validate_case_splits

    cases = {}
    for split in ("train", "val", "test"):
        if split not in data:
            continue
        roots = data[split]
        options = dict(scale=data["scale"],
                       channel_policy=data.get("channel_policy", "identical_rgb_to_gray"),
                       scan_workers=data.get("scan_workers", 4), progress=True,
                       case_limit=data.get("debug_case_limit"))
        for key in ("normalization_scale", "normalization_offset"):
            if key in data:
                options[key] = data[key]
        for config_name, argument in (("scan_cache", "cache_path"), ("legacy_scan_cache", "legacy_cache_path")):
            cache = data.get(config_name, {}).get(split)
            if cache:
                options[argument] = project_path(cache)
        cases[split] = index_cases(project_path(roots["root_lq"]),
                                  project_path(roots["root_hq"]), **options)
    audit = validate_case_splits(cases)
    print(json.dumps({"mri_split_audit": audit, "debug_case_limit": data.get("debug_case_limit"),
                      "operator": "paired_xy", "observed_hr_voxels": 0}), flush=True)
    return cases


def training_data(cfg, case_inventory=None, *, return_base=True, fast_trusted=False):
    """Build datasets and a checkpoint identity from the selected data contract."""
    seed, data = cfg["train"]["seed"], cfg["data"]
    if data.get("kind", "volume_npy") == "paired_mri_png":
        from .mri_data import PairedMRIPatchDataset

        if case_inventory is None:
            cases = index_mri_data(data)
            identity = [{"split": split, "id": case.id, "fingerprint": case.fingerprint}
                        for split, items in cases.items() for case in items]
        else:
            from .mri_inventory import load_inventory
            cases, identity = load_inventory(case_inventory, data, ("train", "val"))
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        datasets = []
        for split, budget_key in (("train", "samples_per_epoch"), ("val", "val_samples_per_epoch")):
            roots = data[split]
            dataset = PairedMRIPatchDataset(
                project_path(roots["root_lq"]), project_path(roots["root_hq"]), split,
                scale=data["scale"], patch_size=tuple(data["patch_size"]),
                samples_per_volume=cfg["train"]["train_samples" if split == "train" else "val_samples"],
                samples_per_epoch=cfg["train"].get(budget_key), seed=seed, cases=cases[split],
                return_base=return_base, fast_trusted=fast_trusted and split == "train")
            print(json.dumps({"split": split, "epoch_patches": len(dataset),
                              "skipped_cases": dataset.skipped_cases}), flush=True)
            datasets.append(dataset)
        return datasets[0], datasets[1], digest, identity
    if case_inventory is not None:
        raise ValueError("A prepared case inventory requires paired MRI")
    manifest = project_path(data["manifest"])
    identity = [{"id": item.id, "size_bytes": item.path.stat().st_size,
                 "mtime_ns": item.path.stat().st_mtime_ns} for item in read_manifest(manifest)]
    options = dict(manifest=manifest, patch_size=tuple(data["patch_size"]), scale=data["scale"], seed=seed)
    train_set = PatchDataset(split="train", samples_per_volume=cfg["train"]["train_samples"], **options)
    val_set = PatchDataset(split="val", samples_per_volume=cfg["train"]["val_samples"], **options)
    return train_set, val_set, fingerprint(manifest), identity

