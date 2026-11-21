"""Independent archive confirmation after whole-VAL model and sampler freeze."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from .mri_data import (_case_directories, _filename_indices, _preserved_indices, _read_raw,
                       SIDECAR, index_cases, natural_key)
from .mri_infer import infer_mri
from .mri_inventory import load_inventory
from .runtime import atomic_save, fingerprint, load_checkpoint, project_path


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _decoded(path, policy):
    with Image.open(path) as image:
        if image.format != "JPEG" or image.mode not in ("L", "RGB"):
            raise ValueError(f"Expected grayscale or equal-channel RGB JPEG: {path}")
        pixels = np.array(image)
    if pixels.ndim == 3:
        if (policy != "identical_rgb_to_gray" or not np.array_equal(pixels[...,0], pixels[...,1])
                or not np.array_equal(pixels[...,0], pixels[...,2])):
            raise ValueError(f"Archive JPEG channel policy mismatch: {path}")
        pixels = pixels[...,0]
    if pixels.ndim != 2 or pixels.dtype != np.uint8:
        raise ValueError(f"Expected an eight-bit decoded archive plane: {path}")
    return pixels


def _frames(folder):
    files = {}
    for p in folder.iterdir():
        if not p.is_file() or p.is_symlink() or p.suffix.lower() not in (".jpg", ".jpeg") or p.stem in files:
            raise ValueError(f"Unexpected or ambiguous archive frame: {p}")
        files[p.stem] = p
    if not files:
        raise ValueError(f"No archive frames in {folder}")
    return files


def _archive_case(roots, cid, sidecar):
    lr, hr = [_frames(root / cid) for root in roots]
    if set(lr) != set(hr):
        raise ValueError(f"Archive LR/HQ stems differ: {cid}")
    stems = sorted(lr, key=natural_key)
    if sidecar is None:
        indices = dict(zip(stems, _filename_indices([lr[s] for s in stems]), strict=True))
    else:
        row = sidecar[cid]
        indices = row["indices"]
        active = set(indices) - set(row["removed_stems"])
        if set(stems) not in (set(indices), active):
            raise ValueError(f"Archive quarantine sidecar mismatch: {cid}")
        stems = [s for s in stems if s in active]
    return lr, hr, stems, indices


def _cohort(pool, excluded, count, seed):
    candidates = [cid for cid in pool if cid.casefold() not in excluded]
    if len(candidates) < count:
        raise ValueError("Insufficient distinct archive cases after excluding all known cohorts")
    candidates.sort(key=lambda cid: hashlib.sha256(f"{seed}:{cid}".encode()).digest())
    return sorted(candidates[:count], key=natural_key)


def _excluded(identity, checkpoint):
    return {row["id"].casefold() for row in [*identity, *checkpoint.get("data_identity", [])]}


def _verify_overlap(roots, raw_sidecar, val, data):
    prepared_roots = [project_path(data["val"]["root_" + field]) for field in ("lq","hq")]
    small_sidecar, _ = _preserved_indices(*prepared_roots)
    total = 0
    for c in val:
        lr, hr, active, indices = _archive_case(roots, c.id, raw_sidecar)
        kept = set(c.stems)
        if small_sidecar is None:
            if set(active) != kept:
                raise ValueError(f"Missing explanation for different validation slices: {c.id}")
        else:
            row = small_sidecar[c.id]
            if (set(row["indices"]) != set(indices) or kept != set(row["indices"]) - set(row["removed_stems"])
                    or not kept <= set(active)):
                raise ValueError(f"Archive and development quarantine disagree: {c.id}")
        for z, stem in enumerate(c.stems):
            if indices[stem] != c.indices[z]:
                raise ValueError(f"Original slice position differs: {c.id}/{stem}")
            for raw, png in ((lr[stem], c.lq[z]), (hr[stem], c.hq[z])):
                arr, bits = _read_raw(png, c.channel_policy)
                if bits != 8 or not np.array_equal(_decoded(raw,c.channel_policy), arr):
                    raise ValueError(f"Decoded JPEG does not exactly reproduce retained PNG: {c.id}/{stem} ({png})")
            total += 1
    if not total:
        raise ValueError("Missing validation overlap for JPEG equivalence check")
    return total


def prepare(checkpoint_path, inventory, archive, output, *, count=84, seed=20261121):
    output = Path(output).resolve()
    saved = load_checkpoint(checkpoint_path)
    data = saved["config"]["data"]
    if data.get("debug_case_limit") is not None or data["scale"] != 4:
        raise ValueError("Generalization audit requires full XYx4 protocol inventory")
    indexed, identity = load_inventory(inventory, data, ("val",))
    archive = Path(archive).resolve()
    roots = (archive / "LQ" / "X4", archive / "HQ")
    lr_cases, hr_cases = [_case_directories(root) for root in roots]
    if set(lr_cases) != set(hr_cases):
        raise ValueError("Archive case identities differ across resolutions")
    raw_sidecar, _ = _preserved_indices(*roots)
    known = _excluded(identity, saved)
    selected = _cohort(lr_cases, known, count, seed)
    policy = data.get("channel_policy", "identical_rgb_to_gray")
    checked_slices = _verify_overlap(roots, raw_sidecar, indexed["val"], data)
    if output.exists():
        raise FileExistsError(f"Choose a new independent cohort directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "cohort"
        records, positions = [], {}
        for cid in selected:
            lr, hr, stems, indices = _archive_case(roots, cid, raw_sidecar)
            if not stems:
                raise ValueError(f"Locked archive case contains no retained slice: {cid}")
            hashes = hashlib.sha256()
            for sub in ("LQ","HQ"):
                (stage / sub / cid).mkdir(parents=True)
            for stem in stems:
                a, b = _decoded(lr[stem],policy), _decoded(hr[stem],policy)
                if b.shape != tuple(n * 4 for n in a.shape):
                    raise ValueError(f"Archive XYx4 alignment differs: {cid}/{stem}")
                for sub, pixels in (("LQ", a), ("HQ", b)):
                    dest = stage / sub / cid / (stem + ".png")
                    Image.fromarray(pixels).save(dest)
                    if not np.array_equal(_read_raw(dest, policy)[0], pixels):
                        raise ValueError(f"Lossless decoded PNG verification failed: {dest}")
                    hashes.update(bytes.fromhex(fingerprint(dest)))
            if raw_sidecar is not None:
                positions[cid] = {"indices": indices, "filenames": {s: s+".png" for s in indices},
                                  "removed_stems": sorted(set(indices) - set(stems), key=natural_key)}
            records.append({"case_id": cid, "stems": stems, "original_indices": [indices[s] for s in stems],
                            "png_sha256": hashes.hexdigest()})
        if raw_sidecar is not None:
            sidecar = dict(version=1, kind="mri_zero_pair_quarantine", root_lq=str(output/"LQ"),
                           root_hq=str(output/"HQ"), patients=positions)
            for sub in ("LQ","HQ"):
                atomic_save(sidecar,stage/sub/SIDECAR,json_file=True)
        record = {"split": "independent_archive", "provenance": "original archive validation directory",
                  "seed": seed, "count": count, "cases": records, "archive": str(archive),
                  "available_archive_cases": len(lr_cases), "known_ids": len(known),
                  "archive_quarantine_sidecar": raw_sidecar is not None,
                  "development_pairs_pixel_verified": checked_slices, "overlap_cases": [c.id for c in indexed["val"]],
                  "inventory_sha256": fingerprint(inventory), "channel_policy": policy,
                  "normalization_scale": data.get("normalization_scale", None),
                  "normalization_offset": data.get("normalization_offset", 0.0)}
        atomic_save(record, stage/"cohort.json", json_file=True)
        os.rename(stage, output)
    return record


def confirm(folder, cohort, output, device):
    folder, cohort, output = (Path(p).resolve() for p in (folder,cohort,output))
    if output.exists():
        raise FileExistsError(output)
    frozen = _read(folder/"freeze.json")
    model, inventory = folder/"model.pt", folder/"inventory.json.gz"
    locked = _read(cohort/"cohort.json")
    if (fingerprint(model) != frozen["selected"]["checkpoint_sha256"] or fingerprint(inventory) != frozen["inventory_sha256"]
            or fingerprint(folder/"manifest.json") != frozen["manifest_sha256"]
            or locked["inventory_sha256"] != frozen["inventory_sha256"] or locked["split"] != "independent_archive"):
        raise ValueError("Frozen model, protocol, or independent archive identity changed")
    saved = load_checkpoint(model)
    data = saved["config"]["data"]
    _, identity = load_inventory(inventory, data, ("val",))
    excluded = _excluded(identity, saved)
    case_ids = [r["case_id"] for r in locked["cases"]]
    pool = _case_directories(Path(locked["archive"])/"HQ")
    if case_ids != _cohort(pool, excluded, 84, 20261121):
        raise ValueError("Independent locked cases differ from the disjoint prespecified sample")
    policy = data.get("channel_policy", "identical_rgb_to_gray")
    if (policy != locked["channel_policy"] or data.get("normalization_scale") != locked["normalization_scale"]
            or data.get("normalization_offset", 0.) != locked["normalization_offset"]):
        raise ValueError("Intensity protocol changed")
    cases = index_cases(cohort/"LQ",cohort/"HQ",scale=4,channel_policy=policy,
                        normalization_scale=locked["normalization_scale"], normalization_offset=locked["normalization_offset"])
    for row, case in zip(locked["cases"], cases, strict=True):
        if case.id != row["case_id"] or list(case.stems) != row["stems"] or list(case.indices) != row["original_indices"]:
            raise ValueError("Independent archive positions changed")
        hashes = hashlib.sha256()
        for pair in zip(case.lq, case.hq, strict=True):
            for p in pair:
                hashes.update(bytes.fromhex(fingerprint(p)))
        if hashes.hexdigest() != row["png_sha256"]:
            raise ValueError("Independent archive pixel files changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        result = infer_mri(model, cohort/"LQ", Path(tmp)/"independent", root_hq=cohort/"HQ",
                           case_ids=case_ids, test_fraction=1., channel_policy=policy, save_volume_limit=0,
                           device_name=device, **frozen["selected"]["inference"])
        summary = _read(result/"summary.json")
        if [r["case_id"] for r in summary["cases"]] != case_ids:
            raise ValueError("Incomplete independent archive confirmation")
        summary.update(split="independent_archive", cohort_sha256=fingerprint(cohort/"cohort.json"),
                       freeze_sha256=fingerprint(folder/"freeze.json"),
                       provenance="held-out cases from same archive; external-center provenance unverified")
        atomic_save(summary,result/"summary.json",json_file=True)
        os.rename(result,output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prep = actions.add_parser("prepare")
    for key in ("checkpoint", "case-inventory", "archive", "output"):
        prep.add_argument("--"+key, required=True)
    evaluation = actions.add_parser("confirm")
    for key in ("folder", "cohort", "output"):
        evaluation.add_argument("--"+key, required=True)
    evaluation.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.action == "prepare":
        r = prepare(args.checkpoint,args.case_inventory,args.archive,args.output)
        print(json.dumps({k: r[k] for k in ("count","development_pairs_pixel_verified","available_archive_cases")}))
    else:
        confirm(args.folder,args.cohort,args.output,args.device)


if __name__ == "__main__":
    main()
