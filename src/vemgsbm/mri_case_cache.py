"""Crash-recoverable case commits for whole-volume inference on a shared filesystem."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from .runtime import atomic_save, fingerprint


PUBLICATION_FILES = {"summary.json", "summary.csv", "per_case.csv", "subset_manifest.json",
                     "evaluation_identity.json", "publication.json"}


def acquire_lock(path):
    """Keep one stable inode. POSIX flock and Windows range locks die with their process."""
    stream = Path(path).open("a+b")
    try:
        if os.name == "posix":
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        stream.close()
        raise FileExistsError(f"MRI inference already active for {path}") from None
    return stream


def evaluation_identity(config, cases, subset, exported, checkpoint_hash, root_lq, root_hq,
                        split, seed, steps, samples, device):
    inputs = []
    for case in cases:
        content = hashlib.sha256()
        for path in (*case.lq, *(case.hq or ())):
            content.update(bytes.fromhex(fingerprint(path)))
        inputs.append({"case_id": case.id, "inventory_fingerprint": case.fingerprint,
                       "image_bytes_sha256": content.hexdigest(), "exported": case.id in exported})
    src = Path(__file__).parent
    modules = ["mri_infer", "mri_metrics", "mri_data", "infer", "bridge", "models", "network",
               "runtime", "haar_residual", "atlas_geometry", "metric", "spatial_atlas", "mri_external"]
    sources = [src/(name+".py") for name in modules] + sorted((src/"momentum").glob("*.py"))
    return {"format_version": 1, "checkpoint_sha256": checkpoint_hash,
            "sources": {p.relative_to(src).as_posix(): fingerprint(p) for p in sources if p.is_file()},
            "split": split or "custom", "root_lq": str(Path(root_lq).resolve()),
            "root_hq": str(Path(root_hq).resolve()) if root_hq else None,
            "subset": subset, "cases": inputs, "sampling": config["sampling"],
            "inference_variant": config["inference_variant"], "predictor": config["predictor"],
            "seed": seed, "steps": steps, "samples": samples, "torch": str(torch.__version__),
            "device": torch.cuda.get_device_name(device) if device.type=="cuda" else device.type}


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",",":")).encode()).hexdigest()


def bind_cache(cache, identity, output):
    """Call under the destination lock, before any rank may access case commits."""
    if output.exists():
        if not (output/"evaluation_identity.json").is_file() or _read(output/"evaluation_identity.json") != identity:
            raise ValueError("Published MRI inference identity differs from this evaluation")
        receipt = _read(output/"publication.json")
        expected_files = {"summary.json","subset_manifest.json","evaluation_identity.json"}
        for name in ("summary.csv", "per_case.csv"):
            if (output/name).exists():
                expected_files.add(name)
        if set(receipt) != expected_files or any(fingerprint(output/name) != receipt[name] for name in expected_files):
            raise ValueError("Published MRI reports differ from completion receipt")
        return True
    cache.mkdir(exist_ok=True)
    manifest = cache/"identity.json"
    if manifest.is_file():
        if _read(manifest) != identity:
            raise ValueError("MRI resume identity changed; select a different output for this evaluation")
    else:
        if (cache/"cases").exists():
            raise ValueError("Unidentified MRI case cache")
        atomic_save(identity,manifest,json_file=True)
    (cache/"cases").mkdir(exist_ok=True)
    return False


def verify_case(directory, key, case, exported):
    receipt = _read(directory/"case_commit.json")
    names = {"metadata.json"}
    if case.hq is not None:
        names.add("metrics.json")
    if exported:
        names.update(("mean.npy","variance.npy"))
    if (receipt["evaluation_sha256"] != key or receipt["case_id"] != case.id
            or set(receipt["files"]) != names
            or any(fingerprint(directory/name) != receipt["files"][name] for name in names)):
        raise ValueError(f"MRI cached case checksum or evaluation identity differs: {case.id}")


def commit_case(cache, case, identity_key, exported, write):
    """Reconstruct only into a private attempt, publish one completed case with one rename."""
    destination = cache/"cases"/case.id
    if destination.exists():
        verify_case(destination, identity_key, case, exported)
        return
    with tempfile.TemporaryDirectory(prefix="case-attempt-", dir=cache) as tmp:
        root = Path(tmp)
        write(root)
        completed = root/case.id
        hashes = {p.name: fingerprint(p) for p in completed.iterdir()}
        atomic_save({"evaluation_sha256": identity_key, "case_id": case.id, "files":hashes},
                    completed/"case_commit.json", json_file=True)
        verify_case(completed, identity_key, case, exported)
        os.rename(completed,destination)


def seal(staged, identity):
    atomic_save(identity, staged/"evaluation_identity.json", json_file=True)
    hashes = {name: fingerprint(staged/name) for name in PUBLICATION_FILES - {"publication.json"}
              if (staged/name).is_file()}
    atomic_save(hashes, staged/"publication.json", json_file=True)
