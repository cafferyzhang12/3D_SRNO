"""Materialize the checked MRI case inventory before starting torchrun."""

import argparse
from dataclasses import fields
import gzip
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from .datasets import index_mri_data
from .mri_data import MRICase
from .runtime import project_path, read_config


def _case_record(case):
    record = {item.name: getattr(case, item.name) for item in fields(MRICase)}
    for key in ("lq", "hq"):
        record[key] = None if record[key] is None else [path.name for path in record[key]]
    return record


def prepare_inventory(config_path, output_path):
    data = read_config(config_path)["data"]
    if data.get("kind") != "paired_mri_png":
        raise ValueError("The inventory is for paired MRI PNG data")
    output = project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"MRI inventory already exists: {output}")
    cases = index_mri_data(data)
    with NamedTemporaryFile(prefix=output.name + ".", suffix=".tmp", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as stream:
            json.dump({"version": 1, "data": data,
                       "splits": {split: [_case_record(case) for case in values]
                                  for split, values in cases.items()}},
                      stream, separators=(",", ":"), allow_nan=False)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"mri_case_inventory": str(output),
                      "case_counts": {split: len(values) for split, values in cases.items()}}), flush=True)


def load_inventory(path, data, splits):
    with gzip.open(project_path(path), "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    # Cache filenames can differ between historical methods on the same images.
    # Roots, scale, normalization, patch and debug limits remain part of the contract.
    def data_contract(settings):
        return {key: value for key, value in settings.items()
                if key not in {"scan_cache", "legacy_scan_cache"}}

    if payload.get("version") != 1 or data_contract(payload["data"]) != data_contract(data):
        raise ValueError("MRI case inventory and configured data contract differ")
    cases, identity = {}, []
    for split, rows in payload["splits"].items():
        identity.extend({"split": split, "id": row["id"], "fingerprint": row["fingerprint"]} for row in rows)
        if split not in splits:
            continue
        cases[split] = []
        roots = {key: project_path(data[split]["root_" + key]) for key in ("lq", "hq")}
        for row in rows:
            for key, root in roots.items():
                if row[key] is not None:
                    directory = root / row["id"]
                    row[key] = tuple(directory / name for name in row[key])
            for key in ("stems", "indices", "lr_shape"):
                row[key] = tuple(row[key])
            cases[split].append(MRICase(**row))
    if set(cases) != set(splits):
        raise ValueError("MRI case inventory is missing a requested split")
    return cases, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare_inventory(args.config, args.output)


if __name__ == "__main__":
    main()
