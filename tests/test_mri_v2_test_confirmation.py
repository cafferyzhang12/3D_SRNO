import importlib.util
import json
from pathlib import Path

import pytest

from vemgsbm.runtime import fingerprint


def test_frozen_test_selection_uses_only_saved_whole_val_reports(tmp_path):
    path = Path(__file__).resolve().parents[1] / "tools" / "mri_v2_test.py"
    spec = importlib.util.spec_from_file_location("v2_test_confirmation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidates = []
    for arm, psnr in (("exp1", 32.5), ("exp3", 32.6)):
        folder = tmp_path / arm
        folder.mkdir()
        checkpoint = folder / "best.pt"
        checkpoint.write_bytes(arm.encode())
        (folder / "final_val.json").write_text(json.dumps({
            "checkpoint_sha256": fingerprint(checkpoint), "metrics": {"psnr": psnr},
        }), encoding="utf-8")
        candidates.append(folder)

    selected, _, report = module.select_candidate(candidates)
    assert selected == candidates[1] / "best.pt"
    assert report["metrics"]["psnr"] == 32.6
    selected.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="does not match"):
        module.select_candidate(candidates)
