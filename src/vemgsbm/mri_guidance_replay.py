"""Read-only replay of actual chart weights in every epoch-38 VAL guidance update."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from .atlas_geometry import BoundSourceMRIAtlas
from .bridge import reconstruct
from .mri_data import read_plane
from .mri_infer import _infer_case
from .mri_inventory import load_inventory
from .mri_stage1 import _check_arm, _tile_from_planes, oracle_fit_scores
from .runtime import atomic_save, build_models, fingerprint, load_checkpoint, setup


class _ObservedChart(BoundSourceMRIAtlas):
    def _local_geometry(self, features, derivative=True):
        local = super()._local_geometry(features, derivative)
        self.observations.append((self.step(), derivative, local["weight"].detach(), local["trust"].detach()))
        return local


class _ObservedGeometry:
    def __init__(self, original, step):
        self.original, self.step = original, step
        self.chart = None

    def bind(self, base):
        actual = self.original.bind(base)
        chart = _ObservedChart(self.original, actual.chart_ids, actual.chart_prior, actual.source_confidence)
        chart.observations, chart.step = [], self.step
        self.chart = chart
        return chart


def trace_tile(model, base, mask, mode, *, geometry, generator, **options):
    comparison = torch.Generator(device=base.device)
    comparison.set_state(generator.get_state())
    reference, differences = [], []

    def save_inputs(state, cond, observed, time, direction):
        reference.append((state.clone(), comparison.get_state(), time, direction))
        return model(state, cond, observed, time, direction)

    ordinary = reconstruct(save_inputs, base, mask, mode, geometry=geometry, generator=comparison, **options)
    calls = 0
    def verify_inputs(state, cond, observed, time, direction):
        nonlocal calls
        other, rng, time0, direction0 = reference[calls]
        calls += 1
        if (time, direction) != (time0, direction0) or not torch.equal(rng, generator.get_state()):
            raise RuntimeError(f"Replay time, direction or RNG stream diverged at drift call {calls}")
        differences.append((state - other).abs().amax())
        return model(state, cond, observed, time, direction)

    tapped = _ObservedGeometry(geometry, lambda: calls)
    replay = reconstruct(verify_inputs, base, mask, mode, geometry=tapped, generator=generator, **options)
    if (calls != len(reference) or torch.stack(differences).amax().item() != 0
            or not torch.equal(ordinary, replay) or not torch.equal(generator.get_state(), comparison.get_state())):
        raise RuntimeError("Observed and ordinary guided trajectories, final outputs or RNG streams differ")
    return replay, tapped.chart, calls


def _collect_chart(chart, fits, totals):
    oracle = fits.amin(1, keepdim=True)
    gaps = fits.gather(1, chart.chart_ids) - oracle
    best = (gaps == 0).float()
    for step, derivative, w, trust in chart.observations:
        branch = "normal_at_origin" if derivative else "position_at_provisional_state"
        values = {"updates": len(w), "lr_prior_gap": float((chart.chart_prior * gaps).sum()),
                  "actual_weights_gap": float((w * gaps).sum()),
                  "lr_prior_oracle_mass": float((chart.chart_prior * best).sum()),
                  "actual_oracle_mass": float((w * best).sum())}
        if derivative:
            # The normal-action cost multiplies the dynamic mixture by chart
            # trust. Report its actual mass and residual contribution before
            # any normalization; missing mass means weaker guidance.
            trusted = w * trust
            values.update(trust_weight_mass=float(trusted.sum()), trust_weight_gap=float((trusted * gaps).sum()),
                          trust_weight_oracle_mass=float((trusted * best).sum()))
        totals[(step, branch)].update(values)


def _rows(totals):
    rows = []
    for (step, branch), counter in sorted(totals.items()):
        summary = {k:v/counter["updates"] for k,v in counter.items() if k != "updates"}
        if counter["trust_weight_mass"]:
            summary["trusted_gap_per_unit_mass"] = counter["trust_weight_gap"] / counter["trust_weight_mass"]
        rows.append({"step": step, "branch":branch, "updates": counter["updates"], **summary})
    return rows


def run(checkpoint_path, inventory, source, output, device_name="cuda", seed=42):
    """Compare the observed sampler, the ordinary sampler and the published guided outputs."""
    source, output = Path(source), Path(output).resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    cfg = checkpoint["config"]
    if checkpoint["epoch"] != 38 or cfg["mode"] != "gsbm":
        raise ValueError("Expected the historical epoch-38 guided checkpoint")
    inventory_cases, _ = load_inventory(inventory, cfg["data"], ("val",))
    cases = inventory_cases["val"]
    published = source / "guided"
    _check_arm(published, cases, "guided", fingerprint(checkpoint_path), seed)
    if output.exists():
        raise FileExistsError(output)
    device = setup(seed, device_name, cfg["train"].get("threads",4))
    cfg = {**cfg, "sampling": {**cfg["sampling"],"inference_sigma":.05}, "inference_variant":"endpoint_safe"}
    model, geometry = build_models({**checkpoint, "config":cfg}, device)
    patch = cfg["data"]["patch_size"]
    output.parent.mkdir(parents=True, exist_ok=True)
    pooled, reports = defaultdict(Counter), []
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp)/"report"
        stage.mkdir()
        for case in cases:
            counters, tiles, compared_calls = defaultdict(Counter), 0, 0
            @lru_cache(maxsize=2 * patch[0])
            def hr(index):
                return read_plane(case.hq[index], case.normalization_scale, case.normalization_offset,
                                  case.channel_policy, case.bits)

            def replay_tile(*args, tile_location, **kwargs):
                nonlocal tiles, compared_calls
                run_bounds, starts = tile_location
                if len(starts) != 1:
                    raise ValueError("Epoch-38 replay requires the original one-tile sampler")
                result, chart, n = trace_tile(*args, **kwargs)
                # Access HR after both complete integrations. Observers only
                # retain detached weights. Neither path takes HR as input.
                reference = _tile_from_planes(hr, run_bounds, starts[0], patch).to(device)
                _collect_chart(chart, oracle_fit_scores(geometry, reference), counters)
                tiles += 1
                compared_calls += n
                return result

            with tempfile.TemporaryDirectory(dir=tmp) as case_dir:
                dest = Path(case_dir)/case.id
                _infer_case(case,dest,model,geometry,cfg,device,seed,20,1,replay_tile=replay_tile)
                orig = np.load(published/case.id/"mean.npy",mmap_mode="r")
                replay = np.load(dest/"mean.npy",mmap_mode="r")
                digest = hashlib.sha256()
                for z in range(len(orig)):
                    if not np.array_equal(orig[z], replay[z]):
                        raise RuntimeError(f"Published guided volume differs from replay at {case.id}, slice {z}")
                    digest.update(replay[z].tobytes())
                del orig, replay
            if not counters:
                raise ValueError("Guided replay recorded zero actual updates")
            for key, counts in counters.items():
                pooled[key].update(counts)
            reports.append({"case_id":case.id, "tiles":tiles, "drift_input_and_rng_matches":compared_calls,
                            "volume_pixel_sha256":digest.hexdigest(),"updates_by_step":_rows(counters)})
            print(f"Verified replay of {tiles} tiles for {case.id}", flush=True)
        aggregate = defaultdict(Counter)
        for (_, branch), counts in pooled.items():
            aggregate[("all", branch)].update(counts)
        atomic_save({"version":1, "split":"val", "seed":seed, "steps":20, "sampling_sigma":.05,
                     "checkpoint_sha256":fingerprint(checkpoint_path),"guided_summary_sha256":fingerprint(published/"summary.json"),
                     "cases":reports,"pooled_by_step":_rows(pooled),"pooled_all_steps":_rows(aggregate)},
                    stage/"summary.json",json_file=True)
        os.rename(stage, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint","case-inventory","stage1-output","output"):
        parser.add_argument("--"+key,required=True)
    parser.add_argument("--device",default="cuda")
    args = parser.parse_args()
    run(args.checkpoint,args.case_inventory,args.stage1_output,args.output,args.device)


if __name__ == "__main__":
    main()
