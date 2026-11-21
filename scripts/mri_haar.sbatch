#!/bin/bash
#SBATCH --job-name=mri-haar
#SBATCH --qos=high
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=reports/slurm/%x-%j.out
#SBATCH --error=reports/slurm/%x-%j.err
set -euo pipefail
cd "${GSBM_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/mri_common.sh
gsbm_environment
case "$VARIANT" in
    direct) template=configs/mri_haar_direct_x4.yaml;;
    *) template=configs/mri_haar_bridge_x4.yaml;;
esac
run="runs/mri/$RUN_ID"
mkdir -p "$run"
exec 9>"$run/.allocation.lock"
flock -n 9
config="$run/experiment.yaml"
args=(--variant "$VARIANT" --seed "${SEED:-42}" --dim "${DIM:-2}" --query-chunk "${QUERY_CHUNK:-2048}"
      --query "${QUERY:-endpoint}" --reference "${REFERENCE:-identity}")
if [[ -n "${DIRECT_CHECKPOINT:-}" ]]; then args+=(--direct-checkpoint "$DIRECT_CHECKPOINT"); fi
if [[ -n "${SPATIAL_BANK:-}" ]]; then args+=(--bank "$SPATIAL_BANK"); fi
if [[ "${PATH_GEOMETRY:-0}" == 1 ]]; then args+=(--path-geometry); fi
if [[ "${SHUFFLE_ROUTING:-0}" == 1 ]]; then args+=(--shuffle-routing); fi
template=$(gsbm_config "$template" bridge)
"$GSBM_PYTHON" -m vemgsbm.mri_haar_experiment config --template "$template" --output "$config" "${args[@]}"
inventory="${MRI_CASE_INVENTORY:-$run/inventory.json.gz}"
if [[ ! -f "$inventory" ]]; then
    "$GSBM_PYTHON" -m vemgsbm.mri_inventory --config "$config" --output "$inventory"
fi
if [[ "$PHASE" == gate ]]; then
    [[ "$MRI_GPUS" == 4 ]]
    "$GSBM_PYTHON" -m pytest -q
    gsbm_torchrun -m vemgsbm.mri_haar_experiment overfit --config "$config" --inventory "$inventory" \
        --directory "$run/smoke" --steps "${OVERFIT_STEPS:-24}" --device cuda
    exit
fi
[[ "$PHASE" == train && -s "$run/smoke/smoke.json" ]]
resume=()
if [[ -f "$run/bridge/last.pt" ]]; then resume+=(--resume "$run/bridge/last.pt"); fi
gsbm_torchrun -m vemgsbm.train --config "$config" --case-inventory "$inventory" \
    --stage bridge --device cuda "${resume[@]}"
export CHECKPOINT="$PWD/$run/bridge/best.pt" MRI_CASE_INVENTORY="$inventory"
export OUTPUT_ROOT="outputs/mri/${RUN_ID}_val"
if [[ ! -f "$OUTPUT_ROOT/summary.json" ]]; then
    source scripts/mri_whole_val.sh
fi
"$GSBM_PYTHON" - "$OUTPUT_ROOT/summary.json" "$CHECKPOINT" <<'PY'
import json,sys
from pathlib import Path
from vemgsbm.runtime import fingerprint
s=json.loads(Path(sys.argv[1]).read_text())
if s['split'] != 'val' or s['checkpoint_sha256'] != fingerprint(sys.argv[2]) or not s['total_slices']:
    raise ValueError('Incomplete or stale whole-case validation')
print('WHOLE_VAL_COMPLETE',s['all_psnr_db'],s['mean_slice_ssim'])
PY
if [[ "$VARIANT" == bridge ]]; then
    # Re-evaluate the very same supervised B weights at the integral readout.
    export OUTPUT_ROOT="outputs/mri/${RUN_ID}_integral_val" MRI_HAAR_READOUT=integral
    if [[ ! -f "$OUTPUT_ROOT/summary.json" ]]; then
        source scripts/mri_whole_val.sh
    fi
fi
