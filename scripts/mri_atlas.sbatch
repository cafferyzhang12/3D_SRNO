#!/bin/bash
#SBATCH --job-name=mri-mls
#SBATCH --qos=high
#SBATCH --output=reports/slurm/mri-mls-%j.out
#SBATCH --error=reports/slurm/mri-mls-%j.err
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=2-00:00:00
set -euo pipefail
cd "${GSBM_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/mri_common.sh
gsbm_environment
: "${CONFIG:?Set CONFIG to an MRI MLS template}"
mkdir -p runs/mri
mkdir "runs/mri/$RUN_ID"
unset REPRESENTATION_CHECKPOINT RESUME

mode=$("$GSBM_PYTHON" - "$CONFIG" <<'PY'
import sys
from vemgsbm.runtime import read_config
cfg = read_config(sys.argv[1])
if not cfg.get('atlas'):
    raise ValueError('Expected MRI MLS atlas config')
print(cfg['mode'])
PY
)
for profile in validate formal; do
    representation_config=$(gsbm_config "$CONFIG" representation "$profile")
    run="$PWD/runs/mri/$RUN_ID"
    if [[ "$profile" == validate ]]; then run="$run/smoke"; fi
    inventory="$run/mri_case_inventory.json.gz"
    echo "Preparing MRI inventory before torchrun ($profile)"
    "$GSBM_PYTHON" -m vemgsbm.mri_inventory --config "$representation_config" --output "$inventory"
    inventory_option=(--case-inventory "$inventory")
    gsbm_torchrun -m vemgsbm.train --config "$representation_config" --stage representation --device cuda "${inventory_option[@]}"
    export REPRESENTATION_CHECKPOINT="$run/representation/best.pt"
    bridge_config=$(gsbm_config "$CONFIG" bridge "$profile")
    if [[ "$mode" == gsbm ]]; then
        gsbm_torchrun -m vemgsbm.train --config "$bridge_config" --stage bridge --device cuda "${inventory_option[@]}"
        export CHECKPOINT="$run/bridge/best.pt"
    else
        gsbm_torchrun -m vemgsbm.momentum.train --config "$bridge_config" --device cuda "${inventory_option[@]}"
        export CHECKPOINT="$run/momentum/best.pt"
    fi
    if [[ "$profile" == validate ]]; then
        mapfile -t validation < <("$GSBM_PYTHON" - "$bridge_config" <<'PY'
import sys
from vemgsbm.runtime import read_config
cfg = read_config(sys.argv[1])
print(cfg['data']['val']['root_lq'])
print(cfg['data']['val']['root_hq'])
print(cfg['sampling']['steps'])
PY
)
        [[ "${#validation[@]}" == 3 ]]
        gsbm_torchrun -m vemgsbm.mri_infer --checkpoint "$CHECKPOINT" \
            --split val --root-lq "${validation[0]}" --root-hq "${validation[1]}" \
            --scan-cache "$run/index/val_inference.json" \
            --output-root "outputs/mri/${RUN_ID}_validation_smoke" --device cuda \
            --max-cases 1 --test-fraction 1 --steps "${validation[2]}" --save-volume-limit 0 \
            --inference-variant endpoint_safe
    fi
    unset REPRESENTATION_CHECKPOINT
done
export MRI_CASE_INVENTORY="$inventory"
source scripts/mri_whole_val.sh
