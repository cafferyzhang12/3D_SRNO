#!/bin/bash
#SBATCH --job-name=mri-momentum-tdm
#SBATCH --qos=high
#SBATCH --output=reports/slurm/mri-momentum-tdm-%j.out
#SBATCH --error=reports/slurm/mri-momentum-tdm-%j.err
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
unset REPRESENTATION_CHECKPOINT
: "${CONFIG:=configs/mri_momentum_tdm_x4.yaml}"
mkdir -p runs/mri
mkdir "runs/mri/$RUN_ID"
for profile in validate formal; do
    config=$(gsbm_config "$CONFIG" bridge "$profile")
    run="$PWD/runs/mri/$RUN_ID"
    if [[ "$profile" == validate ]]; then run="$run/smoke"; fi
    inventory="$run/mri_case_inventory.json.gz"
    "$GSBM_PYTHON" -m vemgsbm.mri_inventory --config "$config" --output "$inventory"
    gsbm_torchrun -m vemgsbm.momentum.train --config "$config" --device cuda --case-inventory "$inventory"
    export CHECKPOINT="$run/momentum/best.pt"
    if [[ "$profile" == validate ]]; then
        mapfile -t validation < <("$GSBM_PYTHON" - "$config" <<'PY'
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
            --root-lq "${validation[0]}" --root-hq "${validation[1]}" \
            --scan-cache "$run/index/val_inference.json" \
            --output-root "outputs/mri/${RUN_ID}_validation_smoke" --device cuda \
            --max-cases 1 --test-fraction 1 --steps "${validation[2]}" --save-volume-limit 0 \
            --inference-variant endpoint_safe
    fi
done
export MRI_CASE_INVENTORY="$inventory" INFERENCE_VARIANT=endpoint_safe
exec bash scripts/mri_test.sbatch
