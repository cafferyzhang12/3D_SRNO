#!/bin/bash
#SBATCH --job-name=mri-x4-s1
#SBATCH --qos=high
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=2-00:00:00
#SBATCH --output=reports/slurm/%x-%j.out
#SBATCH --error=reports/slurm/%x-%j.err
set -euo pipefail
cd "${GSBM_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/mri_common.sh
gsbm_environment
: "${HISTORY_ROOT:?Set historical snapshot directory}"
: "${ARM:?Set stage 1 arm}"
run="$HISTORY_ROOT/runs/mri/mri_x4_gsbm_mls_v2_20261120"
checkpoint="$run/bridge/best.pt"
inventory="$run/mri_case_inventory.json.gz"
folder="outputs/mri/$RUN_ID"
if [[ "$ARM" == smoke || "$ARM" == bilinear ]]; then
    mapfile -t roots < <("$GSBM_PYTHON" - "$checkpoint" <<'PY'
import sys
from vemgsbm.runtime import load_checkpoint
val = load_checkpoint(sys.argv[1])["config"]["data"]["val"]
print(val["root_lq"])
print(val["root_hq"])
PY
)
    args=()
    if [[ "$ARM" == smoke ]]; then
        "$GSBM_PYTHON" -m pytest -q
        args+=(--max-cases 1 --steps 2)
    else
        args+=(--predictor bilinear)
    fi
    gsbm_torchrun -m vemgsbm.mri_infer --checkpoint "$checkpoint" --case-inventory "$inventory" \
        --output-root "$folder/$ARM" --device cuda --split val --root-lq "${roots[0]}" --root-hq "${roots[1]}" \
        --inference-variant endpoint_safe --save-volume-limit 0 --seed 42 "${args[@]}"
else
    "$GSBM_PYTHON" -m vemgsbm.mri_stage1 --checkpoint "$checkpoint" --case-inventory "$inventory" \
        --output-root "$folder" --device cuda --seed 42 --arm "$ARM"
fi
