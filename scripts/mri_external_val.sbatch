#!/bin/bash
#SBATCH --job-name=mri-external-val
#SBATCH --qos=high
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --output=reports/slurm/%x-%j.out
#SBATCH --error=reports/slurm/%x-%j.err
set -euo pipefail
cd "${GSBM_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/mri_common.sh
gsbm_environment
: "${BASELINE:?Set BASELINE to swinir or res_srdiff}"
: "${PROTOCOL_CHECKPOINT:?Set PROTOCOL_CHECKPOINT to the archive checkpoint with the configured MRI roots}"
: "${MRI_CASE_INVENTORY:?Set MRI_CASE_INVENTORY to the matching cohort}"
export MRI_EXTERNAL_ROOT="${MRI_EXTERNAL_ROOT:-$PWD/references}"
export CHECKPOINT="$PWD/runs/mri/$RUN_ID/model.pt"
if [[ ! -e "$CHECKPOINT" ]]; then
    "$GSBM_PYTHON" -m vemgsbm.mri_external --name "$BASELINE" \
        --weights "$MRI_EXTERNAL_ROOT/weights/$BASELINE.pth" --protocol-checkpoint "$PROTOCOL_CHECKPOINT" \
        --output "$CHECKPOINT"
fi
mapfile -t val < <("$GSBM_PYTHON" - "$CHECKPOINT" "$BASELINE" <<'PY'
import sys
from vemgsbm.runtime import load_checkpoint, project_path
cfg = load_checkpoint(sys.argv[1])["config"]
if cfg["external"]["name"] != sys.argv[2]:
    raise ValueError("RUN_ID already contains a different external baseline")
for k in ("root_lq", "root_hq"):
    print(project_path(cfg["data"]["val"][k]))
PY
)
[[ "${#val[@]}" == 2 ]]
smoke="outputs/mri/${RUN_ID}_val_smoke"
if [[ ! -e "$smoke/summary.json" ]]; then
    gsbm_torchrun -m vemgsbm.mri_infer --split val --checkpoint "$CHECKPOINT" \
        --root-lq "${val[0]}" --root-hq "${val[1]}" --output-root "$smoke" \
        --case-inventory "$MRI_CASE_INVENTORY" --max-cases 1 --device cuda \
        --inference-variant endpoint_safe --save-volume-limit 0
fi
source scripts/mri_whole_val.sh
