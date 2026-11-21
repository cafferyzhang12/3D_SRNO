#!/bin/bash
#SBATCH --job-name=gsbm-mri-validate
#SBATCH --qos=high
#SBATCH --output=reports/slurm/mri-validate-%j.out
#SBATCH --error=reports/slurm/mri-validate-%j.err
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
cd "${GSBM_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/mri_common.sh
gsbm_environment
export RUN_ID="${RUN_ID}_gate_${SLURM_JOB_ID}"
CONFIG="${CONFIG:-configs/mri_x4.yaml}"
[[ -z "${RESUME:-}" ]] || { echo 'Validation uses a new run; unset RESUME.' >&2; exit 2; }
mkdir -p runs/mri
mkdir "runs/mri/$RUN_ID"
"$GSBM_PYTHON" -m pytest -q
representation_config=$(gsbm_config "$CONFIG" representation validate)
gsbm_torchrun -m vemgsbm.train --config "$representation_config" --stage representation --device cuda
export REPRESENTATION_CHECKPOINT="$PWD/runs/mri/$RUN_ID/representation/best.pt"
bridge_config=$(gsbm_config "$CONFIG" bridge validate)
gsbm_torchrun -m vemgsbm.train --config "$bridge_config" --stage bridge --device cuda
mapfile -t roots < <("$GSBM_PYTHON" - "$bridge_config" <<'PY'
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8'))
print(cfg['data']['val']['root_lq'])
print(cfg['data']['val']['root_hq'])
print(cfg['data']['scan_cache']['val'])
PY
)
[[ "${#roots[@]}" == 3 ]] || { echo 'Could not read validation roots' >&2; exit 2; }
gsbm_torchrun -m vemgsbm.mri_infer --checkpoint "runs/mri/$RUN_ID/bridge/best.pt" \
    --split val --root-lq "${roots[0]}" --root-hq "${roots[1]}" --scan-cache "${roots[2]}" \
    --output-root "outputs/mri/${RUN_ID}_validation_smoke" --device cuda \
    --max-cases 1 --test-fraction 1 --steps 2 --save-volume-limit 0
printf 'VALIDATION_PASSED run_id=%s job_id=%s\n' "$RUN_ID" "$SLURM_JOB_ID"
