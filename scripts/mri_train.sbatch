#!/bin/bash
#SBATCH --job-name=gsbm-mri-train
#SBATCH --qos=high
#SBATCH --output=reports/slurm/mri-train-%j.out
#SBATCH --error=reports/slurm/mri-train-%j.err
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
: "${CONFIG:?Set CONFIG}"
: "${RUN_ID:?Set RUN_ID}"
STAGE="${STAGE:-bridge}"
[[ "$STAGE" == representation || "$STAGE" == bridge ]] || { echo 'Invalid STAGE' >&2; exit 2; }
cd "${GSBM_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/mri_common.sh
gsbm_environment
effective_config=$(gsbm_config "$CONFIG" "$STAGE")
extra=()
if [[ -n "${RESUME:-}" ]]; then extra+=(--resume "$RESUME"); fi
gsbm_torchrun -m vemgsbm.train --config "$effective_config" --stage "$STAGE" --device cuda "${extra[@]}"
