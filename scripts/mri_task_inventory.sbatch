#!/bin/bash
#SBATCH --job-name=mri-task-inventory
#SBATCH --qos=high
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G
#SBATCH --time=08:00:00
#SBATCH --output=reports/slurm/%x-%j.out
#SBATCH --error=reports/slurm/%x-%j.err
set -euo pipefail
cd "${GSBM_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/mri_common.sh
export MRI_GPUS=1
gsbm_environment
: "${MRI_DATA_ROOT:?Set the paired MRI data root}"
: "${MRI_CASE_INVENTORY:?Set the prepared inventory}"
: "${PROTOCOL_CHECKPOINT:?Set the original protocol checkpoint}"
extra=()
if [[ -n "${FIXED_TEST_MANIFEST:-}" ]]; then extra+=(--fixed-test-manifest "$FIXED_TEST_MANIFEST"); fi
"$GSBM_PYTHON" -m vemgsbm.mri_task_inventory --data-root "$MRI_DATA_ROOT" \
    --protocol-checkpoint "$PROTOCOL_CHECKPOINT" --case-inventory "$MRI_CASE_INVENTORY" \
    --output-dir "reports/mri/$RUN_ID/task_inventory" "${extra[@]}"
