#!/bin/bash
#SBATCH --job-name=mri-trace-guidance
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
: "${HISTORY_ROOT:?Set snapshot from job 5281446}"
: "${STAGE1_OUTPUT:?Set directory containing the completed guided arm}"
: "${REPLAY_OUTPUT:?Set a new directory outside the completed guided arm}"
run="$HISTORY_ROOT/runs/mri/mri_x4_gsbm_mls_v2_20261120"
"$GSBM_PYTHON" -m vemgsbm.mri_guidance_replay --checkpoint "$run/bridge/best.pt" \
    --case-inventory "$run/mri_case_inventory.json.gz" --stage1-output "$STAGE1_OUTPUT" \
    --output "$REPLAY_OUTPUT" --device cuda
