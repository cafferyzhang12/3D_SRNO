#!/bin/bash
#SBATCH --job-name=mri-sr-mls-gate
#SBATCH --qos=high
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --output=reports/slurm/%x-%j.out
#SBATCH --error=reports/slurm/%x-%j.err
set -euo pipefail
cd "${GSBM_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/mri_common.sh
gsbm_environment
: "${MRI_CASE_INVENTORY:?}" "${LEGACY_CHECKPOINT:?}"
mkdir -p outputs reports/mri
"$GSBM_PYTHON" -m pytest -q --basetemp="outputs/sr_pytest_${SLURM_JOB_ID}" --junitxml="reports/mri/sr_pytest_${SLURM_JOB_ID}.xml"
stats=()
if [[ -n "${MRI_HAAR_STATISTICS:-}" ]]; then stats+=(--statistics-cache "$MRI_HAAR_STATISTICS"); fi
"$GSBM_PYTHON" -m vemgsbm.mri_x1_pipeline prepare-sr --template configs/mri_haar_sr_mls_x4.yaml \
    --legacy-checkpoint "$LEGACY_CHECKPOINT" --inventory "$MRI_CASE_INVENTORY" --run-id "$RUN_ID" "${stats[@]}"
config="runs/mri/$RUN_ID/experiment.yaml"
# Full TRAIN scanning and bank construction finish before NCCL initializes.
"$GSBM_PYTHON" -m vemgsbm.mri_x1_pipeline prebuild-mls --config "$config" --inventory "$MRI_CASE_INVENTORY" --device cuda
"$GSBM_PYTHON" -m vemgsbm.mri_x1_pipeline fixed-batch-sr --config "$config" --inventory "$MRI_CASE_INVENTORY" \
    --directory "runs/mri/$RUN_ID/fixed_batch_gate" --steps 24 --device cuda
smoke="runs/mri/$RUN_ID/smoke.yaml"
resume=()
if [[ -f "runs/mri/${RUN_ID}_smoke/bridge/last.pt" ]]; then resume+=(--resume "runs/mri/${RUN_ID}_smoke/bridge/last.pt"); fi
gsbm_torchrun -m vemgsbm.train --config "$smoke" --case-inventory "$MRI_CASE_INVENTORY" --device cuda "${resume[@]}"
"$GSBM_PYTHON" -m vemgsbm.mri_x1_pipeline check-sr-smoke --config "$smoke"
