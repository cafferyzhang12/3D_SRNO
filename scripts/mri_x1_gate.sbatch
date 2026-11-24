#!/bin/bash
#SBATCH --job-name=mri-x1-gate
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
source scripts/mri_common.sh
gsbm_environment
: "${MRI_CASE_INVENTORY:?}" "${LEGACY_CHECKPOINT:?}" "${ODE_ABLATION:?}"
mkdir -p outputs reports/mri
"$GSBM_PYTHON" -m pytest -q --basetemp="outputs/x1_pytest_${SLURM_JOB_ID}" --junitxml="reports/mri/x1_pytest_${SLURM_JOB_ID}.xml"
"$GSBM_PYTHON" -m vemgsbm.mri_x1_pipeline prepare --template configs/mri_haar_bridge_x4.yaml \
    --legacy-checkpoint "$LEGACY_CHECKPOINT" --inventory "$MRI_CASE_INVENTORY" \
    --ablation "$ODE_ABLATION" --run-id "$RUN_ID"
config="runs/mri/$RUN_ID/experiment.yaml"
# Finish the full-cohort statistics before NCCL starts; no rank waits through the scan.
"$GSBM_PYTHON" -m vemgsbm.haar_stats --config "$config" --case-inventory "$MRI_CASE_INVENTORY" --device cuda
gsbm_torchrun -m vemgsbm.mri_haar_experiment overfit --config "$config" --inventory "$MRI_CASE_INVENTORY" \
    --directory "runs/mri/$RUN_ID/fixed_batch_gate" --steps 24 --device cuda
smoke="runs/mri/$RUN_ID/smoke.yaml"
resume=()
if [[ -f "runs/mri/${RUN_ID}_smoke/bridge/last.pt" ]]; then resume+=(--resume "runs/mri/${RUN_ID}_smoke/bridge/last.pt"); fi
gsbm_torchrun -m vemgsbm.train --config "$smoke" --case-inventory "$MRI_CASE_INVENTORY" --device cuda "${resume[@]}"
"$GSBM_PYTHON" -m vemgsbm.mri_x1_pipeline check-smoke --config "$smoke"
