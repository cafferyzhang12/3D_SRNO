#!/bin/bash
#SBATCH --qos=high
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --output=reports/slurm/%x-%j.out
#SBATCH --error=reports/slurm/%x-%j.err
#SBATCH --job-name=mri-ssim-gate
#SBATCH --time=08:00:00
set -euo pipefail
cd "${GSBM_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/mri_common.sh
gsbm_environment
export XDG_CACHE_HOME="$PWD/cache/xdg"
export TORCH_HOME="$PWD/cache/perceptual/torch"
: "${RUN_ID:?}" "${MRI_CASE_INVENTORY:?}" "${MRI_TEST_MANIFEST:?}" "${BASELINE_CHECKPOINT:?}" "${MRI_MLS_BANK:?}"
study="runs/mri/$RUN_ID"
mkdir -p outputs
"$GSBM_PYTHON" -m pytest -q --basetemp="outputs/ssim_pytest_${SLURM_JOB_ID}" --junitxml="reports/mri/ssim_pytest_${SLURM_JOB_ID}.xml"
"$GSBM_PYTHON" -m vemgsbm.mri_perceptual prepare-weights --weights-dir "$PWD/cache/perceptual" --device cuda
"$GSBM_PYTHON" -m vemgsbm.mri_ssim_pipeline prepare --baseline "$BASELINE_CHECKPOINT" --inventory "$MRI_CASE_INVENTORY" --bank "$MRI_MLS_BANK" --run-id "$RUN_ID"
"$GSBM_PYTHON" -m vemgsbm.mri_x1_pipeline prebuild-mls --config "$study/neighbor.yaml" --inventory "$MRI_CASE_INVENTORY" --device cuda
gsbm_torchrun -m vemgsbm.train --config "$study/smoke.yaml" --case-inventory "$MRI_CASE_INVENTORY" --device cuda
"$GSBM_PYTHON" -m vemgsbm.mri_ssim_pipeline check-smoke --config "$study/smoke.yaml"
