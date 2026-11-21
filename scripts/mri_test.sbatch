#!/bin/bash
#SBATCH --job-name=mri-x4-test
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
: "${FREEZE_DIR:?Set FREEZE_DIR to the completed validation freeze}"
gsbm_torchrun -m vemgsbm.mri_freeze confirm --folder "$FREEZE_DIR" \
    --output "${OUTPUT_ROOT:-outputs/mri/${RUN_ID}_test15}" --device cuda
