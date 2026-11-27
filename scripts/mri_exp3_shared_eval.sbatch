#!/bin/bash
#SBATCH --qos=high
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=3-00:00:00
#SBATCH --job-name=mri-exp3-shared-pair-val
#SBATCH --output=reports/slurm/%x-%j.out
#SBATCH --error=reports/slurm/%x-%j.err
set -euo pipefail
cd "${GSBM_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
source scripts/mri_common.sh
gsbm_environment
: "${SHARED_STUDY_DIR:?}"
checkpoint="$SHARED_STUDY_DIR/train/best.pt"
[[ -f "$checkpoint" ]] || checkpoint="$SHARED_STUDY_DIR/train/last.pt"
gsbm_torchrun -m vemgsbm.mri_v2_compare --checkpoint "$checkpoint" \
    --case-inventory "$SHARED_STUDY_DIR/trainval_inventory.json.gz" \
    --output "$SHARED_STUDY_DIR/paired_val_$SLURM_JOB_ID" --device cuda
