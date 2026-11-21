#!/bin/bash
#SBATCH --job-name=mri-val-select
#SBATCH --qos=high
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=reports/slurm/%x-%j.out
#SBATCH --error=reports/slurm/%x-%j.err
set -euo pipefail
cd "${GSBM_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/mri_common.sh
gsbm_environment
if [[ "$STUDY_ACTION" == bank ]]; then
    folder="reports/mri/studies/$STUDY_ID"
    read -r checkpoint < "$folder/descriptor.txt"
    direct_run=$(dirname "$(dirname "$checkpoint")")
    "$GSBM_PYTHON" -m vemgsbm.spatial_atlas --checkpoint "$checkpoint" \
        --case-inventory "${MRI_CASE_INVENTORY:-$direct_run/inventory.json.gz}" \
        --output "$folder/spatial_bank.pt" --device cuda
    exit
fi
"$GSBM_PYTHON" -m vemgsbm.mri_haar_sweep "$STUDY_ACTION" --study "$STUDY_ID" --stage "$STUDY_STAGE"
