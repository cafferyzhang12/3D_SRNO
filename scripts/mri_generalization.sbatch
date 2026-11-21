#!/bin/bash
#SBATCH --job-name=mri-archive
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
: "${COHORT_DIR:?Set COHORT_DIR for the separate archive confirmation cohort}"
case "${PHASE:?Set PHASE=prepare or confirm}" in
    prepare)
        : "${CHECKPOINT:?Set historical CHECKPOINT}" "${MRI_CASE_INVENTORY:?Set matching inventory}"
        "$GSBM_PYTHON" -m vemgsbm.mri_generalization prepare --checkpoint "$CHECKPOINT" \
            --case-inventory "$MRI_CASE_INVENTORY" --archive "${ARCHIVE_ROOT:-$HOME/data/MRI_SR/validation}" \
            --output "$COHORT_DIR" ;;
    confirm)
        : "${FREEZE_DIR:?Set FREEZE_DIR to the published freeze}"
        "$GSBM_PYTHON" -m vemgsbm.mri_generalization confirm --folder "$FREEZE_DIR" \
            --cohort "$COHORT_DIR" --output "${OUTPUT_ROOT:-outputs/mri/${RUN_ID}_archive84}" --device cuda ;;
    *) exit 2 ;;
esac
