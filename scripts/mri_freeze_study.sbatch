#!/bin/bash
#SBATCH --job-name=mri-x4-freeze
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
"$GSBM_PYTHON" tools/mri_freeze_study.py --study "reports/mri/studies/$STUDY_ID/replicate.json" \
    --stage1 "$STAGE1_RESULTS" --legacy-results "$LEGACY_RESULTS" --archive-root "$ARCHIVE_ROOT" \
    --history-root "$HISTORY_ROOT" --external-root "$EXTERNAL_RESULTS_ROOT" --folder "$FREEZE_ROOT" --external-prefix "$EXTERNAL_PREFIX"
