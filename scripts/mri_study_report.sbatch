#!/bin/bash
#SBATCH --job-name=mri-x4-report
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
history="$HISTORY_ROOT/outputs/mri/mri_x4_gsbm_mls_v2_20261120_test15/summary.json"
options=()
if [[ -n "${COHORT_DIR:-}" && -s "$COHORT_DIR/cohort.json" ]]; then
    options+=(--independent-root "$FREEZE_ROOT/../archive84")
fi
"$GSBM_PYTHON" -m vemgsbm.mri_final_report --matrix "$FREEZE_ROOT/matrix.json" \
    --historical "$history" --output "$FREEZE_ROOT/../report" "${options[@]}"
