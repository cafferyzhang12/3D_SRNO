#!/bin/bash
set -euo pipefail
: "${RUN_ID:?}" "${MRI_CASE_INVENTORY:?}" "${MRI_TEST_MANIFEST:?}" "${BASELINE_CHECKPOINT:?}" "${MRI_MLS_BANK:?}"
export GSBM_PROJECT_ROOT="$PWD" MRI_GPUS=4 RUN_ID MRI_CASE_INVENTORY MRI_TEST_MANIFEST BASELINE_CHECKPOINT MRI_MLS_BANK
mkdir -p reports/slurm reports/mri
receipt="reports/mri/${RUN_ID}_ssim_jobs.tsv"
( set -o noclobber; : > "$receipt" )
gate=$(sbatch --parsable scripts/mri_ssim_gate.sbatch); gate="${gate%%;*}"
printf 'gate\t%s\n' "$gate" >> "$receipt"
train=$(sbatch --parsable --dependency="afterok:$gate" --kill-on-invalid-dep=yes scripts/mri_ssim_train.sbatch); train="${train%%;*}"
printf 'train\t%s\n' "$train" >> "$receipt"
eval=$(sbatch --parsable --dependency="afterok:$train" --kill-on-invalid-dep=yes scripts/mri_ssim_eval.sbatch); eval="${eval%%;*}"
printf 'eval\t%s\n' "$eval" >> "$receipt"
cat "$receipt"
