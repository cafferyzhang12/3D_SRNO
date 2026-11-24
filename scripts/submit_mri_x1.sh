#!/bin/bash
# Submit only; all statistics, training and evaluation run inside A800 allocations.
set -euo pipefail
: "${RUN_ID:?Choose a new x1 run ID}" "${MRI_CASE_INVENTORY:?}" "${LEGACY_CHECKPOINT:?}" "${ODE_ABLATION:?}" "${ABLATION_JOB:?}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$ ]]
export GSBM_PROJECT_ROOT="$PWD" MRI_GPUS=4
mkdir -p reports/slurm reports/mri
receipt="reports/mri/${RUN_ID}_jobs.tsv"
set -o noclobber
: > "$receipt"
set +o noclobber
gate=$(sbatch --parsable --dependency="afterok:$ABLATION_JOB" --kill-on-invalid-dep=yes scripts/mri_x1_gate.sbatch)
printf 'gate\t%s\n' "$gate" >> "$receipt"
train=$(sbatch --parsable --dependency="afterok:$gate" --kill-on-invalid-dep=yes scripts/mri_x1_train.sbatch)
printf 'train\t%s\n' "$train" >> "$receipt"
val=$(sbatch --parsable --dependency="afterok:$train" --kill-on-invalid-dep=yes scripts/mri_x1_val.sbatch)
printf 'val\t%s\n' "$val" >> "$receipt"
cat "$receipt"
