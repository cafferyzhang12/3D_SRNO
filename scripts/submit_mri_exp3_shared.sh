#!/bin/bash
# This chain is independent of the running exp1 allocation and has no exp2 arm.
set -euo pipefail
: "${SHARED_STUDY_DIR:?}" "${EXP1_PROJECT_ROOT:?}" "${MRI_PERCEPTUAL_WEIGHTS:?}"
export GSBM_PROJECT_ROOT="$PWD" MRI_GPUS=4 SHARED_STUDY_DIR EXP1_PROJECT_ROOT MRI_PERCEPTUAL_WEIGHTS
source scripts/mri_slurm_dependencies.sh
mkdir -p reports/slurm reports/mri
receipt="$SHARED_STUDY_DIR/jobs.tsv"
(set -o noclobber; : > "$receipt")
exclude=()
[[ -z "${PROTECTED_EXP1_NODE:-}" ]] || exclude=(--exclude "$PROTECTED_EXP1_NODE")
gate=$(mri_slurm_submit "" "${exclude[@]}" scripts/mri_exp3_shared_gate.sbatch)
printf 'gate\t%s\n' "$gate" >> "$receipt"
train=$(mri_slurm_submit "$gate" "${exclude[@]}" scripts/mri_exp3_shared_train.sbatch)
printf 'exp3_train_once\t%s\n' "$train" >> "$receipt"
paired=$(mri_slurm_submit "$train" "${exclude[@]}" scripts/mri_exp3_shared_eval.sbatch)
printf 'same_weight_offline_online_val\t%s\n' "$paired" >> "$receipt"
cat "$receipt"
