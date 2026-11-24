#!/bin/bash
# Submit gate separately, then formal train + frozen evaluation after gate review.
set -euo pipefail
phase="${1:-gate}"
[[ "$phase" == gate || "$phase" == formal || "$phase" == all ]] || { echo 'Use gate, formal, or all' >&2; exit 2; }
: "${RUN_ID:?Choose a new SR+MLS run ID}" "${MRI_CASE_INVENTORY:?}" "${MRI_TEST_MANIFEST:?}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$ ]]
export GSBM_PROJECT_ROOT="$PWD" MRI_GPUS=4 RUN_ID MRI_CASE_INVENTORY MRI_TEST_MANIFEST
mkdir -p reports/slurm reports/mri
receipt="reports/mri/${RUN_ID}_sr_jobs.tsv"
if [[ "$phase" != formal ]]; then
    : "${LEGACY_CHECKPOINT:?}"
    export LEGACY_CHECKPOINT
    set -o noclobber
    : > "$receipt"
    set +o noclobber
    gate=$(sbatch --parsable scripts/mri_sr_gate.sbatch)
    gate="${gate%%;*}"
    printf 'gate\t%s\n' "$gate" >> "$receipt"
else
    [[ -s "$receipt" && -s "runs/mri/${RUN_ID}_smoke/bridge/smoke_gate.json" ]]
    gate=$(awk -F '\t' '$1=="gate" {print $2}' "$receipt")
    [[ "$gate" =~ ^[0-9]+$ ]]
    if grep -q '^train' "$receipt"; then echo 'Formal jobs already submitted' >&2; exit 2; fi
fi
if [[ "$phase" != gate ]]; then
    train=$(sbatch --parsable --dependency="afterok:$gate" --kill-on-invalid-dep=yes scripts/mri_sr_train.sbatch)
    train="${train%%;*}"
    printf 'train\t%s\n' "$train" >> "$receipt"
    evaluation=$(sbatch --parsable --dependency="afterok:$train" --kill-on-invalid-dep=yes scripts/mri_sr_eval.sbatch)
    evaluation="${evaluation%%;*}"
    printf 'val_test84\t%s\n' "$evaluation" >> "$receipt"
fi
cat "$receipt"
