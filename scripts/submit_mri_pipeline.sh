#!/bin/bash
# Development pipeline: correctness gate, representation, bridge and whole-case VAL.
set -euo pipefail
[[ $# == 0 ]] || { echo 'Configure through environment variables only.' >&2; exit 2; }
cd "${GSBM_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export RUN_ID="${RUN_ID:-mri_x4_gsbm_$(date +%Y%m%d_%H%M%S)_$$}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,75}$ ]] || { echo 'Invalid RUN_ID' >&2; exit 2; }
[[ -z "${RESUME:-}" ]] || { echo 'New pipeline forbids RESUME; submit an explicit train job to resume.' >&2; exit 2; }
export MRI_GPUS="${MRI_GPUS:-1}"
export CONFIG="${CONFIG:-configs/mri_x4.yaml}"
[[ -f "$CONFIG" ]] || { echo 'Missing CONFIG' >&2; exit 2; }
export EXPORT_CASE_LIMIT="${EXPORT_CASE_LIMIT:-8}"
[[ "$EXPORT_CASE_LIMIT" =~ ^[0-9]+$ ]] || { echo 'Invalid EXPORT_CASE_LIMIT' >&2; exit 2; }
for value in "${BATCH_SIZE:-}" "${REPRESENTATION_BATCH_SIZE:-}" "${REPRESENTATION_EPOCHS:-}" \
    "${BRIDGE_EPOCHS:-}" "${SAMPLES_PER_EPOCH:-}" "${VAL_SAMPLES_PER_EPOCH:-}"; do
    [[ -z "$value" || "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Overrides must be positive integers.' >&2; exit 2; }
done
mkdir -p reports/slurm reports/mri/pipelines runs/mri
record="reports/mri/pipelines/${RUN_ID}.txt"
(set -o noclobber; printf 'run_id=%s\nconfig=%s\nqos=high\ngpus=%s\nenvironment=fundiff\ntest_fraction=0.15\nsubset_seed=20261110\n' "$RUN_ID" "$CONFIG" "$MRI_GPUS" > "$record")
trap 'status=$?; printf "submission_exit=%s\n" "$status" >> "$record"' EXIT
mkdir "runs/mri/$RUN_ID" || { echo 'RUN_ID already exists; choose a new name.' >&2; exit 2; }
[[ ! -e "${OUTPUT_ROOT:-outputs/mri/${RUN_ID}_test15}" ]] || { echo 'Test output already exists.' >&2; exit 2; }
gate_job=$(DEPENDENCY_JOB_ID= REPRESENTATION_CHECKPOINT= bash scripts/submit_mri.sh validate)
printf 'validation_job=%s\n' "$gate_job" >> "$record"
representation_job=$(STAGE=representation REPRESENTATION_CHECKPOINT= \
    EPOCHS="${REPRESENTATION_EPOCHS:-}" BATCH_SIZE="${REPRESENTATION_BATCH_SIZE:-${BATCH_SIZE:-}}" \
    DEPENDENCY_JOB_ID="$gate_job" TIME_LIMIT="${REPRESENTATION_TIME_LIMIT:-}" bash scripts/submit_mri.sh train)
printf 'representation_job=%s\nrepresentation_dependency=afterok:%s\n' "$representation_job" "$gate_job" >> "$record"
representation_checkpoint="$PWD/runs/mri/$RUN_ID/representation/best.pt"
bridge_job=$(STAGE=bridge REPRESENTATION_CHECKPOINT="$representation_checkpoint" \
    EPOCHS="${BRIDGE_EPOCHS:-}" DEPENDENCY_JOB_ID="$representation_job" \
    TIME_LIMIT="${BRIDGE_TIME_LIMIT:-}" bash scripts/submit_mri.sh train)
printf 'bridge_job=%s\nbridge_dependency=afterok:%s\nrepresentation_checkpoint=%s\n' \
    "$bridge_job" "$representation_job" "$representation_checkpoint" >> "$record"
checkpoint="$PWD/runs/mri/$RUN_ID/bridge/best.pt"
val_job=$(CHECKPOINT="$checkpoint" DEPENDENCY_JOB_ID="$bridge_job" bash scripts/submit_mri.sh val)
printf 'whole_val_job=%s\ncheckpoint=%s\n' "$val_job" "$checkpoint" >> "$record"
cat "$record"
