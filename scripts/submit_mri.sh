#!/bin/bash
# Administrative submission only. No image reads, imports, or compute on login nodes.
set -euo pipefail
[[ $# == 1 ]] || { echo 'Usage: bash scripts/submit_mri.sh {validate|train|val|test}' >&2; exit 2; }
action="$1"
[[ "$action" == validate || "$action" == train || "$action" == test || "$action" == val ]] || { echo 'Unknown action' >&2; exit 2; }
cd "${GSBM_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export RUN_ID="${RUN_ID:-mri_gsbm_${action}_$(date +%Y%m%d_%H%M%S)_$$}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$ ]] || { echo 'Invalid RUN_ID' >&2; exit 2; }
export CONFIG="${CONFIG:-configs/mri_x4.yaml}"
[[ -f "$CONFIG" ]] || { echo 'Missing CONFIG' >&2; exit 2; }
if [[ "$action" == train ]]; then
    [[ "${STAGE:-bridge}" == representation || "${STAGE:-bridge}" == bridge ]] || { echo 'Invalid STAGE' >&2; exit 2; }
fi
if [[ "$action" == test ]]; then
    : "${FREEZE_DIR:?Set FREEZE_DIR}"
    export EXPORT_CASE_LIMIT="${EXPORT_CASE_LIMIT:-8}"
    [[ "$EXPORT_CASE_LIMIT" =~ ^[0-9]+$ ]] || { echo 'EXPORT_CASE_LIMIT must be nonnegative' >&2; exit 2; }
fi
for value in "${BATCH_SIZE:-}" "${EPOCHS:-}" "${SAMPLES_PER_EPOCH:-}" "${VAL_SAMPLES_PER_EPOCH:-}"; do
    [[ -z "$value" || "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Epoch/sample/batch overrides must be positive integers' >&2; exit 2; }
done
export MRI_GPUS="${MRI_GPUS:-1}"
[[ "$MRI_GPUS" == 1 || "$MRI_GPUS" == 4 ]] || exit 2
options=(--gres="gpu:$MRI_GPUS" --cpus-per-task="$((6*MRI_GPUS))")
if [[ -n "${DEPENDENCY_JOB_ID:-}" ]]; then
    [[ "$DEPENDENCY_JOB_ID" =~ ^[0-9]+$ ]] || { echo 'Invalid dependency job number' >&2; exit 2; }
    options+=(--dependency="afterok:$DEPENDENCY_JOB_ID")
fi
if [[ -n "${TIME_LIMIT:-}" ]]; then
    [[ "$TIME_LIMIT" =~ ^([0-9]+-)?[0-9]+(:[0-9]{1,2}){0,2}$ ]] || { echo 'Invalid TIME_LIMIT' >&2; exit 2; }
    options+=(--time="$TIME_LIMIT")
fi
mkdir -p reports/slurm reports/mri/submissions
record="reports/mri/submissions/${RUN_ID}_${action}_$(date +%Y%m%d_%H%M%S)_$$.txt"
(set -o noclobber; printf 'run_id=%s\naction=%s\nstage=%s\nconfig=%s\nqos=high\ngpus=%s\nenvironment=fundiff\ndependency=%s\n' \
    "$RUN_ID" "$action" "${STAGE:-none}" "$CONFIG" "${DEPENDENCY_JOB_ID:-none}" "$MRI_GPUS" > "$record")
response=$(sbatch --parsable --qos=high --kill-on-invalid-dep=yes --export=ALL "${options[@]}" "scripts/mri_${action}.sbatch")
job="${response%%;*}"
[[ "$job" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch response: $response" >&2; exit 2; }
printf 'job_id=%s\n' "$job" >> "$record"
printf '%s\n' "$job"
