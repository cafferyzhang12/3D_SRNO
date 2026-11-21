#!/bin/bash
# Administrative scheduler. A successful immutable study-wide freeze releases all TEST jobs.
set -euo pipefail
: "${STUDY_ID:?Set STUDY_ID}" "${FREEZE_ROOT:?Set FREEZE_ROOT}" "${HISTORY_ROOT:?Set HISTORY_ROOT}"
: "${STAGE1_RESULTS:?Set STAGE1_RESULTS}" "${LEGACY_RESULTS:?Set LEGACY_RESULTS}" "${ARCHIVE_ROOT:?Set ARCHIVE_ROOT}"
: "${EXTERNAL_RESULTS_ROOT:?Set EXTERNAL_RESULTS_ROOT}" "${EXTERNAL_PREFIX:?Set EXTERNAL_PREFIX}"
export MRI_GPUS=1
ledger="reports/mri/studies/$STUDY_ID/confirmation"
mkdir -p "$ledger" reports/slurm
exec 8>"$ledger/admin.lock"
flock -n 8
submit() {
    local name="$1" script="$2" deps="${3:-}" job state options=()
    if [[ -s "$ledger/$name.job" ]]; then
        read -r job < "$ledger/$name.job"
        state=$(sacct -nXj "$job" -o State -P | head -1)
        case "$state" in COMPLETED|RUNNING|PENDING|CONFIGURING|COMPLETING) echo "$job"; return;; esac
    fi
    if [[ -n "$deps" ]]; then options+=(--dependency="afterok:$deps" --kill-on-invalid-dep=yes); fi
    job=$(sbatch --parsable --export=ALL "${options[@]}" "$script")
    job=${job%%;*}
    [[ "$job" =~ ^[0-9]+$ ]]
    echo "$job" > "$ledger/$name.job"
    echo "$job"
}
freeze=$(submit freeze scripts/mri_freeze_study.sbatch "${BASELINE_VAL_JOBS:-}")
until [[ -s "$FREEZE_ROOT/matrix.json" ]]; do
    state=$(sacct -nXj "$freeze" -o State -P | head -1)
    case "$state" in RUNNING|PENDING|CONFIGURING|COMPLETING|"") sleep 30;; *)
        [[ -s "$FREEZE_ROOT/matrix.json" ]] || { echo "Study freeze $freeze ended with $state" >&2; exit 1; };;
    esac
done
jobs=()
while IFS=$'\t' read -r name frozen output; do
    job=$(FREEZE_DIR="$frozen" OUTPUT_ROOT="$output" RUN_ID="${STUDY_ID}_${name}_test" submit "$name" scripts/mri_test.sbatch "$freeze")
    jobs+=("$job")
done < "$FREEZE_ROOT/tests.tsv"
if [[ -n "${COHORT_DIR:-}" && -s "$COHORT_DIR/cohort.json" ]]; then
    while IFS=$'\t' read -r name frozen; do
        job=$(FREEZE_DIR="$frozen" OUTPUT_ROOT="$FREEZE_ROOT/../archive84/$name" PHASE=confirm \
            RUN_ID="${STUDY_ID}_${name}_archive" submit "${name}_archive" scripts/mri_generalization.sbatch "$freeze")
        jobs+=("$job")
        echo "archive $name $job"
    done < "$FREEZE_ROOT/archive.tsv"
fi
dependency=$(IFS=:; echo "${jobs[*]}")
report=$(submit report scripts/mri_study_report.sbatch "$dependency")
echo "freeze=$freeze test_jobs=$dependency report=$report"
