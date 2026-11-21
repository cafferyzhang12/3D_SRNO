#!/bin/bash
# Administrative scheduler. A successful immutable study-wide freeze releases all TEST jobs.
set -euo pipefail
source scripts/mri_slurm_dependencies.sh
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
        state=$(mri_slurm_state "$job") || return 1
        if [[ "$state" == COMPLETED ]] || mri_slurm_live "$state"; then echo "$job"; return; fi
    fi
    job=$(mri_slurm_submit "$deps" "$script") || return 1
    echo "$job" > "$ledger/$name.job"
    echo "$job"
}
freeze=$(submit freeze scripts/mri_freeze_study.sbatch "${BASELINE_VAL_JOBS:-}")
until [[ -s "$FREEZE_ROOT/matrix.json" ]]; do
    state=$(mri_slurm_state "$freeze")
    if mri_slurm_live "$state"; then sleep 30; else
        [[ "$state" == COMPLETED && -s "$FREEZE_ROOT/matrix.json" ]] || {
            echo "Study freeze $freeze ended with $state before matrix publication" >&2; exit 1; }
    fi
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
