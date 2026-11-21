#!/bin/bash
# Administrative scheduler. A successful immutable study-wide freeze releases all TEST jobs.
set -euo pipefail
source scripts/mri_slurm_dependencies.sh
[[ $# == 0 || ( $# == 1 && "$1" == --wait ) ]] || { echo "Usage: $0 [--wait]" >&2; exit 2; }
wait_mode=${1:-}
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
names=()
outputs=()
freezes=()
scripts=()
add_evaluation() {
    local key="$1" frozen_path="$2" result="$3" script="$4" job
    job=$(FREEZE_DIR="$frozen_path" OUTPUT_ROOT="$result" RUN_ID="${STUDY_ID}_$key" PHASE=confirm \
        submit "$key" "$script" "$freeze")
    jobs+=("$job")
    names+=("$key")
    outputs+=("$result")
    freezes+=("$frozen_path")
    scripts+=("$script")
}
while IFS=$'\t' read -r name frozen output; do
    add_evaluation "$name" "$frozen" "$output" scripts/mri_test.sbatch
done < "$FREEZE_ROOT/tests.tsv"
if [[ -n "${COHORT_DIR:-}" ]]; then
    [[ -s "$COHORT_DIR/cohort.json" ]] || { echo "Missing requested independent cohort: $COHORT_DIR" >&2; exit 1; }
    while IFS=$'\t' read -r name frozen; do
        add_evaluation "${name}_archive" "$frozen" "$FREEZE_ROOT/../archive84/$name" scripts/mri_generalization.sbatch
    done < "$FREEZE_ROOT/archive.tsv"
fi
[[ ${#jobs[@]} -gt 0 ]]
wait_final() {
    local job="$1" state
    while true; do
        state=$(mri_slurm_state "$job") || return 1
        if ! mri_slurm_live "$state"; then echo "$state"; return; fi
        sleep "${MRI_STATUS_INTERVAL:-60}"
    done
}
if [[ "$wait_mode" == --wait ]]; then
    # All evaluation jobs run concurrently. A retry can reuse sealed case directories.
    for index in "${!jobs[@]}"; do
        name=${names[index]}
        output=${outputs[index]}
        retries="$ledger/$name.retries"
        while true; do
            job=${jobs[index]}
            state=$(wait_final "$job")
            if [[ "$state" == COMPLETED && -s "$output/summary.json" ]]; then break; fi
            case "$state" in
                TIMEOUT|PREEMPTED|NODE_FAIL)
                    count=0
                    if [[ -s "$retries" ]]; then count=$(wc -l < "$retries"); fi
                    if ((count >= ${MAX_CONTINUATIONS:-12})); then
                        echo "$name exhausted continuations ($job, $state)" >&2; exit 1
                    fi
                    printf '%s\t%s\n' "$job" "$state" >> "$retries"
                    jobs[index]=$(FREEZE_DIR="${freezes[index]}" OUTPUT_ROOT="$output" PHASE=confirm \
                        RUN_ID="${STUDY_ID}_$name" submit "$name" "${scripts[index]}" "$freeze")
                    ;;
                *) echo "$name has no completed result ($job, $state, $output)" >&2; exit 1;;
            esac
        done
    done
fi
dependency=$(IFS=:; echo "${jobs[*]}")
report=$(submit report scripts/mri_study_report.sbatch "$dependency")
echo "freeze=$freeze test_jobs=$dependency report=$report"
if [[ "$wait_mode" == --wait ]]; then
    [[ "$(wait_final "$report")" == COMPLETED ]]
    for file in report.md methods.csv paired.csv cases.csv runs.csv; do
        [[ -s "$FREEZE_ROOT/../report/$file" ]] || { echo "Missing final report file $file" >&2; exit 1; }
    done
    echo "Completed study $STUDY_ID. Report: $FREEZE_ROOT/../report/report.md"
fi
