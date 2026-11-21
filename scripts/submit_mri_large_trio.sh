#!/bin/bash
# Submit/resume only the three large backbone experiments and their frozen TEST.
set -euo pipefail
cd "${GSBM_PROJECT_ROOT:-$PWD}"
source scripts/mri_slurm_dependencies.sh
: "${STUDY_ID:?}" "${MRI_DATA_ROOT:?}" "${MRI_CASE_INVENTORY:?}" "${FIXED_TEST_MANIFEST:?}"
[[ "$STUDY_ID" =~ ^[a-zA-Z0-9_.-]{1,40}$ ]]
export MRI_GPUS=4 MRI_LARGE_PAIRED=1
export OVERFIT_STEPS="${OVERFIT_STEPS:-128}"
ledger="reports/mri/$STUDY_ID"
export MRI_BATCH_FILE="$PWD/$ledger/batch.txt"
mkdir -p "$ledger" reports/slurm
exec 8>"$ledger/trio.lock"
flock -n 8
submit_once() {
    local key=$1 deps=$2 id state
    shift 2
    if [[ -s "$ledger/$key.job" ]]; then
        read -r id < "$ledger/$key.job"
        state=$(mri_slurm_state "$id") || return 1
        if [[ "$state" == COMPLETED ]] || mri_slurm_live "$state"; then echo "$id"; return; fi
    fi
    id=$(mri_slurm_submit "$deps" "$@") || return 1
    echo "$id" > "$ledger/$key.job"
    echo "$id"
}
wait_success() {
    local id=$1 state
    while true; do
        state=$(mri_slurm_state "$id") || return 1
        if [[ "$state" == COMPLETED ]]; then return 0; fi
        mri_slurm_live "$state" || { echo "$id $state" >&2; return 1; }
        sleep 60
    done
}
export DEPENDENCY_JOB_ID
DEPENDENCY_JOB_ID=$(submit_once probe '' scripts/mri_large_probe.sbatch)
arms=(direct bridge drift_only)
train_job() {
    local arm=$1 response
    response=$(RUN_ID="${STUDY_ID}_$arm" VARIANT="$arm" bash scripts/submit_mri_haar.sh) || return 1
    echo "${response##*experiment_job=}"
}
for arm in "${arms[@]}"; do
    job=$(train_job "$arm"); echo "$job" > "$ledger/$arm.job"
    echo "$arm training $job"
done
watch_training() {
    local arm=$1 job checkpoint stamp file
    read -r job < "$ledger/$arm.job"
    while ! wait_success "$job"; do
        case "$(mri_slurm_state "$job")" in TIMEOUT|PREEMPTED|NODE_FAIL) ;; *) exit 1;; esac
        checkpoint="runs/mri/${STUDY_ID}_$arm/bridge/last.pt"
        stamp=$(stat -c '%s:%Y' "$checkpoint")
        file="$ledger/$arm.continuations"
        if [[ -s "$file" && "$(tail -1 "$file" | cut -f2)" == "$stamp" ]]; then
            echo "$arm checkpoint did not advance" >&2; exit 1
        fi
        printf '%s\t%s\n' "$job" "$stamp" >> "$file"
        job=$(train_job "$arm"); echo "$job" > "$ledger/$arm.job"
    done
}
watchers=()
for arm in "${arms[@]}"; do
    watch_training "$arm" &
    watchers+=("$!")
done
failed=0
for watcher in "${watchers[@]}"; do
    wait "$watcher" || failed=1
done
[[ "$failed" == 0 ]]
# Complete all validation-based selections before the first TEST job is released.
freezes=()
for arm in "${arms[@]}"; do
    rid="${STUDY_ID}_$arm"
    ckpt="$PWD/runs/mri/$rid/bridge/best.pt"
    options=(--candidate "$ckpt" "$PWD/outputs/mri/${rid}_val")
    if [[ "$arm" == bridge ]]; then options+=(--candidate "$ckpt" "$PWD/outputs/mri/${rid}_integral_val"); fi
    freeze=$(MRI_GPUS=1 submit_once "freeze_$arm" '' scripts/mri_freeze.sbatch "${options[@]}" \
        --inventory "$MRI_CASE_INVENTORY" --manifest "$FIXED_TEST_MANIFEST" --folder "$PWD/$ledger/frozen/$arm")
    freezes+=("$freeze")
done
deps=$(IFS=:; echo "${freezes[*]}")
test_job() {
    local arm=$1
    RUN_ID="${STUDY_ID}_${arm}_test" FREEZE_DIR="$PWD/$ledger/frozen/$arm" OUTPUT_ROOT="$PWD/outputs/mri/${STUDY_ID}_${arm}_test" \
        submit_once "test_$arm" "$deps" --gres=gpu:4 --cpus-per-task=24 scripts/mri_test.sbatch
}
for arm in "${arms[@]}"; do job=$(test_job "$arm"); echo "$arm TEST $job"; done
for arm in "${arms[@]}"; do
    read -r job < "$ledger/test_$arm.job"
    until wait_success "$job"; do
        case "$(mri_slurm_state "$job")" in TIMEOUT|PREEMPTED|NODE_FAIL) ;; *) exit 1;; esac
        job=$(test_job "$arm")
    done
    [[ -s "outputs/mri/${STUDY_ID}_${arm}_test/summary.json" ]]
done
echo "All three frozen tests completed."
