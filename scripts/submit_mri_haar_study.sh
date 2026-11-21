#!/bin/bash
# Administrative orchestration only. All checkpoint/data reads and selection run under Slurm.
set -euo pipefail
: "${STUDY_ID:?Set a new study identifier}"
[[ "$STUDY_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$ ]]
[[ $# == 1 ]]
cd "${GSBM_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/mri_slurm_dependencies.sh
folder="reports/mri/studies/$STUDY_ID"
mkdir -p "$folder" reports/slurm
exec 8>"$folder/admin.lock"
flock -n 8
stages=(backbone dimensions controls reference replicate)
if [[ "$1" != all ]]; then
    case "$1" in backbone|dimensions|controls|reference|replicate) stages=("$1");; *) exit 2;; esac
fi
select_job() {
    local action="$1" stage="$2" dependency="$3" file="$folder/${1}_${2}.job" opts=() job state
    if [[ -s "$file" ]]; then
        read -r job < "$file"
        state=$(mri_slurm_state "$job") || return 1
        if [[ "$state" == COMPLETED ]] || mri_slurm_live "$state"; then echo "$job"; return; fi
    fi
    if [[ "$action" == bank ]]; then opts+=(--time=2-00:00:00); fi
    job=$(MRI_GPUS=1 STUDY_ACTION="$action" STUDY_STAGE="$stage" mri_slurm_submit "$dependency" "${opts[@]}" scripts/mri_haar_select.sbatch) || return 1
    echo "$job" > "$file"
    echo "$job"
}
wait_for_job() {
    local job="$1" state
    while true; do
        state=$(mri_slurm_state "$job") || return 1
        if [[ "$state" == COMPLETED ]]; then return 0; fi
        if ! mri_slurm_live "$state"; then echo "Slurm job $job ended with $state" >&2; return 1; fi
        sleep 30
    done
}
submit_stage() {
    local stage="$1" descriptor="" bank="" jobs=() depends job select_id
    if [[ "$stage" != backbone ]]; then
        read -r descriptor < "$folder/descriptor.txt"
        bank="$PWD/$folder/spatial_bank.pt"
    fi
    while IFS=$'\t' read -r rid variant dim query path reference shuffle seed readout; do
        response=$(RUN_ID="$rid" VARIANT="$variant" DIM="$dim" QUERY="$query" PATH_GEOMETRY="$path" \
            REFERENCE="$reference" SHUFFLE_ROUTING="$shuffle" SEED="$seed" DIRECT_CHECKPOINT="$descriptor" SPATIAL_BANK="$bank" \
            bash scripts/submit_mri_haar.sh) || return 1
        job="${response##*experiment_job=}"
        [[ "$job" =~ ^[0-9]+$ ]] || return 1
        if [[ " ${jobs[*]} " != *" $job "* ]]; then jobs+=("$job"); fi
    done < "$folder/$stage.tsv"
    [[ "${#jobs[@]}" -gt 0 ]] || return 1
    depends=$(IFS=:; echo "${jobs[*]}")
    select_id=$(select_job select "$stage" "$depends") || return 1
    printf 'stage=%s train_jobs=%s select_job=%s\n' "$stage" "$depends" "$select_id" >&2
    echo "$select_id"
}

recoverable_interruption() {
    local stage="$1" count=0 ledger rid rest job state checkpoint stamp seen attempts last previous_job
    local limit="${MAX_CONTINUATIONS:-12}"
    [[ "$limit" =~ ^[0-9]+$ ]]
    while IFS=$'\t' read -r rid rest; do
        ledger="reports/mri/submissions/$rid"
        read -r job < "$ledger/train.job"
        state=$(mri_slurm_state "$job") || return 1
        if mri_slurm_live "$state"; then continue; fi
        case "$state" in
            COMPLETED) continue;;
            TIMEOUT|PREEMPTED|NODE_FAIL) ;;
            *) echo "Training job $job requires investigation ($state)" >&2; return 1;;
        esac
        checkpoint="runs/mri/$rid/bridge/last.pt"
        [[ -s "$checkpoint" ]] || { echo "No completed epoch for $rid" >&2; return 1; }
        # Stat metadata only. Training publishes last.pt by atomic replacement.
        stamp=$(stat -c '%s:%Y' "$checkpoint")
        seen="$ledger/continuations.tsv"
        if [[ -s "$seen" ]]; then
            IFS=$'\t' read -r previous_job last < <(tail -1 "$seen")
            if [[ "$job" == "$previous_job" ]]; then continue; fi
            if [[ "$last" == "$stamp" ]]; then echo "Checkpoint stopped advancing: $rid" >&2; return 1; fi
        fi
        attempts=0
        if [[ -f "$seen" ]]; then attempts=$(wc -l < "$seen"); fi
        (( attempts < limit )) || { echo "Continuation limit reached: $rid" >&2; return 1; }
        printf '%s\t%s\n' "$job" "$stamp" >> "$seen"
        count=$((count+1))
    done < "$folder/$stage.tsv"
    ((count > 0))
}

if [[ ! -s "$folder/backbone.tsv" ]]; then
    wait_for_job "$(select_job init backbone "${DEPENDENCY_JOB_ID:-}")"
fi
for stage in "${stages[@]}"; do
    [[ -s "$folder/$stage.tsv" ]] || { echo "Complete the preceding VAL selection first" >&2; exit 2; }
    if [[ "$stage" != backbone && ! -s "$folder/spatial_bank.pt" ]]; then
        wait_for_job "$(select_job bank dimensions '')"
        [[ -s "$folder/spatial_bank.pt" ]]
    fi
    while true; do
        select_id=$(submit_stage "$stage")
        if [[ "$1" != all ]]; then break; fi
        if wait_for_job "$select_id"; then
            [[ -s "$folder/$stage.json" ]]
            break
        fi
        recoverable_interruption "$stage" || exit 1
    done
done
