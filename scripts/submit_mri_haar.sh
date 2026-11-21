#!/bin/bash
# One independent experiment; repeat with the same RUN_ID to resume from last.pt.
set -euo pipefail
: "${RUN_ID:?Set an independent RUN_ID}" "${VARIANT:?direct|bridge|drift_only|mean|pca|mls}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,75}$ ]]
case "$VARIANT" in
    direct|bridge|drift_only) ;;
    mean|pca|mls) : "${DIRECT_CHECKPOINT:?Set the frozen direct model path}";;
    *) exit 2;;
esac
training_gpus="${MRI_GPUS:-1}"
[[ "$training_gpus" == 1 || "$training_gpus" == 4 ]]
mkdir -p reports/slurm "reports/mri/submissions/$RUN_ID"
ledger="reports/mri/submissions/$RUN_ID"
exec 9>"$ledger/submit.lock"
flock -n 9
submit() {
    local phase="$1" gpus="$2" dependency="$3" state job
    local options=(--gres="gpu:$gpus" --cpus-per-task="$((6 * gpus))" --job-name="haar-$phase-$VARIANT")
    if [[ -s "$ledger/$phase.job" ]]; then
        read -r job < "$ledger/$phase.job"
        state=$(sacct -nXj "$job" -o State -P | head -1)
        case "$state" in
            COMPLETED|RUNNING|PENDING|CONFIGURING|COMPLETING) echo "$job"; return;;
        esac
    fi
    if [[ -n "$dependency" ]]; then
        [[ "$dependency" =~ ^[0-9]+(:[0-9]+)*$ ]]
        options+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
    fi
    job=$(MRI_GPUS="$gpus" PHASE="$phase" sbatch --parsable --export=ALL "${options[@]}" scripts/mri_haar.sbatch)
    job=${job%%;*}
    [[ "$job" =~ ^[0-9]+$ ]]
    printf '%s\n' "$job" > "$ledger/$phase.job"
    printf '%s\n' "$job"
}
gate=$(submit gate 4 "${DEPENDENCY_JOB_ID:-}")
experiment=$(submit train "$training_gpus" "$gate")
printf 'ddp_gate_job=%s experiment_job=%s\n' "$gate" "$experiment"
