#!/bin/bash
# Submit from a versioned deployment directory. Reuse RUN_ID to recover failed arms.
set -euo pipefail
: "${RUN_ID:?Set a unique study RUN_ID}" "${HISTORY_ROOT:?Set HISTORY_ROOT}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,75}$ ]]
export MRI_GPUS=1
ledger="reports/mri/submissions/$RUN_ID"
mkdir -p "$ledger" reports/slurm
exec 9>"$ledger/submit.lock"
flock -n 9
submit() {
    local arm="$1" dependency="${2:-}" job state options=()
    if [[ -s "$ledger/$arm.job" ]]; then
        read -r job < "$ledger/$arm.job"
        state=$(sacct -nXj "$job" -o State -P | head -1)
        case "$state" in
            COMPLETED|RUNNING|PENDING|CONFIGURING|COMPLETING) echo "$job"; return;;
        esac
    fi
    if [[ -n "$dependency" ]]; then options+=(--dependency="afterok:$dependency"); fi
    job=$(ARM="$arm" sbatch --parsable --export=ALL --kill-on-invalid-dep=yes \
        --job-name="x4-$arm" "${options[@]}" scripts/mri_stage1.sbatch)
    job=${job%%;*}
    [[ "$job" =~ ^[0-9]+$ ]]
    echo "$job" > "$ledger/$arm.job"
    echo "$job"
}
smoke=$(submit smoke)
dependency="$smoke${HISTORY_GATE_JOB:+:$HISTORY_GATE_JOB}"
plain=$(submit plain "$dependency")
guided=$(submit guided "$dependency")
for arm in corrected equal_calls reference bilinear; do submit "$arm" "$dependency"; done
submit diagnostics "$plain:$guided"
for record in "$ledger"/*.job; do echo "$(basename "$record" .job) $(cat "$record")"; done
