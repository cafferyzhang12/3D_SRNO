#!/bin/bash
# All helpers query controller/accounting metadata only. stdout is machine-readable.

mri_slurm_live() {
    case "$1" in
        PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|REQUEUE_FED|REQUEUE_HOLD|RESIZING|SUSPENDED|STOPPED|SIGNALING|STAGE_OUT) return 0 ;;
        *) return 1 ;;
    esac
}

mri_slurm_state() {
    local job="$1" attempts="${SLURM_LOOKUP_ATTEMPTS:-12}" delay="${SLURM_LOOKUP_DELAY:-5}"
    local attempt id state code rows
    if [[ ! "$job" =~ ^[0-9]+$ || ! "$attempts" =~ ^[1-9][0-9]*$ || ! "$delay" =~ ^[0-9]+$ ]]; then
        echo "Invalid job ID or Slurm lookup settings" >&2
        return 1
    fi
    for ((attempt=1; attempt<=attempts; attempt++)); do
        # Only a job still visible to the controller is safe to submit as a live dependency.
        rows=$(squeue -h -j "$job" -o '%i|%T' 2>/dev/null) || rows=""
        while IFS='|' read -r id state; do
            if [[ "$id" == "$job" ]] && mri_slurm_live "$state"; then
                echo "$state"
                return 0
            fi
        done <<< "$rows"
        # Match the exact allocation, ignoring .batch, .extern and unrelated rows.
        rows=$(sacct -nX -j "$job" -o JobIDRaw,State,ExitCode -P 2>/dev/null) || rows=""
        while IFS='|' read -r id state code; do
            [[ "$id" == "$job" ]] || continue
            state=${state%% *}
            state=${state%+}
            case "$state" in
                COMPLETED)
                    case "$code" in
                        0:0) echo COMPLETED; return 0 ;;
                        "") ;;  # The final exit record can arrive later.
                        *) echo "FAILED_EXIT_$code"; return 0 ;;
                    esac ;;
                BOOT_FAIL|CANCELLED|DEADLINE|FAILED|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|REVOKED|TIMEOUT)
                    echo "$state"
                    return 0 ;;
            esac
        done <<< "$rows"
        if ((attempt < attempts)); then sleep "$delay"; fi
    done
    echo "Unable to verify live or final Slurm status for job $job after $attempts lookups" >&2
    return 1
}

mri_slurm_dependencies() {
    local dependencies="${1:-}" job state result="" seen=":"
    local -a ids
    if [[ -z "$dependencies" ]]; then return 0; fi
    if [[ ! "$dependencies" =~ ^[0-9]+(:[0-9]+)*$ ]]; then
        echo "Invalid afterok job list: $dependencies" >&2
        return 1
    fi
    IFS=: read -r -a ids <<< "$dependencies"
    for job in "${ids[@]}"; do
        [[ "$seen" == *":$job:"* ]] && continue
        seen+="$job:"
        state=$(mri_slurm_state "$job") || return 1
        if [[ "$state" == COMPLETED ]]; then continue; fi
        if ! mri_slurm_live "$state"; then
            echo "Cannot submit afterok dependency: job $job ended with $state" >&2
            return 1
        fi
        result+="${result:+:}$job"
    done
    printf '%s\n' "$result"
}

mri_slurm_submit() {
    local dependencies="$1" live retry job errfile diagnostic line
    shift
    local -a options ids
    live=$(mri_slurm_dependencies "$dependencies") || return 1
    for retry in 0 1; do
        options=()
        if [[ -n "$live" ]]; then options+=(--dependency="afterok:$live" --kill-on-invalid-dep=yes); fi
        errfile=$(mktemp) || return 1
        if job=$(sbatch --parsable --export=ALL "${options[@]}" "$@" 2>"$errfile"); then
            cat "$errfile" >&2
            rm -f "$errfile"
            # Site wrappers also print informational messages on stdout. Only a
            # complete line containing JobID or JobID;cluster identifies a job.
            ids=()
            while IFS= read -r line; do
                if [[ "$line" =~ ^[[:space:]]*([0-9]+)(\;[^\;[:space:]]+)?[[:space:]]*$ ]]; then
                    ids+=("${BASH_REMATCH[1]}")
                elif [[ -n "$line" ]]; then
                    printf '%s\n' "$line" >&2
                fi
            done <<< "$job"
            if [[ "${#ids[@]}" != 1 ]]; then
                echo "Expected one standalone job ID in successful sbatch reply: $job. Check the queue before resubmitting." >&2
                return 1
            fi
            echo "${ids[0]}"
            return 0
        fi
        diagnostic=$(<"$errfile")
        rm -f "$errfile"
        # A remaining job can complete and be purged between squeue and sbatch.
        # Retry only after an explicit dependency rejection and a changed live set.
        if [[ $retry == 0 && -n "$live" ]]; then
            case "${diagnostic,,}" in
                *"job dependency problem"*|*"invalid job id"*|*"invalid dependency"*)
                    local refreshed
                    refreshed=$(mri_slurm_dependencies "$dependencies") || return 1
                    if [[ "$refreshed" != "$live" ]]; then live="$refreshed"; continue; fi ;;
            esac
        fi
        echo "$diagnostic" >&2
        return 1
    done
}
