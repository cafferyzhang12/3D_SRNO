#!/bin/bash
# Re-evaluate archived checkpoints on full VAL. The guided checkpoint is evaluated in stage 1.
set -euo pipefail
: "${ARCHIVE_ROOT:?Set ARCHIVE_ROOT to the parent of the historical snapshots}"
: "${HISTORY_ROOT:?Set the 5281446 snapshot}" "${RUN_ID:?Set the study ID}"
export MRI_CASE_INVENTORY="$HISTORY_ROOT/runs/mri/mri_x4_gsbm_mls_v2_20261120/mri_case_inventory.json.gz"
[[ -f "$MRI_CASE_INVENTORY" ]]
arms=(
    "gsbm|vem_gsbm/runs/mri/mri_x4_gsbm_20261111_001/bridge/best.pt"
    "momentum|vem_gsbm/runs/mri/mri_x4_parallel_20261111_001_momentum/momentum/best.pt"
    "unguided_mls|vem_gsbm_mls_20261113_inventory_fix/runs/mri/mri_x4_gsbm_mls_inventory_fix_20261113/bridge/best.pt"
    "momentum_mls|vem_gsbm_mls_v2_20261120_f08f30129d6a/runs/mri/mri_x4_momentum_mls_v2_20261120/momentum/best.pt"
    "tdm|vem_gsbm_momentum_tdm_20261116/runs/mri/mri_x4_momentum_tdm_20261116/momentum/best.pt"
)
ledger="reports/mri/submissions/${RUN_ID}_legacy"
mkdir -p "$ledger" reports/slurm
exec 9>"$ledger/submit.lock"
flock -n 9
for arm in "${arms[@]}"; do
    name=${arm%%|*}
    checkpoint="$ARCHIVE_ROOT/${arm#*|}"
    [[ -s "$checkpoint" ]]
    if [[ -s "$ledger/$name.job" ]]; then
        state=$(sacct -nXj "$(cat "$ledger/$name.job")" -o State -P | head -1)
        case "$state" in COMPLETED|RUNNING|PENDING|CONFIGURING|COMPLETING) continue;; esac
    fi
    RUN_ID="${RUN_ID}_$name" CHECKPOINT="$checkpoint" MRI_GPUS=1 bash scripts/submit_mri.sh val > "$ledger/$name.job"
done
for file in "$ledger"/*.job; do echo "$(basename "$file" .job) $(cat "$file")"; done
