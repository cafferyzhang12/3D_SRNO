#!/bin/bash
#SBATCH --job-name=mri-x4-gate
#SBATCH --qos=high
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=reports/slurm/%x-%j.out
#SBATCH --error=reports/slurm/%x-%j.err
set -euo pipefail
source scripts/mri_common.sh
gsbm_environment
: "${HISTORY_ROOT:?Set HISTORY_ROOT to the snapshot from job 5281446}"
run="$HISTORY_ROOT/runs/mri/mri_x4_gsbm_mls_v2_20261120"
test_result="$HISTORY_ROOT/outputs/mri/mri_x4_gsbm_mls_v2_20261120_test15"
"$GSBM_PYTHON" -m pytest -q
gsbm_torchrun tools/mri_history_gate.py --checkpoint "$run/bridge/best.pt" \
  --inventory "$run/mri_case_inventory.json.gz" --historical-summary "$test_result/summary.json" \
  --manifest "$test_result/subset_manifest.json" --output "runs/mri/$RUN_ID/history_gate.json"
