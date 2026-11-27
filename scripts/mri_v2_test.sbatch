#!/bin/bash
#SBATCH --qos=high
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=2-00:00:00
#SBATCH --output=reports/slurm/%x-%j.out
#SBATCH --error=reports/slurm/%x-%j.err
#SBATCH --job-name=mri-v2-test
set -euo pipefail
cd "${GSBM_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
source scripts/mri_common.sh
gsbm_environment
: "${MRI_CASE_INVENTORY:?}" "${MRI_TEST_MANIFEST:?}" "${RUN_ID:?}" "${V2_ARMS:?}"
root="runs/mri/v2/$RUN_ID"
read -r -a arms <<< "$V2_ARMS"
args=()
for arm in "${arms[@]}"; do
    [[ "$arm" =~ ^[1-4]$ ]]
    args+=(--arm "$root/exp$arm")
done
gsbm_torchrun tools/mri_v2_test.py "${args[@]}" --inventory "$MRI_CASE_INVENTORY" \
    --manifest "$MRI_TEST_MANIFEST" --output "$root/test15" --device cuda
