#!/bin/bash
#SBATCH --qos=high
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=1-00:00:00
#SBATCH --output=reports/slurm/%x-%j.out
#SBATCH --error=reports/slurm/%x-%j.err
#SBATCH --job-name=mri-v2-val
set -euo pipefail
cd "${GSBM_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
source scripts/mri_common.sh
gsbm_environment
: "${RUN_ID:?}" "${V2_ARM:?}"
root="runs/mri/v2/$RUN_ID"
checkpoint="$root/exp${V2_ARM}/best.pt"
[[ -f "$checkpoint" ]] || checkpoint="$root/exp${V2_ARM}/last.pt"
gsbm_torchrun -m vemgsbm.mri_v2_train --config "$root/exp${V2_ARM}.yaml" \
    --case-inventory "$root/trainval_inventory.json.gz" --device cuda --resume "$checkpoint" --evaluate
