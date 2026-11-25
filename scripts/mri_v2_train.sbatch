#!/bin/bash
#SBATCH --qos=high
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=7-00:00:00
#SBATCH --output=reports/slurm/%x-%j.out
#SBATCH --error=reports/slurm/%x-%j.err
#SBATCH --job-name=mri-v2-train
set -euo pipefail
cd "${GSBM_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
source scripts/mri_common.sh
gsbm_environment
: "${RUN_ID:?}" "${V2_ARM:?}"
root="runs/mri/v2/$RUN_ID"
[[ -f "$root/gate_passed" && "$V2_ARM" =~ ^[1-4]$ ]]
config="$root/exp${V2_ARM}.yaml"
resume=()
[[ ! -f "$root/exp${V2_ARM}/last.pt" ]] || resume=(--resume "$root/exp${V2_ARM}/last.pt")
gsbm_torchrun -m vemgsbm.mri_v2_train --config "$config" \
    --case-inventory "$root/trainval_inventory.json.gz" --device cuda "${resume[@]}"
