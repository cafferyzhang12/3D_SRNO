#!/bin/bash
#SBATCH --job-name=mri-x1-train
#SBATCH --qos=high
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=2-00:00:00
#SBATCH --output=reports/slurm/%x-%j.out
#SBATCH --error=reports/slurm/%x-%j.err
set -euo pipefail
source scripts/mri_common.sh
gsbm_environment
[[ -s "runs/mri/${RUN_ID}_smoke/bridge/smoke_gate.json" ]]
config="runs/mri/$RUN_ID/experiment.yaml"
resume=()
if [[ -f "runs/mri/$RUN_ID/bridge/last.pt" ]]; then resume+=(--resume "runs/mri/$RUN_ID/bridge/last.pt"); fi
gsbm_torchrun -m vemgsbm.train --config "$config" --case-inventory "$MRI_CASE_INVENTORY" --device cuda "${resume[@]}"
