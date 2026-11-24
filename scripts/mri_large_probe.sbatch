#!/bin/bash
#SBATCH --job-name=mri-large-probe
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --qos=high
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
folder=$(dirname "${MRI_BATCH_FILE:?}")
mkdir -p "$folder"
"$GSBM_PYTHON" -m vemgsbm.mri_haar_experiment config --variant bridge --large-paired \
    --template configs/mri_haar_bridge_legacy_x4.yaml --output "$folder/probe.yaml"
gsbm_torchrun -m vemgsbm.mri_batch_probe "$folder/probe.yaml" "$folder"
