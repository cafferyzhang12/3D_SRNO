#!/bin/bash
#SBATCH --job-name=mri-x1-val
#SBATCH --qos=high
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --output=reports/slurm/%x-%j.out
#SBATCH --error=reports/slurm/%x-%j.err
set -euo pipefail
source scripts/mri_common.sh
gsbm_environment
export CHECKPOINT="$PWD/runs/mri/$RUN_ID/bridge/best.pt"
export OUTPUT_ROOT="$PWD/outputs/mri/${RUN_ID}_val" MRI_SAMPLER_TYPE=ode MRI_NUM_SAMPLES=1
source scripts/mri_whole_val.sh
"$GSBM_PYTHON" -m vemgsbm.mri_x1_pipeline report --config "runs/mri/$RUN_ID/experiment.yaml" --validation "$OUTPUT_ROOT"
