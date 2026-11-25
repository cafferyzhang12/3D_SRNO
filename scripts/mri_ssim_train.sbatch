#!/bin/bash
#SBATCH --qos=high
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --output=reports/slurm/%x-%j.out
#SBATCH --error=reports/slurm/%x-%j.err
#SBATCH --job-name=mri-ssim-train
#SBATCH --time=2-00:00:00
set -euo pipefail
cd "${GSBM_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/mri_common.sh
gsbm_environment
export XDG_CACHE_HOME="$PWD/cache/xdg"
export TORCH_HOME="$PWD/cache/perceptual/torch"
: "${RUN_ID:?}" "${MRI_CASE_INVENTORY:?}" "${MRI_TEST_MANIFEST:?}" "${BASELINE_CHECKPOINT:?}" "${MRI_MLS_BANK:?}"
study="runs/mri/$RUN_ID"
[[ -s "$study/smoke/bridge/structural_smoke_gate.json" ]]
for variant in control ssim stratified neighbor; do
    gsbm_torchrun -m vemgsbm.train --config "$study/$variant.yaml" --case-inventory "$MRI_CASE_INVENTORY" --device cuda
done
"$GSBM_PYTHON" -m vemgsbm.mri_ssim_pipeline select --study "$study/study.json"
