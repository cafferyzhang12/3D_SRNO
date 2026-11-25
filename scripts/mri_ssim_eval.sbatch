#!/bin/bash
#SBATCH --qos=high
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --output=reports/slurm/%x-%j.out
#SBATCH --error=reports/slurm/%x-%j.err
#SBATCH --job-name=mri-ssim-eval
#SBATCH --time=1-00:00:00
set -euo pipefail
cd "${GSBM_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/mri_common.sh
gsbm_environment
export XDG_CACHE_HOME="$PWD/cache/xdg"
export TORCH_HOME="$PWD/cache/perceptual/torch"
: "${RUN_ID:?}" "${MRI_CASE_INVENTORY:?}" "${MRI_TEST_MANIFEST:?}" "${BASELINE_CHECKPOINT:?}" "${MRI_MLS_BANK:?}"
study="runs/mri/$RUN_ID"
export CHECKPOINT
CHECKPOINT=$(cat "$study/selected_checkpoint.txt")
export OUTPUT_ROOT="$PWD/outputs/mri/${RUN_ID}_val" MRI_SAMPLER_TYPE=ode MRI_NUM_SAMPLES=1 MRI_HAAR_READOUT=endpoint MRI_INFER_STEPS=4
source scripts/mri_whole_val.sh
"$GSBM_PYTHON" -m vemgsbm.mri_freeze freeze --candidate "$CHECKPOINT" "$OUTPUT_ROOT" --inventory "$MRI_CASE_INVENTORY" --manifest "$MRI_TEST_MANIFEST" --folder "$study/frozen_test"
for candidate in baseline selected; do
    ckpt="$BASELINE_CHECKPOINT"
    [[ "$candidate" == baseline ]] || ckpt="$study/frozen_test/model.pt"
    for split in val test; do
        extra=()
        [[ "$split" != test ]] || extra+=(--manifest "$study/frozen_test/manifest.json")
        gsbm_torchrun -m vemgsbm.mri_perceptual evaluate --checkpoint "$ckpt" --inventory "$study/frozen_test/inventory.json.gz" --split "$split" "${extra[@]}" --weights-dir "$PWD/cache/perceptual" --output "outputs/mri/${RUN_ID}_${candidate}_${split}_perceptual" --device cuda --batch-size 8
    done
done
