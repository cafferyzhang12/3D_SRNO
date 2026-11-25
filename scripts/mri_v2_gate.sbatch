#!/bin/bash
#SBATCH --qos=high
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00
#SBATCH --output=reports/slurm/%x-%j.out
#SBATCH --error=reports/slurm/%x-%j.err
#SBATCH --job-name=mri-v2-gate
set -euo pipefail
cd "${GSBM_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
source scripts/mri_common.sh
gsbm_environment
: "${MRI_CASE_INVENTORY:?}" "${RUN_ID:?}"
export TORCH_HOME="${MRI_PERCEPTUAL_WEIGHTS:-$PWD/cache/perceptual/torch}"
"$GSBM_PYTHON" -m pytest -q --junitxml="reports/mri/${RUN_ID}_pytest.xml"
"$GSBM_PYTHON" tools/mri_v2_prepare.py --inventory "$MRI_CASE_INVENTORY" --run-id "$RUN_ID"
root="runs/mri/v2/$RUN_ID"
"$GSBM_PYTHON" -m vemgsbm.mri_perceptual prepare-weights --weights-dir "$TORCH_HOME" --device cuda
"$GSBM_PYTHON" tools/mri_v2_smoke.py --device cuda --weights-dir "$TORCH_HOME" --output "$root/smoke.json"
"$GSBM_PYTHON" tools/mri_v2_memory_probe.py --configs-dir "$root" --output "$root/memory.json"
legacy=()
[[ -z "${MRI_LEGACY_CHECKPOINT:-}" ]] || legacy=(--checkpoint "$MRI_LEGACY_CHECKPOINT" --legacy-inventory "$MRI_CASE_INVENTORY")
"$GSBM_PYTHON" tools/mri_ceiling_audit.py --config "$root/exp1.yaml" --inventory "$root/trainval_inventory.json.gz" \
    --weights-dir "$TORCH_HOME" --require-perceptual --device cuda --output "$root/ceiling_val.json" "${legacy[@]}"
touch "$root/gate_passed"
