#!/bin/bash
#SBATCH --job-name=gsbm-mri-test15
#SBATCH --qos=high
#SBATCH --output=reports/slurm/mri-test-%j.out
#SBATCH --error=reports/slurm/mri-test-%j.err
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
: "${CHECKPOINT:?Set CHECKPOINT to the trained bridge/best.pt}"
cd "${GSBM_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/mri_common.sh
gsbm_environment
mapfile -t roots < <("$GSBM_PYTHON" - "$CHECKPOINT" <<'PY'
import sys
import math
from vemgsbm.runtime import load_checkpoint
saved = load_checkpoint(sys.argv[1])
data = saved['config']['data']
if saved['stage'] != 'reconstruction' or data['kind'] != 'paired_mri_png' or data['scale'] != 4:
    raise ValueError('Expected the paired-MRI X4 bridge checkpoint')
print(data['test']['root_lq'])
print(data['test']['root_hq'])
print(data['scan_cache']['test'])
test = saved['config'].get('test', {})
fraction = float(test.get('fraction', .15))
if not math.isfinite(fraction) or not 0 < fraction <= 1:
    raise ValueError('Invalid test fraction')
print(fraction)
print(int(test.get('subset_seed', 20261110)))
print(f'{100 * fraction:g}')
PY
)
[[ "${#roots[@]}" == 6 ]] || { echo 'Could not read test settings' >&2; exit 2; }
options=(--inference-variant "${INFERENCE_VARIANT:-legacy}")
if [[ -n "${SUBSET_MANIFEST:-}" ]]; then
    options+=(--subset-manifest "$SUBSET_MANIFEST")
fi
if [[ -n "${MRI_CASE_INVENTORY:-}" ]]; then
    options+=(--case-inventory "$MRI_CASE_INVENTORY")
fi
# The fraction is fixed by the user, not a max-cases truncation or per-rank fraction.
gsbm_torchrun -m vemgsbm.mri_infer --checkpoint "$CHECKPOINT" \
    --root-lq "${roots[0]}" --root-hq "${roots[1]}" --scan-cache "${roots[2]}" \
    --output-root "${OUTPUT_ROOT:-outputs/mri/${RUN_ID}_test${roots[5]}}" --device cuda \
    --test-fraction "${roots[3]}" --subset-seed "${roots[4]}" --save-volume-limit "${EXPORT_CASE_LIMIT:-8}" "${options[@]}"
