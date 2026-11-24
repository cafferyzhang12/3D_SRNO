#!/bin/bash
set -euo pipefail
: "${CHECKPOINT:?Set CHECKPOINT}"
mapfile -t val < <("$GSBM_PYTHON" - "$CHECKPOINT" <<'PY'
import sys
from vemgsbm.runtime import load_checkpoint, project_path
cfg = load_checkpoint(sys.argv[1])['config']
for key in ('root_lq', 'root_hq'):
    print(project_path(cfg['data']['val'][key]))
PY
)
[[ "${#val[@]}" == 2 ]]
args=()
if [[ -n "${MRI_SAMPLER_TYPE:-}" ]]; then args+=(--sampler-type "$MRI_SAMPLER_TYPE"); fi
if [[ -n "${MRI_INFER_STEPS:-}" ]]; then args+=(--steps "$MRI_INFER_STEPS"); fi
if [[ -n "${MRI_NUM_SAMPLES:-}" ]]; then args+=(--samples "$MRI_NUM_SAMPLES"); fi
if [[ -n "${MRI_HAAR_READOUT:-}" ]]; then args+=(--readout "$MRI_HAAR_READOUT"); fi
if [[ -n "${MRI_CASE_INVENTORY:-}" ]]; then args+=(--case-inventory "$MRI_CASE_INVENTORY"); fi
gsbm_torchrun -m vemgsbm.mri_infer --split val --test-fraction 1 --checkpoint "$CHECKPOINT" \
    --root-lq "${val[0]}" --root-hq "${val[1]}" --output-root "${OUTPUT_ROOT:-outputs/mri/${RUN_ID}_val}" \
    --device cuda --inference-variant endpoint_safe --save-volume-limit 0 --resume-cases "${args[@]}"
