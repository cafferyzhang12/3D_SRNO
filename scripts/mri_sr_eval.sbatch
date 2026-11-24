#!/bin/bash
#SBATCH --job-name=mri-sr-mls-val-test
#SBATCH --qos=high
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=1-00:00:00
#SBATCH --output=reports/slurm/%x-%j.out
#SBATCH --error=reports/slurm/%x-%j.err
set -euo pipefail
cd "${GSBM_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/mri_common.sh
gsbm_environment
: "${MRI_CASE_INVENTORY:?}" "${MRI_TEST_MANIFEST:?Use the fixed historical 84-case manifest}"
export CHECKPOINT="$PWD/runs/mri/$RUN_ID/bridge/best.pt"
export OUTPUT_ROOT="$PWD/outputs/mri/${RUN_ID}_val" MRI_SAMPLER_TYPE=ode MRI_NUM_SAMPLES=1 MRI_HAAR_READOUT=endpoint
unset MRI_INFER_STEPS
source scripts/mri_whole_val.sh
"$GSBM_PYTHON" -m vemgsbm.mri_x1_pipeline report-sr --config "runs/mri/$RUN_ID/experiment.yaml" --validation "$OUTPUT_ROOT"
export FREEZE_DIR="$PWD/runs/mri/$RUN_ID/frozen_test"
# Selection is complete on VAL; TEST is a single pre-authorized frozen replay.
"$GSBM_PYTHON" - <<'PY'
import json, os
from pathlib import Path
from vemgsbm.mri_freeze import freeze
from vemgsbm.runtime import fingerprint
folder = Path(os.environ['FREEZE_DIR'])
checkpoint, inventory, manifest = [os.environ[k] for k in ('CHECKPOINT','MRI_CASE_INVENTORY','MRI_TEST_MANIFEST')]
if folder.exists():
    record = json.loads((folder/'freeze.json').read_text())
    expected = (fingerprint(checkpoint),fingerprint(inventory),fingerprint(manifest))
    actual = (record['selected']['checkpoint_sha256'],record['inventory_sha256'],record['manifest_sha256'])
    if actual != expected:
        raise ValueError('Existing TEST freeze belongs to different artifacts')
else:
    freeze([(checkpoint,os.environ['OUTPUT_ROOT'])],inventory,manifest,folder)
PY
gsbm_torchrun -m vemgsbm.mri_freeze confirm --folder "$FREEZE_DIR" --output "outputs/mri/${RUN_ID}_test84" --device cuda
