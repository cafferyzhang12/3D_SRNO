#!/bin/bash
#SBATCH --qos=high
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --job-name=mri-exp3-shared-gate
#SBATCH --output=reports/slurm/%x-%j.out
#SBATCH --error=reports/slurm/%x-%j.err
set -euo pipefail
cd "${GSBM_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
source scripts/mri_common.sh
gsbm_environment
: "${SHARED_STUDY_DIR:?}" "${EXP1_PROJECT_ROOT:?}" "${MRI_PERCEPTUAL_WEIGHTS:?}"
# Reuse the proven full-size gate only with byte-identical training/metric code.
for name in hybrid jetatlas mri_v2 mri_v2_train mri_v2_losses mri_metrics mri_perceptual mri_infer; do
    cmp "$EXP1_PROJECT_ROOT/src/vemgsbm/$name.py" "$PWD/src/vemgsbm/$name.py"
done
"$GSBM_PYTHON" -m pytest -q --junitxml="$SHARED_STUDY_DIR/pytest.xml"
"$GSBM_PYTHON" tools/mri_v2_e2e.py --variants 3 --device cuda \
    --weights-dir "$MRI_PERCEPTUAL_WEIGHTS" --output "$SHARED_STUDY_DIR/e2e_train.json"
"$GSBM_PYTHON" - "$SHARED_STUDY_DIR" <<'PY'
import json,sys
from pathlib import Path
from vemgsbm.mri_inventory import prepare_inventory
root=Path(sys.argv[1]);row=json.loads((root/'e2e_train.json').read_text())['variants']['3']
if row['status']!='passed':raise RuntimeError('Single-training checkpoint smoke failed')
prepare_inventory(row['config'],root/'smoke_inventory.json.gz')
(root/'smoke_checkpoint.txt').write_text(row['checkpoint'])
PY
gsbm_torchrun -m vemgsbm.mri_v2_compare --checkpoint "$(<"$SHARED_STUDY_DIR/smoke_checkpoint.txt")" \
    --case-inventory "$SHARED_STUDY_DIR/smoke_inventory.json.gz" \
    --output "$SHARED_STUDY_DIR/smoke_pair" --device cuda
touch "$SHARED_STUDY_DIR/gate_passed"
