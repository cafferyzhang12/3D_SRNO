#!/bin/bash
#SBATCH --job-name=momentum-mri-x4
#SBATCH --qos=high
#SBATCH --output=reports/slurm/mri-momentum-%j.out
#SBATCH --error=reports/slurm/mri-momentum-%j.err
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
set -euo pipefail
cd "${GSBM_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/mri_common.sh
gsbm_environment
: "${SUBSET_MANIFEST:?Set the locked test cohort manifest}"
[[ -f "$SUBSET_MANIFEST" ]] || { echo 'Missing locked test manifest' >&2; exit 2; }
export CONFIG="${CONFIG:-configs/mri_momentum_x4.yaml}"
mkdir -p runs/mri reports/mri/run_configs
mkdir "runs/mri/$RUN_ID"

# Generate independent writable indices for the smoke and formal runs.
mapfile -t configs < <("$GSBM_PYTHON" - <<'PY'
import copy
import os
from pathlib import Path
import yaml

template = yaml.safe_load(Path(os.environ['CONFIG']).read_text(encoding='utf-8'))
run = Path('runs/mri') / os.environ['RUN_ID']
for profile in ('smoke', 'formal'):
    cfg = copy.deepcopy(template)
    directory = run / 'smoke' if profile == 'smoke' else run
    cfg['run_dir'] = str(directory)
    cfg['data']['scan_cache'] = {split: str(directory / 'index' / (split + '.json'))
                                 for split in ('train', 'val', 'test')}
    if profile == 'smoke':
        cfg['data']['debug_case_limit'] = 2
        cfg['model']['width'] = 4
        cfg['train'].update(epochs=1, batch_size=1, samples_per_epoch=8,
                            val_samples_per_epoch=4, workers=0)
        cfg['pca'].update(bank_patches=8, tokens_per_patch=8, centers=4, neighbors=12)
        cfg['path'].update(iterations=2, quadrature=3)
        cfg['sampling']['steps'] = 2
    filename = Path('reports/mri/run_configs') / f"{os.environ['RUN_ID']}_{os.environ['SLURM_JOB_ID']}_{profile}.yaml"
    with filename.open('x', encoding='utf-8') as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    print(filename.resolve())
PY
)
[[ "${#configs[@]}" == 2 ]] || { echo 'Could not generate momentum configs' >&2; exit 2; }
"$GSBM_PYTHON" -m pytest -q tests/test_momentum.py tests/test_endpoint_safe.py
gsbm_torchrun -m vemgsbm.momentum.train --config "${configs[0]}" --device cuda
mapfile -t val_roots < <("$GSBM_PYTHON" - "${configs[0]}" <<'PY'
import sys
from vemgsbm.runtime import read_config
roots = read_config(sys.argv[1])["data"]["val"]
print(roots["root_lq"])
print(roots["root_hq"])
PY
)
[[ "${#val_roots[@]}" == 2 ]] || { echo 'Could not read validation roots' >&2; exit 2; }
gsbm_torchrun -m vemgsbm.mri_infer --checkpoint "runs/mri/$RUN_ID/smoke/momentum/best.pt" \
    --split val --root-lq "${val_roots[0]}" \
    --root-hq "${val_roots[1]}" \
    --scan-cache "runs/mri/$RUN_ID/smoke/index/val_inference.json" \
    --output-root "outputs/mri/${RUN_ID}_validation_smoke" --device cuda \
    --max-cases 1 --test-fraction 1 --steps 2 --save-volume-limit 0
gsbm_torchrun -m vemgsbm.momentum.train --config "${configs[1]}" --device cuda
export CHECKPOINT="$PWD/runs/mri/$RUN_ID/momentum/best.pt"
export INFERENCE_VARIANT=endpoint_safe
source scripts/mri_whole_val.sh
