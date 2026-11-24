#!/bin/bash
#SBATCH --job-name=mri-x1-ablation
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
: "${CHECKPOINT:?Provide the historical supervised B epoch-24 checkpoint}"
export MRI_HAAR_READOUT=endpoint MRI_SAMPLER_TYPE=ode MRI_NUM_SAMPLES=1
for MRI_INFER_STEPS in 20 8 4 2 1; do
    export MRI_INFER_STEPS
    export OUTPUT_ROOT="$PWD/outputs/mri/${RUN_ID}/ode_${MRI_INFER_STEPS}"
    source scripts/mri_whole_val.sh
done
"$GSBM_PYTHON" - "$PWD/outputs/mri/$RUN_ID" <<'PY'
import json, sys
from pathlib import Path
from vemgsbm.runtime import atomic_save
root = Path(sys.argv[1])
rows = []
for n in (20, 8, 4, 2, 1):
    s = json.loads((root / f'ode_{n}/summary.json').read_text())
    rows.append(dict(nfe=n, psnr=s['all_psnr_db'], ssim=s['mean_slice_ssim'],
                     seconds=[c['evaluation_seconds'] for c in s['cases']],
                     cases=len(s['cases']), slices=s['total_slices']))
if any(r['cases'] != 10 or r['slices'] != 595 for r in rows):
    raise ValueError('Ablation must cover all 10 VAL cases / 595 slices')
atomic_save(dict(rows=rows, suggested_steps=4 if abs(rows[2]['psnr']-rows[0]['psnr']) < .02 else 8),
            root/'ablation.json', json_file=True)
print(json.dumps(rows), flush=True)
PY
