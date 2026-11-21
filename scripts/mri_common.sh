#!/bin/bash
# Setup and config generation are called only inside a Slurm A800 allocation.
gsbm_environment() {
    : "${SLURM_JOB_ID:?Run computation through sbatch on allocated A800 GPUs}"
    [[ "${SLURM_JOB_NUM_NODES:-1}" == 1 && "${SLURM_NTASKS:-1}" == 1 ]] || {
        echo 'Expected one node and one Slurm task.' >&2; return 2;
    }
    cd "${GSBM_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
    export GSBM_PYTHON="${GSBM_PYTHON:-$HOME/.conda/envs/fundiff/bin/python}"
    [[ -x "$GSBM_PYTHON" ]] || { echo "fundiff Python missing: $GSBM_PYTHON" >&2; return 2; }
    if [[ -n "${NEURAL_ATLAS_SRC:-}" ]]; then
        [[ -f "$NEURAL_ATLAS_SRC/neural_atlas/mls_atlas.py" ]] || {
            echo "MLS source missing in $NEURAL_ATLAS_SRC" >&2; return 2;
        }
    fi
    export PYTHONPATH="$PWD/src${NEURAL_ATLAS_SRC:+:$NEURAL_ATLAS_SRC}${PYTHONPATH:+:$PYTHONPATH}"
    export MRI_GPUS="${MRI_GPUS:-${SLURM_GPUS_ON_NODE:-4}}"
    [[ "$MRI_GPUS" == 1 || "$MRI_GPUS" == 4 ]] || { echo "Expected one or four GPUs" >&2; return 2; }
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=4
    export NUMEXPR_NUM_THREADS=4
    export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    export RUN_ID="${RUN_ID:-mri_gsbm_${SLURM_JOB_ID}}"
    [[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$ ]] || { echo 'Invalid RUN_ID' >&2; return 2; }
    mkdir -p reports/mri reports/slurm
    "$GSBM_PYTHON" - <<'PY'
import os, sys
if sys.version_info < (3, 10):
    raise RuntimeError('vem_gsbm requires Python >= 3.10')
import torch
count = int(os.environ['MRI_GPUS'])
if not torch.cuda.is_available() or torch.cuda.device_count() != count:
    raise RuntimeError('Visible GPUs must match the Slurm allocation')
names = [torch.cuda.get_device_name(i) for i in range(count)]
if any('A800' not in name for name in names):
    raise RuntimeError(f'Expected A800 GPUs: {names}')
print({'job_id': os.environ['SLURM_JOB_ID'], 'python': sys.executable,
       'torch': torch.__version__, 'gpus': names,
       'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')}, flush=True)
PY
}

gsbm_config() {
    local template="$1" stage="$2" profile="${3:-formal}"
    "$GSBM_PYTHON" - "$template" "$stage" "$profile" <<'PY'
import os, sys
from pathlib import Path
import yaml
template, stage, profile = sys.argv[1:]
if stage not in ('representation', 'bridge'):
    raise ValueError('Expected representation or bridge stage')
cfg = yaml.safe_load(Path(template).read_text(encoding='utf-8'))
if os.environ.get('MRI_DATA_ROOT'):
    root = Path(os.environ['MRI_DATA_ROOT'])
    for split in ('train', 'val', 'test'):
        for key, path in cfg['data'][split].items():
            cfg['data'][split][key] = str(root / Path(path).relative_to('data/MRI_SR'))
if cfg['data']['kind'] != 'paired_mri_png' or cfg['data']['degradation'] != 'paired_xy':
    raise ValueError('This entry requires paired MRI XY data')
if cfg['data']['scale'] != 4:
    raise ValueError('This pipeline uses the supplied X4 test data')
cfg['run_dir'] = str(Path('runs/mri') / os.environ['RUN_ID'])
cfg['train']['epochs'] = cfg['train'].get(stage + '_epochs', cfg['train']['epochs'])
native_haar = cfg['mode'] in ('haar_direct', 'haar_bridge')
if os.environ.get('REPRESENTATION_CHECKPOINT') and not native_haar:
    cfg['representation']['checkpoint'] = os.environ['REPRESENTATION_CHECKPOINT']
for env_name, key in [('BATCH_SIZE', 'batch_size'), ('EPOCHS', 'epochs'),
                       ('SAMPLES_PER_EPOCH', 'samples_per_epoch'),
                       ('VAL_SAMPLES_PER_EPOCH', 'val_samples_per_epoch')]:
    if os.environ.get(env_name):
        value = int(os.environ[env_name])
        if value < 1:
            raise ValueError(f'{env_name} must be positive')
        cfg['train'][key] = value
if profile == 'validate':
    if cfg.get('atlas') or native_haar:
        cfg['run_dir'] = str(Path(cfg['run_dir']) / 'smoke')
    cfg['data']['debug_case_limit'] = 2
    cfg['data']['scan_cache'] = {
        split: str(Path(cfg['run_dir']) / 'index' / f'{split}.json')
        for split in cfg['data']['scan_cache']
    }
    cfg['train'].update(epochs=1, batch_size=1, samples_per_epoch=8,
                        val_samples_per_epoch=4, workers=0)
    if cfg.get('chart_momentum') is not None:
        cfg['data']['debug_case_limit'] = 8
        cfg['train'].update(samples_per_epoch=32, val_samples_per_epoch=8)
        cfg['atlas'].update(bank_patches=24, charts=4, neighborhoods=[12, 24, 48], dim_candidates=[2, 3])
        cfg['atlas']['mls'].update(degree=1, neighbors=24, anchor_limit=6)
    elif not native_haar:
        cfg['representation'].update(width=4, feature_dim=8, clusters=4, bank_samples=8)
    if cfg.get('atlas') and cfg.get('chart_momentum') is None:
        cfg['train'].update(samples_per_epoch=32, val_samples_per_epoch=8)
        cfg['representation']['bank_samples'] = 32
        cfg['atlas'].update(charts=2, neighborhoods=[8, 16, 32], dim_candidates=[2])
        cfg['atlas']['mls'].update(degree=1, neighbors=24, anchor_limit=4)
    if native_haar:
        cfg['model']['widths'] = [4, 8, 12]
    else:
        cfg['model']['width'] = 4
    cfg['path']['iterations'] = 2
    geometry_sampling = any(cfg['sampling'].get(key) for key in ('geometry_guidance', 'momentum_geometry'))
    cfg['sampling']['steps'] = 3 if geometry_sampling else 2
elif profile != 'formal':
    raise ValueError('Unknown config profile')
cfg['train'][stage + '_epochs'] = cfg['train']['epochs']
if (stage == 'bridge' and not native_haar and cfg.get('chart_momentum') is None
        and not cfg['representation'].get('checkpoint') and not os.environ.get('RESUME')):
    raise ValueError('Set REPRESENTATION_CHECKPOINT to the matching representation/best.pt')
folder = Path('reports/mri/run_configs')
folder.mkdir(parents=True, exist_ok=True)
suffix = f"_{profile}" if cfg.get('atlas') else ""
path = folder / f"{os.environ['RUN_ID']}_{os.environ['SLURM_JOB_ID']}_{stage}{suffix}.yaml"
with path.open('x', encoding='utf-8') as stream:
    yaml.safe_dump(cfg, stream, sort_keys=False)
print(path.resolve())
PY
}

gsbm_torchrun() {
    "$GSBM_PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
        --nproc-per-node="$MRI_GPUS" --max-restarts=0 "$@"
}
