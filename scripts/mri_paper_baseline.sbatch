#!/bin/bash
#SBATCH --job-name=mri-paper-baseline
#SBATCH --qos=high
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --time=6-00:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --output=reports/slurm/%x-%j.out
#SBATCH --error=reports/slurm/%x-%j.err
#SBATCH --open-mode=append
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export GSBM_PROJECT_ROOT="$PWD" MRI_GPUS=1
source scripts/mri_common.sh
gsbm_environment
export PATH="$(dirname "$GSBM_PYTHON"):$PATH"
: "${MRI_BASELINE_REFERENCE:?}" "${MRI_BASELINE_CONFIG:?}" "${MRI_CASE_INVENTORY:?}" \
  "${MRI_TEST_MANIFEST:?}" "${MRI_PERCEPTUAL_WEIGHTS:?}" "${MRI_BASELINE_SOURCE_SHA256:?}"
export TORCH_HOME="$MRI_PERCEPTUAL_WEIGHTS"

run_baseline() {
    local method="$1" batch="$2" status
    shift 2
    if [[ "$method" == medgssr ]]; then
        source /gpfs/share/software/module/tools/modules/init/profile.sh
        module load gcc/12.3.0 cuda/12.8
        local nvcc_path
        nvcc_path=$(command -v nvcc)
        export CUDA_HOME="$(dirname "$(dirname "$nvcc_path")")"
        export TORCH_CUDA_ARCH_LIST=8.0
        export TORCH_EXTENSIONS_DIR="$PWD/cache/torch-extensions"
        export MAX_JOBS=2
    fi
    local command=("$GSBM_PYTHON" -m vemgsbm.mri_paper_baseline --method "$method"
        --upstream "$MRI_BASELINE_REFERENCE" --config "$MRI_BASELINE_CONFIG"
        --inventory "$MRI_CASE_INVENTORY" --manifest "$MRI_TEST_MANIFEST"
        --weights-dir "$MRI_PERCEPTUAL_WEIGHTS" --output "$PWD/runs/mri/$method"
        --micro-batch "$batch" "$@")
    if [[ "$#" != 0 ]]; then
        "${command[@]}"
        return
    fi
    local child signalled=0
    trap 'signalled=1; kill -USR1 "$child" 2>/dev/null || true' USR1
    "${command[@]}" &
    child=$!
    set +e
    wait "$child"
    status=$?
    if [[ "$signalled" == 1 && "$status" == 138 ]]; then
        wait "$child"
        status=$?
    fi
    set -e
    trap - USR1
    if [[ "$status" == 75 ]]; then
        scontrol requeue "$SLURM_JOB_ID"
        return
    fi
    return "$status"
}

case "${1:-}" in
    smoke) run_baseline 3d-edm 8 --smoke; run_baseline medgssr 4 --smoke ;;
    3d-edm) run_baseline 3d-edm 8 ;;
    medgssr) run_baseline medgssr 4 ;;
    *) echo 'Expected smoke, 3d-edm, or medgssr' >&2; exit 2 ;;
esac
