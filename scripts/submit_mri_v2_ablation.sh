#!/bin/bash
# Administrative submission only; every computation is in an A800 allocation.
set -euo pipefail
: "${RUN_ID:?}" "${MRI_CASE_INVENTORY:?}"
export GSBM_PROJECT_ROOT="$PWD" MRI_GPUS=4 RUN_ID MRI_CASE_INVENTORY
export V2_ARMS="${V2_ARMS:-1 2 3 4}" MRI_TEST_MANIFEST="${MRI_TEST_MANIFEST:-}"
[[ "$V2_ARMS" =~ ^[1-4](\ [1-4])*$ ]]
read -r -a arms <<< "$V2_ARMS"
declare -A seen_arms=()
for arm in "${arms[@]}"; do
    [[ ! "${seen_arms[$arm]+set}" ]] || { echo "Duplicate v2 arm: $arm" >&2; exit 2; }
    seen_arms[$arm]=1
done
[[ -z "$MRI_TEST_MANIFEST" || -r "$MRI_TEST_MANIFEST" ]]
source scripts/mri_slurm_dependencies.sh
mkdir -p reports/slurm reports/mri
receipt="reports/mri/${RUN_ID}_v2_jobs.tsv"
(set -o noclobber; : > "$receipt")
gate=$(mri_slurm_submit "" scripts/mri_v2_gate.sbatch)
printf 'gate\t%s\n' "$gate" >> "$receipt"
val_jobs=()
for V2_ARM in "${arms[@]}"; do
    export V2_ARM
    train=$(mri_slurm_submit "$gate" scripts/mri_v2_train.sbatch)
    printf 'exp%s_train\t%s\n' "$V2_ARM" "$train" >> "$receipt"
    eval_job=$(mri_slurm_submit "$train" scripts/mri_v2_eval.sbatch)
    printf 'exp%s_val\t%s\n' "$V2_ARM" "$eval_job" >> "$receipt"
    val_jobs+=("$eval_job")
done
if [[ -n "$MRI_TEST_MANIFEST" ]]; then
    val_dependencies=$(IFS=:; printf '%s' "${val_jobs[*]}")
    test_job=$(mri_slurm_submit "$val_dependencies" scripts/mri_v2_test.sbatch)
    printf 'test15\t%s\n' "$test_job" >> "$receipt"
fi
cat "$receipt"
