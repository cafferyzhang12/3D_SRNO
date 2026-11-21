#!/bin/bash
# Run from a cluster project checkout after the input/VAL baselines are prepared.
set -euo pipefail
cd "${GSBM_PROJECT_ROOT:-$PWD}"
: "${STUDY_ID:?Set an independent study identifier}" "${FREEZE_ROOT:?Set the freeze destination}"
: "${STAGE1_RESULTS:?Set the historical sampler VAL result folder}" "${LEGACY_RESULTS:?Set the six-method VAL parent}"
: "${HISTORY_ROOT:?Set the epoch-38 snapshot root}" "${ARCHIVE_ROOT:?Set the historical snapshot parent}"
: "${EXTERNAL_RESULTS_ROOT:?Set the external baseline checkout}" "${EXTERNAL_PREFIX:?Set the baseline RUN_ID prefix}"
bash scripts/submit_mri_haar_study.sh all
exec bash scripts/submit_mri_confirmation.sh --wait
