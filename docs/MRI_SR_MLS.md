# MRI SR reconstruction with paired MLS

The `mri_haar_sr_mls_x4.yaml` experiment optimizes reconstruction from the actual
inference start and from a short deployment-aligned trajectory. It keeps the
16-band TRAIN whitening transform as a network parameterization. Reconstruction,
high-frequency, XY edge, and curved MLS errors are measured in physical intensity
units and share one TRAIN variance normalization. The weights are 1 for the
start, 1 for rollout, 0.1 for high-frequency error, 0.05 for edges, and 0.05 for MLS.
Optional bridge auxiliary supervision has weight zero.

The initial state and bridge noise are zero. Training and deployment both use
four Euler ODE steps with endpoint readout. Rollout supervision uses one eighth
of each rank's batch and detaches the preceding trajectory. The formal budget is
40 epochs, 262,144 TRAIN patches per epoch, batch 64 on each of four A800 GPUs,
with widths 96/160/256 and EMA checkpoint selection on complete whole-case VAL.

The paired curved atlas is fitted exclusively from TRAIN residuals and paired
baseline Haar coefficients. Five case-disjoint folds prevent a training case
from fitting the atlas used to score its own reconstruction. Each fold fits at
most 65,536 sites, 128 centers, and 64 neighbors, with tangent dimension four.
The bank retains 256 sites from four patches per TRAIN case. MLS is a training
loss; deployment requires no atlas query or TEST fitting.

Run preparation takes data paths from the historical checkpoint and verifies
3,919 TRAIN cases / 292,657 slices and 10 VAL cases / 595 slices against the
inventory. `MRI_HAAR_STATISTICS` may point to an existing full-TRAIN whitening
cache. Its entire TRAIN identity must match; otherwise the gate fails. Without
an explicit path, the legacy checkpoint's cache is reused if valid, or a new
cache is computed on all TRAIN slices.

All computation runs inside `high` QoS A800 allocations. The gate runs the full
pytest suite, prepares full-TRAIN statistics, builds the MLS bank and checks a
bounded atlas fit before initializing NCCL, and optimizes four fixed TRAIN
patches for 24 updates. Both the composite loss and actual four-step deployment
pixel MSE must decrease. A one-epoch, four-GPU smoke then exercises batch 64 per
rank, MLS, finite gradients, and complete whole-case VAL. The bank is reused by
all ranks; each rank constructs its bounded fold fits locally.

Submit the gate from the synchronized project root:

```bash
export RUN_ID=sr_mls_YYYYMMDD
export MRI_CASE_INVENTORY=/absolute/path/inventory.json.gz
export LEGACY_CHECKPOINT=/absolute/path/historical/bridge/best.pt
export MRI_TEST_MANIFEST=/absolute/path/historical/test_manifest.json
# Optional exact-identity cache reuse:
export MRI_HAAR_STATISTICS=/absolute/path/train_haar_statistics.pt
bash scripts/submit_mri_sr.sh gate
```

After the gate succeeds, submit the formal training and dependent evaluation:

```bash
bash scripts/submit_mri_sr.sh formal
```

The receipt is `reports/mri/${RUN_ID}_sr_jobs.tsv`. The `all` phase can submit the
entire dependency chain at once. The evaluation job performs complete VAL using
the VAL-selected best checkpoint, freezes its checkpoint hash, inventory,
inference settings and historical TEST manifest through `mri_freeze`, then
replays exactly 84 TEST cases / 14,991 slices / 3,929,800,704 pixels. This is the
authorized final TEST evaluation: it does not apply the earlier 32.10 dB gate,
fit on TEST, select settings on TEST, or scan alternative TEST cohorts.

Outputs are `outputs/mri/${RUN_ID}_val`, `outputs/mri/${RUN_ID}_test84`, and
`runs/mri/${RUN_ID}/frozen_test`. The SR report records late stability using only
negative PSNR changes within the final five epochs; early positive learning is
not classified as instability. This diagnostic does not block the authorized
frozen evaluation. Once formal training has started and evaluation is queued,
the jobs proceed without continued chat monitoring.
