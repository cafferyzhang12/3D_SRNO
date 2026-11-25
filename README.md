# 3D MRI 超分辨率

将 MLS 图册提供的几何用于配对 LR→HR 的 MRI 重建。支持配对漂移匹配、动量桥，以及由真实 HR 监督的确定性端点迭代重建。

## 安装

从本目录运行。先从几何学习仓库安装共享图册包，再安装超分项目。

```powershell
python -m pip install -e "<geometry-repository>/neural_atlas_lab"
python -m pip install -e ".[atlas,test]"
python -m pytest -q
```

集群的 `fundiff` 环境也需要安装同版本图册包。可以在登录节点部署源码后安装对应路径，或先构建 `neural_atlas_lab` wheel，再在集群环境中安装该 wheel。

## 运行

- [配对 SR 与二阶 MLS](docs/MRI_SR_MLS.md)说明当前重建目标：从部署起点及模型自身轨迹监督 HR，使用原始强度单位的像素、高频、边缘与可微曲面约束；包含四卡 A800、high QoS 的训练及固定 TEST 入口。

- [单向 x1 Bridge Matching](docs/MRI_X1_BRIDGE.md)说明新版白化、无噪声端点 ODE、EMA、训练期 MLS，以及四卡训练后自动验证的运行命令。

- [MRI MLS 桥](docs/MRI_MLS.md)说明训练 HR 如何构建图册、配置和推断入口。
- [集群命令](docs/CLUSTER.md)说明数据路径、表征训练、桥训练和留出评估。
- [动量桥](docs/MOMENTUM.md)说明加速度匹配。
- [历史配对 MRI 汇总](reports/mri/summary.json)记录同一批 84 个病例的 endpoint-safe 和动量桥结果。

配对病例清单和本地数据放在 `data/private/`。源码发布入口见 [发布说明](docs/RELEASE.md)。


## SSIM-preserving MRI fine-tuning study

`bash scripts/submit_mri_ssim.sh` submits a four-A800, high-QOS preflight, matched
fine-tuning study, and dependent frozen evaluation. Set `RUN_ID`,
`BASELINE_CHECKPOINT` (immutable deterministic four-step x1 EMA checkpoint),
`MRI_MLS_BANK` (matching full-TRAIN paired bank), `MRI_CASE_INVENTORY`,
`MRI_TEST_MANIFEST` (historical 84-case manifest), `GSBM_PYTHON` (fundiff Python),
and `NEURAL_ATLAS_SRC` before submission. Run from the synchronized project root.
Compute, including tests and pretrained feature smoke checks, runs inside Slurm.

All four variants start from the same baseline EMA, use fresh optimizers, 8 epochs,
262144 patches/epoch, batch 544/GPU (global 2176), four data workers/rank,
learning rate 2e-5, and four-step deployment:
control; SSIM weight 0.3; SSIM plus stratified MLS; and the latter plus MLS XY
neighbor weight 0.1 (inside MLS weight 0.05). SSIM uses the evaluation 7x7 uniform
window/sample covariance and the existing one-eighth self-rollout budget.
Neighbor constraints compare target-anchored curved defects, never Z differences.

The 544/rank setting was measured on four 80-GiB A800 GPUs; the running SSIM
variant occupied about 80% of device memory. This is workload-specific, not an
OOM guarantee for other hardware. The fixed patch budget gives 121 optimizer
updates/epoch (including the final partial batch), compared with 1024 at 64/rank.
The bounded preflight smoke retains batch 64/rank and 1024 total patches.

Whole-VAL PSNR must be at least the re-evaluated frozen baseline, with no dB
allowance, and SSIM must improve before `best.pt` replaces the baseline fallback.
`runs/mri/$RUN_ID/selection.json` records all ablations and whether any improvement
was found. This selection constraint does not guarantee unseen TEST performance.
The old training, weights, and results remain preserved in their original project.

Evaluation reports pooled PSNR, slice-equal SSIM, LPIPS-Alex v0.1 and cohort FID
for both baseline and selected weights on VAL and the fixed TEST cohort.
Install `lpips==0.1.4`, `torch-fidelity==0.3.0`, and a torchvision version matching
the existing torch in fundiff only (torch2.8 pairs with torchvision0.23); avoid
upgrading the working torch environment through unconstrained dependencies.
Preflight downloads official feature weights into `cache/perceptual`, validates
hashes and real feature outputs, and subsequent evaluation uses verified offline
weights. FID uses TF-compatible Inception2048 on RGB-repeated grayscale uint8
slices and globally merged feature moments. These 2D ImageNet metrics depend on
the cohort/protocol; correlated MRI slices and the 595-slice VAL covariance rank
limit are recorded explicitly. LPIPS/FID never select TEST cases or checkpoints.
