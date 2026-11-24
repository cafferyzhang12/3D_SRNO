# MRI XY×4、Z×1：单向 x1 Bridge Matching

新版入口是 `configs/mri_haar_bridge_x4.yaml`。历史双向漂移模型仍由检查点内的旧配置构造，原实验脚本改用 `configs/mri_haar_bridge_legacy_x4.yaml`，可以复评既有 A/B 权重。新版检查点版本为 5，不可把旧漂移权重当作新版 x1 权重加载。

## 实现与训练协议

- 16 通道 TRAIN 残差统计使用全切片 `HaarXY(HR - bilinear(LR))`，float64 在线合并均值与总体方差。统计所有保留切片，包括不足 5 层的连续段，不重复抽样窗口。缓存包含病例身份、归一化约定、编码版本和计数；相同路径身份不一致会拒绝复用。均值/标准差随权重保存，部署不依赖外部统计文件。
- 白化使用 `(R - mean)/(std + 1e-6)`。桥为 `(1-t) X0 + t R_white + sigma_bridge sqrt(t(1-t)) eps`；`sigma_0=0.03`，`sigma_bridge=0.08`，训练时间在 `[1e-4, 1-1e-4]`。
- x1 网络仅有干净残差输出头，没有 direction embedding、反向损失或漂移头。宽度为 96/160/256；LR 深层新增 XY 空洞卷积残差块，dilation 为 `(1,1,1)/(1,2,2)/(1,4,4)`，Z 不下采样。总参数 15,550,832。
- 损失包含白化频带 Charbonnier/L1、逆白化逆 Haar 后的像素 MSE、每卡 1/8 批次的 rollout MSE。rollout 默认 3 步，前缀及其中的条件特征停止梯度，最后读出和对应 LR 编码参与反传。
- 每卡 batch 64；4 卡全局 batch 256；每轮 262,144 块，即 1,024 次优化更新。梯度范数裁剪 1.0；按更新次数执行 2 轮 warmup + 40 轮总时域 cosine；EMA decay 0.999。整病例验证、选模和部署默认使用 EMA，原始训练参数与 optimizer/scheduler/EMA 状态同时保存用于恢复。
- 每轮整病例 VAL 保留原 Hann 融合后 `[0,1]` 裁剪、pooled MSE/PSNR、切片等权 SSIM。新增 16 通道 Haar MSE 及 15 个非 LL 通道的等权 MSE，不改变原主指标。第 15/24/40 轮额外保留检查点。
- 训练期 MLS 在白化 TRAIN 残差上拟合二阶图册，按病例分折避免自身拟合；用停止梯度的真值残差选图并求二阶曲面的局部 Jacobian，惩罚预测误差的法向分量。每块默认抽取最多 256 个位置控制开销；部署不加载或检索该图册。纯主干默认 `mls_reg_weight: 0.0`，达标后才改为 0.05，并使用新的 run_dir。

## ODE 的数学含义

本次按指导中的明确公式实现 `u=(R_hat-X)/(1-t)` 的无噪声端点 ODE。它是 Brownian bridge **前向漂移去噪声后的确定性采样器**，不是该随机桥的严格 probability-flow ODE；后者还需要 score 修正。不要把消融结果描述成已证明具有相同 Fokker–Planck 边缘分布，或已求解最优 Schrödinger bridge。相关区别见 [Stochastic Interpolants](https://arxiv.org/abs/2303.08797)。

ODE 积分过程不抽取扩散噪声，但 `X0` 仍来自非退化高斯先验。因此结果在固定种子下可复现，并非对所有初始种子相同。每个病例内按固定瓦片次序抽取 X0/SDE 噪声，改变 `tile_batch` 不改变随机轨迹。

Euler 的 N 步对应 N 次网络调用；Heun 对应 `2*N-1` 次，报告会记录实际调用数。最后一步直接读取端点，不计算末端除法。新版 NFE=1 在 `t=0` 读出。旧 B 零训练消融保留其历史漂移和 `.02/.98` 时间截断，只把积分噪声关掉，防止把旧漂移头误作新版 x1 头。

SDE 必须显式开启，噪声绑定 `sigma_bridge`；显式提供不一致的噪声会报错。支持多轨迹均值。Euler/Heun 的积分规则与末步读出都在 `bridge.py`。

## 本地验证（PowerShell）

```powershell
Set-Location D:\3D_SRNO_4
conda activate zsztorch
$env:PYTHONPATH = "$PWD\src;$PWD\dist\x1_atlas_dependency"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest -q --basetemp=outputs/x1_manual_tests
```

`dist/x1_atlas_dependency` 是从原集群项目复制的只读共享图册 Python 源码，供历史图册测试导入；未修改 base，也没有安装或修改系统文件。新版纯 x1 主干本身不依赖 `neural_atlas`。本地测试使用生成的微型 PNG，不代表真实 MRI 四卡冒烟或指标验收。

## 集群提交

所有统计、测试、模型加载、训练与评估都在 A800 `sbatch` 分配内执行；登录节点仅同步、校验文件、查询和提交。使用 `fundiff`、`high` QoS，每个新阶段部署到独立项目目录。下面的路径与本次提交对应；`submit_mri_x1.sh` 会拒绝重复的任务回执，重新研究需选择新 RUN_ID。

```bash
cd "$HOME/3D_SRNO_4/x1_bridge_20261123"
export GSBM_PROJECT_ROOT="$PWD" MRI_GPUS=4 RUN_ID=x1_core_20261123
export GSBM_PYTHON="$HOME/.conda/envs/fundiff/bin/python"
export NEURAL_ATLAS_SRC="$HOME/Learning_Geometric_State_Spaces_for_Probabilistic_Trajectory_Inference/neural_atlas_lab/src"
export MRI_CASE_INVENTORY="$HOME/3D_SRNO_4/vem_gsbm_mls_v2_20261120_f08f30129d6a/runs/mri/mri_x4_gsbm_mls_v2_20261120/mri_case_inventory.json.gz"
export LEGACY_CHECKPOINT="$HOME/3D_SRNO_4/paired_xy4_20261121_r1/haar_four_r2/runs/mri/x4_large_r2b_bridge/bridge/best.pt"
export ODE_ABLATION="$HOME/3D_SRNO_4/x1_ablation_20261123/outputs/mri/x1_legacy_ode_20261123/ablation.json"
export ABLATION_JOB=5303583
bash scripts/submit_mri_x1.sh
```

依赖顺序：历史 B 五种步数完整 VAL → 全量 pytest、TRAIN 统计、固定批次过拟合、两轮四卡真实 PNG 冒烟 → 40 轮纯主干训练 → 最佳 EMA 整病例 VAL 与 `core_gate.json`。冒烟每轮缩减至 4,096 块，但保持正式模型、每卡 batch 64、完整 TRAIN 统计和完整 10 例 VAL。

旧 B 的 4 步与 20 步 pooled PSNR 差小于 0.02 dB 时，生成的新版配置使用 4 步，否则使用 8 步。这只是后续起始设置，不能据此断言新版模型的精度拐点也相同。

```bash
# 查询本次提交回执及排队状态；不会重复提交。
cat reports/mri/x1_core_20261123_jobs.tsv
squeue -u "$USER"
tail -n 20 reports/slurm/mri-x1-train-<JOB_ID>.out

# 已分配的 A800 作业内直接运行/恢复（不要在登录节点执行这些 Python 命令）
export PYTHONPATH="$PWD/src:$NEURAL_ATLAS_SRC${PYTHONPATH:+:$PYTHONPATH}"
"$GSBM_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m vemgsbm.train --config runs/mri/x1_core_20261123/experiment.yaml \
  --case-inventory "$MRI_CASE_INVENTORY" --device cuda
# 恢复时追加：--resume runs/mri/x1_core_20261123/bridge/last.pt
```

验证后单独提交训练可使用 `sbatch scripts/mri_x1_train.sbatch`；它会检查真实冒烟验收文件。正式训练完成后单独复评用 `sbatch scripts/mri_x1_val.sbatch`。保持相同输出目录进行 `--resume-cases` 复评必须使用相同代码、权重、输入和采样协议；协议变化要换输出目录。

## 推理和第二阶段接口

以下命令同样必须在已分配的计算节点运行：

```bash
# NFE=1/2/4/8/20：替换 --steps；与训练一致的完整病例协议
"$GSBM_PYTHON" -m vemgsbm.mri_infer --checkpoint "$CHECKPOINT" \
  --root-lq "$MRI_DATA_ROOT/val_small/LQ/X4_png" --root-hq "$MRI_DATA_ROOT/val_small/HQ_png" \
  --output-root outputs/mri/x1_ode8 --split val --sampler-type ode --steps 8 \
  --samples 1 --device cuda --save-volume-limit 0 --resume-cases
# 多样性评价：使用新的输出目录，并改为 --sampler-type sde --steps 8 --samples 4
# 二阶积分：--sampler-type ode --solver heun（实际网络调用为 2*steps-1）
```

可选拉直训练配置增加：

```yaml
straightening:
  teacher_checkpoint: runs/mri/x1_core_20261123/bridge/best.pt
  teacher_steps: 8
sigma_bridge: 0.0
# 同时把已有 path.sigma、sampling.sigma 设为 0.0；保留 sigma_0=0.03。
```

`sample_induced_coupling` 从冻结的一阶段 EMA 模型生成 `(X0, R_teacher)`，二阶段使用无桥噪声的直线插值训练；像素和 rollout 项仍监督配对 HR。教师与学生必须具有相同数据身份和白化统计。应使用新的 run_dir，并重新在 VAL 上验证 2/4 步，而不能直接覆盖一阶段结果。

`core_gate.json` 记录第 15/24/40 轮、最后三轮是否均达到 32.10 dB/0.900、相邻轮次波动和是否满足 MLS 放行条件。指标是实验验收目标，代码修改不能保证达成。固定 84 例 TEST、独立 84 例及三种随机种子确认仍需在主干和 MLS 消融完成、外部基线按同一 TRAIN 微调并冻结协议后执行；当前自动流水线结束于 VAL，不会提前消费 TEST 进行调参。
