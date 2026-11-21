# 原生 LR 残差桥实验

## 提交与恢复

在集群项目目录、完成第 38 轮整病例 VAL 诊断后运行以下命令。设置 `STUDY_ID` 为本轮独立标识，`MRI_DATA_ROOT` 和 `NEURAL_ATLAS_SRC` 沿用已核实的本机路径。每个新训练配置先申请单节点四张 A800、一个 task，用 torchrun 运行固定小批过拟合及一个完整 VAL 病例。通过后每个正式实验申请一张 A800，使用 fundiff、高 QoS。阶段间的模型选择也通过 sbatch 执行。

```bash
export STUDY_ID=x4_haar_study_01
bash scripts/submit_mri_haar_study.sh backbone
```

`backbone` 作业完成后检查 `reports/mri/studies/$STUDY_ID/backbone.json` 内的 pooled PSNR、SSIM、边缘 MSE 和病例数，再提交 `dimensions`。后续阶段依次为 `controls`、`reference`、`replicate`。需要按预设规则连续执行所有阶段时运行下列命令。提交端只轮询 Slurm 状态，各阶段验证结果落盘后才生成下一批配置。

```bash
bash scripts/submit_mri_haar_study.sh all
```

可用同一命令及原 STUDY_ID 恢复中断的提交。每个独立训练的 RUN_ID 由研究标识、方法和训练参数确定，已完成或仍在队列中的作业复用已记录的作业号。中断的训练从最后一个完整 epoch 载入模型、AdamW、最佳分数及早停计数。Slurm 进程退出会释放目录训练锁。每阶段的 job ID 和计划分别保存在 `reports/mri/submissions` 和研究目录。需要单独复现实验或统一使用四卡训练时可运行以下命令。

```bash
VARIANT=direct RUN_ID=x4_direct_a bash scripts/submit_mri_haar.sh
VARIANT=mls DIM=4 REFERENCE=geometry DIRECT_CHECKPOINT=runs/mri/x4_direct_a/bridge/best.pt \
  RUN_ID=x4_mls_f MRI_GPUS=4 bash scripts/submit_mri_haar.sh
```

## 消融与读出

| 阶段 | 比较 | 后续选择 |
| --- | --- | --- |
| backbone | A 单步、B 无端点损失积分、B 加端点损失的积分与干净端点 | A 的 VAL 最佳权重提供冻结检索编码器。B 的两种监督读出共用同一权重 |
| dimensions | C 均值；D PCA、E MLS 各取局部维数 2、4、6、8 | 分别选择 D、E 的最佳维数；各维同时保留 D 对 E 的匹配比较 |
| controls | 入围 E 的干净端点与噪声状态查询，各自配对路径优化开关；另做打乱关联 | 由验证结果在使用真实路由的模型之间选择 F 的母模型 |
| reference | 同一母模型下的单位阵、LR 条件协方差、同谱旋转 | 保留单位阵母模型、真实方向与旋转方向，一并复核 |
| replicate | 入围 A、B、C、D、E 和两种 F | 固定几何参数和读出，用 42、43、44 三个训练种子。按三个完整 VAL 的 pooled PSNR 均值排序，并列报告 SSIM |

A 在完整 16 带上预测 `HaarXY(HR-base)`，包括最低频。在原生 LR 的三个 XY 分辨率提取条件，桥支路接收同网格的状态、时间和方向。B 的样条导数和正反向漂移复用配对条件高斯路径。无几何时最小控制代价的均值路径是直线，方差采用 Brownian bridge，此配置直接使用其极小点；启用几何路径后再迭代优化内部结点。几何特征和门控分别进入干净端点头与漂移头，独立的自由残差预测保留端点自由度。

C/D/E 的训练库只接受相同 TRAIN 指纹的 A 权重，按病例和局部组织结构采样 LR 描述子与 16 带残差。检索时训练病例使用排除同折病例的图，VAL 使用完整 TRAIN 图。F 在每次采样开始从 LR 构造矩阵，所有时间步复用同一 Cholesky 因子。训练的控制代价、方差和漂移匹配处于白化坐标，曲面查询和逆 Haar 像素监督回到残差坐标。

学习率与早停以各有效配置为准，当前 A/B 模板的 epoch 上限为 160，连续 12 轮整病例 pooled MSE 未下降即结束。每轮输出 patch 验证与整病例 VAL，`best.pt` 按整病例 pooled MSE 最小保存，对应 pooled PSNR 最大。`runs/mri/$RUN_ID/bridge/whole_val/epoch_*.json` 保留病例记录，`metrics.jsonl` 记录局部有效支持、条件数、回退比例和路径修正。版本 4 权重内嵌检索编码器和 TRAIN 记录，可独立加载。原始库另存为 `spatial_bank.pt` 以便在相同来源下复用。

`outputs/mri/${RUN_ID}_val` 提供 VAL 指标和 CSV。B 的同权重积分臂保存到 `${RUN_ID}_integral_val`。研究计划只产生 VAL 结果。选定配置后按 `vemgsbm.mri_freeze` 的 freeze / confirm 步骤使用历史固定 TEST 清单，选择记录保留实际读出、噪声和采样步数。

## 分块开销

在本地 CPU 的 4 个线程、单个 5×32×32 原生 LR 块、48/96/128 通道、128 张图、3 个候选和 4 维 MLS 上，测得结合 LR 检索及协方差的 bind 与一次空间特征计算耗时如下。输入和图册是合成张量，预热 1 次后取 3 次中位数。

| 每个查询块的位置数 | bind（ms） | 特征（ms） |
| ---: | ---: | ---: |
| 128 | 91.4 | 31.6 |
| 512 | 62.7 | 17.4 |
| 1024 | 58.6 | 16.2 |
| 2048 | 53.9 | 17.0 |
| 5120 | 58.2 | 14.6 |

默认查询块大小取 2048。可用 `QUERY_CHUNK` 调整。GPU 门控作业会在真实 TRAIN 小批上同步计时，并把 rank 0 的 bind 和每次特征计算中位数写入 `smoke/smoke.json`。
`all` 会检查 TIMEOUT、PREEMPTED、NODE_FAIL 后的 last.pt。已有完整 epoch 且 checkpoint 持续前进时自动续投，默认每个训练最多 12 次，可用 `MAX_CONTINUATIONS` 指定次数。每次续投的作业号和 checkpoint 元数据记入 `continuations.tsv`。A 选定后在独立 A800 作业构造一次完整 TRAIN 记录库，研究目录的 `spatial_bank.pt` 供全部 C～F 的门控、正式训练和不同种子共享。各分折在同一记录库内拟合局部模型。
