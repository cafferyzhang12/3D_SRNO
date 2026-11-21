# 独立 GSBM 项目的集群运行

从仓库根目录提交。脚本默认使用 `$HOME/.conda/envs/fundiff/bin/python`。
`GSBM_PROJECT_ROOT` 和 `GSBM_PYTHON` 可分别指定仓库和解释器。
将 [MRI 模板](../configs/mri_x4.yaml) 复制到 `configs/mri_x4.local.yaml`，填好
三个数据划分的 LR/HQ 路径，再通过 `CONFIG` 传入。

## 作业与资源

所有图像读取、预训练、度量拟合、数值检查、神经漂移匹配和推理均在
`sbatch` 分配的 A800 计算节点执行。登录节点只进行源文件同步、依赖安装、
路径检查、作业提交和日志检查。每个计算入口要求四张 Slurm 分配的 A800，
使用单节点、单 Slurm task，由 `torch.distributed.run` 启动四个 rank。
QOS 在脚本和提交命令中均显式设置为 `high`。

在项目目录提交：

```bash
CONFIG=configs/mri_x4.local.yaml RUN_ID=mri_x4_gsbm_example bash scripts/submit_mri_pipeline.sh
```

流水线通过 `afterok` 依次连接四个作业：

1. `validate`：在四卡 allocation 内执行项目测试；真实 MRI 的小规模
   表征训练、bridge matching 训练和单个验证病例重建。
2. `representation`：正式三维自监督表征训练和训练 HR 特征库的度量拟合。
3. `bridge`：固定表征和度量，执行条件路径优化与双向神经漂移匹配。
4. `test`：最佳 bridge checkpoint 的固定 15% 独立测试。

只有前一作业成功退出，后一作业才可开始。设置
`--kill-on-invalid-dep=yes`，失败的上游不会导致依赖作业误运行。
提交成功只表示作业已进入调度队列，不表示四卡验证或正式训练已经通过。
请以 Slurm 状态与作业退出码、日志和最终产物为准。

## GSBM 与动量桥的 MLS 对照

在源码目录设置 `MRI_DATA_ROOT` 为原配对 MRI 数据根目录，设置
`SUBSET_MANIFEST` 为已有 15% 测试病例清单的绝对路径。通过
`NEURAL_ATLAS_SRC` 指定集群上与实验相同版本的 `neural_atlas_lab/src`，
也可使用 `fundiff` 中已安装的对应版本。为每次训练设置独立的 `RUN_ID`。

```bash
mkdir -p reports/slurm
CONFIG=configs/mri_gsbm_mls_guided_x4.yaml RUN_ID=mri_x4_gsbm_mls_trial \
  sbatch --parsable --qos=high scripts/mri_atlas.sbatch
CONFIG=configs/mri_momentum_mls_x4.yaml RUN_ID=mri_x4_momentum_mls_trial \
  sbatch --parsable --qos=high scripts/mri_atlas.sbatch
```

每条作业在同一四卡 A800 allocation 内执行验证集小规模重建、正式表征和桥训练、
固定 15% 测试。每次审计生成的紧凑病例索引供四个 rank 共享，并由该作业的表征训练、
桥训练和正式测试使用。测试结果分别写入
`outputs/mri/<RUN_ID>_test15/summary.json`。

## 正式配置

`configs/mri_x4.yaml` 保存可审查的默认值：表征预训练 10 个 epoch，bridge
训练 40 个 epoch，每卡 batch 4、四卡全局 batch 16，每个 epoch 全局抽取
4096 个三维窗口、固定验证 64 个窗口。HR 窗口为 `[5,128,128]`，输入真实
LR 为 `[5,32,32]`。这些 epoch 是窗口预算，不是完整读取全部体素一次。

表征宽度 8、特征维度 32、训练特征库最多 1024 个样本、32 个局部中心。
漂移网络宽度 16。条件路径有 5 个节点，每个 batch 执行 5 次内部优化，
5 个积分采样点，同时优化均值和有界方差样条（std_bound=0.75）；参考噪声 sigma 0.1，state 与 metric 项权重均为 0.01，
均值修正限制为 0.25。最终采样默认 20 步。

参数是可运行的起始研究配置，不能预先声称达到最佳精度或吞吐量。
验证门同时检查数学模块和工程接口；它不证明正式实验的质量或跨模态性能。
`BATCH_SIZE`、`REPRESENTATION_BATCH_SIZE`、`REPRESENTATION_EPOCHS`、
`BRIDGE_EPOCHS`、`SAMPLES_PER_EPOCH`、`VAL_SAMPLES_PER_EPOCH`
可在提交前覆盖；运行时生成的有效 YAML 保存在 `reports/mri/run_configs/`。
改变算力预算进行比较时，应同时记录优化步数、路径内部优化次数和采样步数。

## 验证门的隔离

验证使用单独的 `${RUN_ID}_gate_${SLURM_JOB_ID}` 目录，不写入正式权重。
它使用两个病例的索引上限、每阶段 1 epoch、全局 8 个训练窗口和 4 个
验证窗口、每卡 batch 1。缩小后的表征/漂移宽度均为 4，特征维度 8，
路径内部优化两次。最后仅对一个**验证**病例做两步重建，不接触测试成绩。
正式训练的模型和窗口预算来自正式有效配置。

## 15% 测试与输出约定

测试入口固定传入 `--test-fraction 0.15 --subset-seed 20261110`。在对完整
候选病例排序和校验后，只选择一次全局病例子集，再将病例分给四个 rank。
当前 560 个病例对应 84 个完整病例，不是各 rank 独立抽样，也不是仅取
前 84 个自然排序病例。病例清单与种子保存为 `subset_manifest.json`，
并在输出目录旁保留清单，以便重跑和对照实验核对。

原 MRI 的 `case/slice.png` 配对方式、逐层 XY 插值、原始 Z 位置、强度
归一化、完整切片评估与重叠三维窗口融合方式保持对应接口。每个病例
输出 `metadata.json` 和 `metrics.json`，根目录成功发布 `summary.json`。
默认前 8 个被选病例保留 `mean.npy`、`variance.npy`，其他被选病例仍完成
全部切片评价。`EXPORT_CASE_LIMIT` 只影响数组导出数量，不影响 15% 评价范围。
单次采样的 variance 为零，不能解释为已校准的不确定性。

训练权重分别位于 `runs/mri/<RUN_ID>/representation/` 和
`runs/mri/<RUN_ID>/bridge/`。bridge checkpoint 为兼容推理接口记录
`stage=reconstruction`，同时保留 `training_stage=bridge` 和独立算法标识。

Slurm 日志在 `reports/slurm/`，提交记录在 `reports/mri/submissions/`，
流水线记录在 `reports/mri/pipelines/<RUN_ID>.txt`。

```bash
squeue -u "$USER"
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed,NodeList
```

数据和模态迁移不自动等于零样本泛化。适配电镜时需要在其训练体上学习或
重新拟合表征/几何，并验证其观测算子；无需指定脑组织分割、骨架或树拓扑。
