# 配对 MRI 的集群实验

在项目目录提交。计算使用 `$HOME/.conda/envs/fundiff/bin/python`，由 `sbatch` 分配 A800 节点。每个正式实验用一张 A800，四卡 DDP 验证使用一个 Slurm task 和四个 torchrun 进程。两者都用 high QoS。登录节点负责源文件同步、依赖安装、提交和查询作业。

## 中断后恢复整病例评价

正式冻结确认和 `scripts/mri_whole_val.sh` 使用 `--resume-cases`。作业被抢占后，以相同权重、清单、种子、采样参数和输出目录重新提交。每个完整病例先写入私有目录，经输入与输出校验后原子提交到输出旁的 `.<输出目录名>.resume/cases/`。中断的病例从头重建，已提交且文件摘要匹配的病例直接复用。各 rank 可以在重启后重新分配病例。最终结果中的 `evaluation_identity.json` 和每例的 `case_commit.json` 记录评价协议和文件摘要。

保留输出旁的隐藏锁文件和恢复目录。Slurm 作业退出时内核会释放锁。修改权重、代码、输入像素或评价协议时请使用新的输出目录。较早作业的完整结果仍可直接读取，要生成新的恢复校验记录可另建目录重评。

## 历史验证和采样消融

从一个独立的源码部署目录提交。以下路径与作业 5281446 的快照对应。

```bash
export ARCHIVE_ROOT="$HOME/3D_SRNO_4"
export HISTORY_ROOT="$ARCHIVE_ROOT/vem_gsbm_mls_v2_20261120_f08f30129d6a"
export NEURAL_ATLAS_SRC="$HOME/Learning_Geometric_State_Spaces_for_Probabilistic_Trajectory_Inference/neural_atlas_lab/src"
export MRI_DATA_ROOT="$HOME/data/MRI_SR"
export MRI_CASE_INVENTORY="$HISTORY_ROOT/runs/mri/mri_x4_gsbm_mls_v2_20261120/mri_case_inventory.json.gz"
export RUN_ID="x4_stage1_$(date +%Y%m%d_%H%M%S)"
mkdir -p reports/slurm
HISTORY_GATE_JOB=$(MRI_GPUS=4 sbatch --parsable scripts/mri_history_gate.sbatch)
export HISTORY_GATE_JOB
bash scripts/submit_mri_stage1.sh
DEPENDENCY_JOB_ID="$HISTORY_GATE_JOB" bash scripts/submit_mri_historical_val.sh
```

`scripts/mri_history_gate.sbatch` 检查历史文件、固定清单、四进程训练和真实训练小批上的损失下降。`submit_mri_stage1.sh` 先做 pytest 和一个完整 VAL 病例的重建，再运行相同种子、相同末步策略的模型对照。保留相同 RUN_ID 重新调用，可恢复失败作业。每个作业的记录在 `reports/mri/submissions/`。作业号、退出码、相应输出目录的 `summary.json` 共同作为完成依据。

| 整病例 VAL 臂 | 步数 | 每窗漂移网络调用 | 推理噪声 |
|---|---:|---:|---:|
| plain，无在线引导 | 20 | 20 | 0.05 |
| guided，开启在线引导 | 20 | 20 | 0.05 |
| corrected，变扩散系数的漂移 | 20 | 40 | 0.05 |
| equal_calls，同网络调用预算 | 10 | 20 | 0.05 |
| reference，训练参考扩散系数 | 20 | 20 | 0.10 |

五臂由同一个第 38 轮权重生成。在 `paths.py` 的反向时间约定下，当前速度为 ((f-r)/2)，扩散项的调整给出 (0.625f-0.375r)。`diagnostics` 在成对的 Hann 融合、裁剪输出上计算引导误差分解和选图统计。`mri_stage1_replay.sbatch` 重放 guided 的实际逐步权重。两项诊断只读取 VAL。完成以后，将 `STAGE1_RESULTS` 设为 `outputs/mri/$RUN_ID` 的绝对路径，提交 `scripts/mri_stage1_review.sbatch`，生成 `stage1_decision.json` 和 `stage1_ranking.csv`。

## 开发、冻结和正式确认

1. 从 TRAIN 的真实配对窗口训练。旧模型复评已有权重。新模型按全部 VAL 病例的 pooled PSNR 选择 epoch 和配置，同时保留 SSIM、前景、边缘和 Z 向指标。主干、维数、负对照、参考协方差和多种子入口见 [MRI_HAAR.md](MRI_HAAR.md)。
2. 主研究入口为 `STUDY_ID=<新标识> bash scripts/submit_mri_haar_study.sh all`。默认每个实验一张卡，可设置 `MRI_GPUS=4` 复现四卡模式。控制脚本在提交端只查询 Slurm 和作业清单，在计算节点上检查、选择和训练。外部权重与输入适配见 [MRI_外部基线.md](MRI_外部基线.md)。
3. 单独冻结一种方法时，将已完成的同协议候选传给 `scripts/mri_freeze.sbatch`。冻结会核查每个 VAL 病例的指纹和 checkpoint 哈希，再把选定权重、清单、整病例 inventory、推理参数和选模依据复制到新的冻结目录。整项研究使用 `submit_mri_confirmation.sh`，先冻结整个候选矩阵，再放行 TEST 作业。

```bash
locked="$HISTORY_ROOT/outputs/mri/mri_x4_gsbm_mls_v2_20261120_test15/subset_manifest.json"
freeze_job=$(sbatch --parsable scripts/mri_freeze.sbatch \
    --candidate "$CHECKPOINT" "$VALIDATION_OUTPUT" --inventory "$MRI_CASE_INVENTORY" \
    --manifest "$locked" --folder "$FREEZE_DIR")
export FREEZE_DIR RUN_ID
sbatch --dependency="afterok:$freeze_job" --kill-on-invalid-dep=yes scripts/mri_test.sbatch
```

正式确认一律读取历史固定 15% 清单。TDM 的原配置保留训练时记录的 10% 默认值，确认入口显式传入 15%、种子 20261110 和固定清单，核验 84 例、14,991 张和 3,929,800,704 个体素。`mri_test.sbatch` 需要冻结目录；旧表征/桥流水线和 MLS 训练脚本结束于整病例 VAL。每次评价输出逐病例目录、`per_case.csv`、`summary.csv` 和 `summary.json`。配对差通过在两臂同时重抽病例计算，同一次抽样内重新汇总体素 MSE、PSNR 和切片 SSIM。

```bash
squeue -u "$USER"
sacct -X -j "$job" --format=JobID,State,ExitCode,Elapsed,NodeList
```
