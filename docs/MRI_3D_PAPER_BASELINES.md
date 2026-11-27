# 3D EDM 和 MedGSSR 的配对 MRI 基线

> 2026-11-27更新：3D-EDM作业5344286已成功完成最终VAL/TEST，MedGSSR作业5344315正在第2轮训练。聚合结果与数据范围详见 [当前算法与集群报告](MRI_ALGORITHM_AND_CLUSTER_REPORT_20261127.md)。以下为原始协议与部署记录。

对照任务使用项目现有的配对 MRI PNG 数据。输入为原生 LR，倍率为 XY ×4、Z ×1。训练体块为 5×32×32 LR 和 5×128×128 HR，固定 3,919 例 TRAIN、10 例完整 VAL 和 84 例历史 TEST。整病例使用主实验的滑窗、Hann 融合和融合后裁剪。PSNR 由全病例体素合并的 MSE 计算，SSIM 和 LPIPS 按切片等权，FID 在同一队列上汇总 Inception 特征后计算。

| 协议项 | 基线执行方式 |
|---|---|
| 每轮训练样本 | 262,144，与活跃主实验一致 |
| 等效 batch | 单卡梯度累积到 2,176，尾批保留，每轮 121 次更新 |
| 训练轮数与随机种子 | 8 轮，种子 42，复用主实验的体块抽样和 shuffle |
| 优化与初始化 | 两个基线各自从头训练并保留方法的损失和优化器。主实验此阶段从第 22 轮 EMA 检查点微调 |
| 验证与选模 | 每轮重建完整 VAL，以 pooled PSNR 为主、切片 SSIM 为平局次序。主实验微调使用针对其初始检查点的 PSNR 下限。基线完成选模后记录 checkpoint、配置、清单和源码哈希，再评估锁定的 TEST |
| 调度 | 每个正式作业在一张 A800 上顺序完成训练、最终 VAL 和 TEST，使用 fundiff 和 high QoS |

## 官方源码与适配

- **3D EDM** 固定官方提交 d9d82789f1b182cf05f7b654e77bf30a71a57250。使用上游 50.7M 参数 3D U-Net、EDM 噪声条件与预条件 MSE、AdamW 和 20 步 Euler 采样。把上游 XY ×2 改为 XY ×4，保留层间形状处理，输入强度用可逆线性变换映射到 [-1,1]。上游补丁仅延后无关训练日志及上游 LPIPS 的导入。[论文](https://arxiv.org/abs/2603.14667v2) · [上游源码](https://github.com/Magic-Solutions/MRI-Super-Resolution/tree/d9d82789f1b182cf05f7b654e77bf30a71a57250)
- **MedGSSR** 固定官方提交 77c9f1ccdbe5215d0163f22ccbe1d615be08c4ed。使用公开编码器、两分辨率投影与数值偏置，保留官方代码中粗细分支复用投影权重的实现。按论文式 (8) 使用体素 L1 和 Adam，学习率由 1e-4 对数下降到 1e-5，覆盖本对照预算。官方训练脚本和权重未发布，训练入口依据论文补写。[论文](https://arxiv.org/html/2609.06874v1) · [上游源码](https://github.com/Azusa309/MedGSSR/tree/77c9f1ccdbe5215d0163f22ccbe1d615be08c4ed)
- MedGSSR 适配为非立方坐标网格。从上游坐标中心和可学习高斯构造出发，按网格的三个轴限制偏移。修复公开渲染器的构造函数，以及 CUDA 快速路径中的条件线程同步和前反向支持域不一致的问题。CUDA 用论文式 (7) 的三倍 Mahalanobis 半径和一致的梯度，在 A800 测试时与稠密 PyTorch 公式及自动微分逐项比对。

本地上游克隆位于项目的 references/3D-EDM 和 references/MedGSSR。补丁保存在 patches/mri-paper-baselines，直接应用于上述提交。正式作业由 scripts/mri_paper_baseline.sbatch 启动，产物位于集群部署目录 runs/mri 下的方法子目录。final-test/summary.json 包含固定 TEST 的指标与病例记录。

2026-11-25 部署到集群的 ~/3D_SRNO_4/paper_baselines_xy4_20261125。单卡 A800 冒烟作业 5344259 同时通过了两个模型的真实配对体块训练和推理检查。3D EDM 正式作业为 5344286，MedGSSR 为 5344315。MedGSSR 等待 3D EDM 作业结束后才具备调度资格。提交回执保存在部署目录，作业输出在 reports/slurm。
