# XY×4 外部权重评估

两种模型都读取同一份真实原生 LR 的单张切片，Z 坐标保持不变。推理使用已发布的预训练权重。评估保留浮点预测，使用配对 PNG 的固定强度标定、整个 XY 视野以及与桥模型相同的 pooled PSNR 和逐切片 SSIM。导入程序将官方参数和本地数据契约存为独立的部署权重，记录原权重 SHA-256，并在加载时检查上游源码版本。

| 模型 | 训练来源 | 适配与计算量 |
|---|---|---|
| SwinIR-M classical ×4 | DF2K 自然图像 | 灰度复制到 RGB，使用官方窗口对称扩展，输出取三通道均值。每张 512×512 输出切片调用网络一次。[官方代码和权重](https://github.com/JingyunLiang/SwinIR/releases) |
| Res-SRDiff | 作者公开的前列腺 MRI 权重 | 单通道，将真实 LR 最近邻放大四倍作为同网格条件，将固定标定的 [0,1] 线性映射到模型输入的 [−1,1]，复用官方四步残差转移采样器。每张 512×512 输出切片调用网络四次。[官方代码和权重](https://github.com/mosaf/Res-SRDiff) |

512×512 以上的图像分窗，窗口重叠一半并作 Hann 融合。每个病例记录含图像读取、整例重建和指标计算的经过时间，写入 `per_case.csv` 的 `evaluation_seconds`。模型的方法名和预训练数据随 `summary.json` 保存。两种方法当前均为预训练迁移评估，未经本项目配对 MRI 微调。

## 单卡作业

在集群项目根目录准备两个上游仓库和两个权重。准备脚本只进行下载和版本检查。fundiff 需要与其 PyTorch 匹配的 torchvision，以及 timm 和 gdown。Res-SRDiff 发布包包含优化器和训练状态，原始下载约 4.19 GB。

```bash
export MRI_EXTERNAL_ROOT="$PWD/references"
export GSBM_PYTHON="$HOME/.conda/envs/fundiff/bin/python"
"$GSBM_PYTHON" -m pip install timm==0.6.13 gdown==5.2.0
bash scripts/prepare_mri_external.sh swinir
bash scripts/prepare_mri_external.sh res_srdiff
archive="$HOME/3D_SRNO_4/vem_gsbm_mls_v2_20261120_f08f30129d6a"
run="$archive/runs/mri/mri_x4_gsbm_mls_v2_20261120"
export PROTOCOL_CHECKPOINT="$run/bridge/best.pt"
export MRI_CASE_INVENTORY="$run/mri_case_inventory.json.gz"
mkdir -p reports/slurm
BASELINE=swinir RUN_ID="swinir_x4_$(date +%s)" sbatch --parsable scripts/mri_external_val.sbatch
BASELINE=res_srdiff RUN_ID="res_srdiff_x4_$(date +%s)" sbatch --parsable scripts/mri_external_val.sbatch
```

每个作业在单张 A800 上导入权重、完成一个整病例 VAL 冒烟测试、评价完整 VAL。用各自的部署权重和 VAL 输出调用项目 `mri_freeze` 与 `mri_test.sbatch` 可执行冻结确认。冻结和确认作业需能读取同一个 `MRI_EXTERNAL_ROOT`。比较输入为两个完整 `summary.json` 的路径，如下命令输出按病例成对重采样的 95% 区间。

```bash
"$GSBM_PYTHON" -m vemgsbm.mri_compare "$reference_summary" "$candidate_summary"
```

## 其他 MRI 方法的实现核查

CHARMS 论文报告了逐切片 ×4 模型，数据可用性段落称作者计划发布代码。此次核查未找到可以与论文对应的作者权重和实现。[CHARMS 原文](https://www.mdpi.com/1424-8220/26/2/738)

SR-DBM 论文描述了 MRI 配对条件和十步确定性采样。本次检查的论文入口未给出可核验的作者实现或权重，正式排名需要在本项目协议下完成重建。[SR-DBM 原文](https://arxiv.org/abs/2608.08819)
