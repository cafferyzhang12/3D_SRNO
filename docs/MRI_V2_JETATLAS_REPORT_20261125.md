# MRI v2：频带桥与 SC-JetAtlas 重构验收报告

日期：2026-11-25。基于 main `60fc2eb`；备份 `archive/pre-ssim-upgrade`，开发 `upgrade/jetatlas-freq-gsbm-sr`。

## 结论和证据范围

阶段1–4代码、四组同预算配置及A800四卡high QOS提交链已实现。真实MRI的SSIM≥0.95尚未验证；本地仅CPU与合成MRI，固定TEST未打开。单幅含噪MRI无法识别无条件物理SSIM上界，噪声工具输出的是带假设的估计。

## 基线、兼容性与源码审计

- 原始 `pytest -q` 因缺少可选neural_atlas无法收集；加入本机依赖源码路径后遇到OpenMP运行时异常。固定OMP/MKL/OpenBLAS线程并启用本机重复运行时兼容后，修改前265项全部通过（79.91秒）。当前工作区原有测试数量比任务所述264多1。
- 完整最终测试346 passed /78.33秒；包含球面2-jet的Itô径向偏离、正交特征坐标等变性与像素精修LL不变性回归测试。完整XML随报告保留。
- mri_metrics.py和mri_perceptual.py没有修改；7×7 uniform/sample-covariance SSIM、pooled SSE PSNR、AlexNet v0.1 LPIPS、Inception2048 float64分布式moments直接复用。
- mri_infer.py的Hann融合未修改；入场时已存在volume_predictor/tile-batch改动，作为必要适配保留并纳入提交，不是本轮改写Hann。其余入场时未跟踪的论文基线/文档保留在工作区。
- 旧SSIM只覆盖1/8样本、前缀stop-gradient，其他物理损失除以TRAIN频带方差后才加SSIM；实际贡献由梯度探针测量。
- 旧Haar特征在32→16→8网格计算，输出仍在LR Haar网格；新模型增加NAF、窗口注意力、跨Z注意力与PixelShuffle→HR NAF精修。
- 旧端点ODE没有score修正；新桥额外训练epsilon头，PF采用连续Gaussian bridge的条件流速度，低温SDE另加score修正。固定配对训练不等同于已求解无配对Schrödinger最优耦合。
- 旧图册1×1/16维，固定ridge、多处clamp，二次支持条件要求有效邻居数大于完整设计矩阵列数。新图册3×3 HF/135维→每fold TRAIN PCA、低秩法向2jet/GCV/soft边界。
- exact SSIM的float64五通道中间量与HR精修是显存热点；采用分片checkpoint、bf16及正式四arm实际显存探测。

## 合成上限摸底与真实四指标接口

6张64×64合成切片；以下不是VAL结果。

| 插值 | PSNR dB | SSIM | LPIPS | FID | 局部方差比 |
|---|---:|---:|---:|---:|---:|
| bicubic | 24.634880 | 0.77722177 | 0.22389216 | 164.793416 | 0.506520 |
| lanczos | 24.869056 | 0.79758463 | 0.22711575 | 142.117901 | 0.524358 |

噪声估计的每病例背景/均匀组织sigma、加性独立噪声假设下的oracle SSIM估计、各损失梯度占比均在原始ceiling JSON。合成SSIM加权梯度占比68.68%，局部标准差4.47%。本地没有历史x1权重/真实VAL挂载，旧模型梯度与硬裁剪比例明确标为unavailable，集群gate提供原权重与匹配inventory后实测。

## 四组20步实际优化

相同微型骨干、可从LR恢复对比度的3×8×8→3×32×32合成纹理。每组实际V2Objective优化20次；不是解析oracle，也不是模型泛化结果。

| 实验 | 20步SSIM | PSNR dB | 方差比 | LPIPS | FID |
|---|---:|---:|---:|---:|---:|
| exp1 | 0.99227080 | 37.034775 | 1.086672 | 0.00157741 | 119.527810 |
| exp2 | 0.99143229 | 36.585615 | 1.096212 | 0.00169404 | 123.340646 |
| exp3 | 0.99250166 | 37.324510 | 1.087150 | 0.00143201 | 112.440623 |
| exp4 | 0.99250166 | 37.324510 | 1.087150 | 0.00142881 | 113.606421 |

该表过拟合曲线末点是单步训练读出；感知项来自8步部署读出（exp4含在线收缩），不可把两者当作同一图像。完整逐步曲线见JSON。只有3张合成切片，FID虽由真实2048模型计算，但不能用于效果归因。方差比约1.09显示已有轻微过冲，真实训练需通过VAL控制。

## 步数稳定性（20步训练后PF-ODE）

| 实验 | 步数 | PSNR dB | SSIM | 方差比 | 有限 |
|---|---:|---:|---:|---:|---|
| exp1 | 1 | 37.034775 | 0.99227080 | 1.086672 | True |
| exp1 | 4 | 37.034775 | 0.99227080 | 1.086672 | True |
| exp1 | 8 | 37.034775 | 0.99227080 | 1.086672 | True |
| exp1 | 16 | 37.034775 | 0.99227080 | 1.086672 | True |
| exp2 | 1 | 36.585615 | 0.99143229 | 1.096212 | True |
| exp2 | 4 | 36.643692 | 0.99162553 | 1.096043 | True |
| exp2 | 8 | 36.623637 | 0.99155323 | 1.096279 | True |
| exp2 | 16 | 36.610279 | 0.99149965 | 1.096435 | True |
| exp3 | 1 | 37.324510 | 0.99250166 | 1.087150 | True |
| exp3 | 4 | 37.398146 | 0.99263387 | 1.087261 | True |
| exp3 | 8 | 37.397351 | 0.99266478 | 1.087398 | True |
| exp3 | 16 | 37.394174 | 0.99267150 | 1.087489 | True |
| exp4 | 1 | 37.328031 | 0.99250120 | 1.087191 | True |
| exp4 | 4 | 37.402145 | 0.99263590 | 1.087300 | True |
| exp4 | 8 | 37.401356 | 0.99266742 | 1.087438 | True |
| exp4 | 16 | 37.398171 | 0.99267438 | 1.087534 | True |

所有ODE/SDE 1/4/8/16步均无NaN/Inf。exp2 4→16步SSIM有约0.000126小幅下降，不能宣称严格单调；原29.60dB式崩溃在该合成验证中未出现。解析双模态后验测试另验证受控SDE/随机内部初态PF可恢复方差，但确定性零初态PF不能凭空生成多模态。

## 二阶几何与防泄漏实测

- J/Hessian autograd最大误差0；H中心差分误差1.80e-8；PT幂等误差7.77e-16。
- soft边界最小导数约1.0e-6，有限测试范围内非零；极远处浮点饱和仍可能使梯度极小。
- 合成支持认证100%；真实TRAIN支持率须实测，不能预先保证>95%。
- 修改某病例全部bank特征后，该病例被排除fold的PCA和图卡buffers完全不变。路由器/在线EB API不接收HR。
- TRAIN bank、白化、PCA、crossfit/router教师均仅从TRAIN建立；geometry缓存校验病例身份/基底/白化/patch/seed/atlas settings；v2运行inventory剔除TEST元数据，拒绝TEST配置与VAL/TEST拟合。
- 主干梯度：SSIM .40976、local_std .06026、Jet-asym .03890、Jet势能 .002793；routerKL对router .18845，对主模型0符合detach设计。
- Itô曲率使用stride4稀疏几何场后上采样以控制吞吐；在线EB逐点并默认仅最终步。
- 推理法向收缩在PCA特征图卡空间成立；重叠邻域逆映射与HR精修后，不宣称全像素空间切向完全不变。

## 端到端磁盘验收与显存

四组均生成真实PNG：5个TRAIN病例、1个VAL病例；完成TRAIN白化/五折图册、1轮2updates、format6保存、重新加载、完整3slice/3072pixels Hann推理、真实四指标评估，逐项复现。详见v2_e2e.json。CPU没有GPU，显存项如实为null。

集群gate将对完整7×64²LR→7×256²HR、64/128/192宽度、bf16+checkpoint的四组运行实际前后向及optimizer一步，测试batch8/16/32/48/64，按四组最重实测峰值≤82%显存统一批量与累积，保证全局128–256。不凭CPU推算A800显存。

## 运行中发现并修复的问题

1. Python依赖与OpenMP基线环境异常：补依赖路径、固定线程，取得绿色基线。
2. 把无网络图的teacher Xt用于VJet会令主干势能梯度为零：改为模型生成Xt并实测非零梯度。
3. bf16进入QR/cdist/Gram求解存在精度/算子风险：几何内部关闭autocast并转float32，保留反传。
4. 旧geometry缓存只比较病例digest，可能跨基底/参数复用：加入完整geometry contract。
5. SDE推理若复用旧固定sigma校验会拒绝低温频带扩散：v2单独处理名义扩散幅度，保留旧模型规则。
6. 大图全点曲率查询吞吐过高：仅Itô曲率stride4，EB保持逐点且默认最终步。
7. 本地无GPU不能验显存：新增真实GPU校准gate，未通过不放行训练。
8. 无候选通过宽松PSNR保护下限时可能没有best.pt：最终评估明确回退last.pt，不将其标为合格最优模型。
9. HR精修直接依赖随机HF会使最终像素LL重新变随机：将HR精修条件改为LR特征+确定性LL参考，同时保留完整像素精修能力；简单删除精修LL虽保证确定性，但20步过拟合降至约0.963，已用上述确定性条件设计解决，重新通过所有四组20步门限。
10. 正式run_id、inventory非空与split、旧format checkpoint等都加fail-closed校验。

## 集群入口与预算

四份configs/mri_v2_exp*.yaml保持相同60轮、262144 patches/轮、lr2.5e-4、3轮warmup+cosine、bf16+checkpoint、同种子/同VAL；exp1/2 atlas_weight严格0。新默认7张上下文/64²LR；轻量可将所有运行YAML patch_size同步改为[5,256,256]并重建匹配inventory。每轮整病例VAL自动PSNR/SSIM/LPIPS/FID/局部方差/纹理区SSIM。SSIM主导，PSNR仅保护baseline-0.5dB。

```bash
export RUN_ID=mri_v2_20261125
export MRI_CASE_INVENTORY=/path/to/audited_inventory.json.gz
export NEURAL_ATLAS_SRC=/path/to/neural_atlas_lab/src
export MRI_PERCEPTUAL_WEIGHTS=/path/to/verified/perceptual
export MRI_LEGACY_CHECKPOINT=/path/to/old_x1_best.pt
bash scripts/submit_mri_v2_ablation.sh
```

提交链：四卡gate→exp1四卡train→VAL→exp2→VAL→exp3→VAL→exp4→VAL；全部high QOS，fundiff，A800分区，afterok依赖，登录节点仅同步/提交。固定TEST不提交。实际作业号及同步commit见提交记录。

## 实际部署与提交回执

代码提交 `d75ec93` 已推送 GitHub main；source.tar.gz SHA256 `8e95697cba7f7125ca97698f1c6afd268898df684a6665411fd3fe6859388a9e` 本地与集群一致。部署 `/gpfs/share/home/t2504041104/3D_SRNO_4/jetatlas_freq_v2_20261125`。neural_atlas 使用本机基线通过的源码快照，避免其他工程改动影响本任务。

北京时间2026-11-25 20:05核验：gate **5345082** 已在 **gpua800n24** 运行；4×A800、high QOS、fundiff。训练/VAL依赖任务依次为：exp1 **5345083/5345084**，exp2 **5345085/5345086**，exp3 **5345087/5345088**，exp4 **5345089/5345090**。九个任务全部实际核验为high、gres:gpu:4。原始回执v2_cluster_jobs.tsv，结构化记录v2_cluster_submission.json。

当前只证明提交与gate启动，正式GPU显存、真实VAL天花板、训练改善和最终SSIM仍等待任务产出。依赖afterok会在预检失败时阻止重训练。按用户要求不持续监督长训练。
