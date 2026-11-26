# MRI v2 CUDA 启动失败修复记录

本轮针对作业5345082的完整尺寸显存预检失败修复。原正式训练和VAL依赖任务均未启动；没有将合成结果当作真实MRI结果。

## 定位证据

诊断作业5362197，high QOS、4×A800、fundiff、gpua800n04，开启CUDA_LAUNCH_BLOCKING=1后复现。失败的跨Z注意力SDPA输入为 `[131072,4,7,16]`，dtype bf16。131072来自每卡batch32×64×64个独立空间位置。CUDA报invalid configuration argument，发生于SDPA调用，非显存OOM。

## 针对性修复

- 窗口注意力与跨Z注意力统一沿独立序列batch分块，每次最多16384；不切分序列内token/head，不裁剪输入，不降低视野和模型宽度。
- 数值回归校验分块前后的输出、q/k/v梯度一致，保留非整除尾块；GPU直接验证原始131072批量的float32/bf16前后向。
- 显存校准异常报告新增forward/backward/optimizer阶段、完整traceback和PyTorch/CUDA版本；已覆盖的较小成功batch不再重复测试。
- 预检新增四rank实际DDP更新与参数同步检查，覆盖四组正式7×256² HR配置；预检未通过时依赖任务不会运行。
- 正式训练首batch即记录loss与显存，便于确认启动，不必等待32批。
- PSNR/SSIM/LPIPS/FID协议、Hann融合与TRAIN/VAL/TEST隔离规则未修改。

## 已完成验证

修改前本地346 passed；修改后本地354 passed、2 GPU checks skipped（本机仅CPU）。A800作业5362215以COMPLETED/0:0结束：10个注意力测试全部通过，包含两项GPU回归；完整batch32 exp1前后向与Adam/EMA一步通过。loss=0.0074013775，梯度范数0.0429142，显存allocated峰值约28.1 GiB，reserved峰值约34.1 GiB，torch2.8.0+cu128。

本轮尚需完成统一显存校准、四rank更新预检，再确认正式四卡任务启动。后续作业号与运行参数追加于此；不在登录节点运行计算。

## 统一预检结果

正式源码a00a03b已同步并推送main；集群release SHA256 `d3d5651c5f4701aa2adddd78fdd71be4f11fc14b4d3be6f34afc27d67aa0c6d7` 与本地一致。部署目录 `/gpfs/share/home/t2504041104/3D_SRNO_4/jetatlas_freq_v2_cuda_fix_20261126`。

预检5362246在gpua800n04以COMPLETED/0:0结束，耗时8分51秒：

- 集群全量356 passed（含GPU回归），57.25秒。
- 四组batch32和48全尺寸前后向、Adam/EMA一步均成功；选中每卡48、累积1、全局192。
- batch48四组reserved峰值约50.99/57.55/58.14/58.14 GiB；batch64仅预测将越安全余量，明确标为未实测，不伪造显存值。
- 四rank DDP四组各2次更新通过，所有rank参数同步；峰值reserved约50.99/57.58/57.83/57.83 GiB。
- 原始真实VAL 10病例595切片的Bicubic/Lanczos四指标诊断成功；固定TEST未打开。此为插值诊断，不是新网络训练后的结果。
- 旧模型TRAIN探针512sites硬裁剪比例77.54%，二次支持率72.50%；各损失真实梯度已保存至去病例标识汇总JSON。

重新提交的任务均为high QOS、4×A800、fundiff：

| 实验 | 训练 | 整病例VAL |
|---|---:|---:|
| exp1 强回归 | 5362247 | 5362248 |
| exp2 频带桥无图册 | 5362249 | 5362250 |
| exp3 训练图册 | 5362251 | 5362252 |
| exp4 在线图册 | 5362253 | 5362254 |

预检通过后exp1已在gpua800n04进入3919 TRAIN病例统计准备，数据集报告无skipped cases。后续按afterok串行放行，保持同预算，固定TEST未提交。

## 真实训练首批核验

3919个TRAIN病例准备完成后，5362247在gpua800n04实际完成epoch1/batch1：loss **0.0659493734**，GPU allocated峰值 **41.846 GiB**。该日志在loss.backward、有限梯度裁剪、optimizer.step和EMA更新之后产生，因此验证了真实数据首个优化更新。其余三组和四组VAL保持high/4GPU依赖队列，不持续监督长训练。

训练日志仅有torchvision弃用、NCCL设备推断及张量转标量提示，未出现CUDA异常或非有限梯度错误。该首批记录不是收敛或最终VAL效果证明。
