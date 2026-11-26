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
