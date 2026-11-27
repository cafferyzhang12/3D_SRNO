# exp3 单次训练与同权重在线图册对照

用户于2026-11-27授权exp3独立使用4×A800/high QOS，与正在运行的exp1并行；取消尚未启动的exp2–exp4旧任务，exp2暂缓、exp4不再单独训练。

已逐一校验5362249–5362254均为PENDING且RunTime=00:00:00，并仅取消这六项。exp1训练5362247和后续VAL5362248保留，原源码、运行配置和输出目录不改动。

新部署目录 `/gpfs/share/home/t2504041104/3D_SRNO_4/jetatlas_exp3_shared_20261127`。复制已通过完整尺寸4卡预检的exp3配置，仅更换run_dir；维持每卡batch48/全局192、60轮、相同学习率/种子/损失/训练图册。新任务排除exp1当前节点gpua800n04。

训练仍以offline整病例VAL选模。训练完成后选择best.pt（无合格best时明确回退last.pt），复制为一个冻结checkpoint；offline和online两个评估都加载这同一组EMA/model和router权重、同一病例清单、相同随机种子/步数/求解器。两个配置仅online_atlas开关不同，在线结果不再反向选择另一份权重。

评估产物包括冻结权重SHA256、训练epoch、选择方式、两组PSNR/SSIM/LPIPS/FID/局部方差/纹理SSIM以及online-minus-offline差值。固定TEST不参与。

本地已用实际合成PNG训练所得exp3 checkpoint跑通两组完整Hann四指标评估，输出同一SHA256；预检还将在新A800四卡分配内执行全量pytest及四进程同权重评估冒烟。训练代码与已通过预检版本逐字节比较。

入口：设置SHARED_STUDY_DIR、EXP1_PROJECT_ROOT、MRI_PERCEPTUAL_WEIGHTS、PROTECTED_EXP1_NODE后，运行 `bash scripts/submit_mri_exp3_shared.sh`。仅提交新预检→exp3训练一次→同权重双VAL，无exp1依赖、无exp2或exp4训练任务。
