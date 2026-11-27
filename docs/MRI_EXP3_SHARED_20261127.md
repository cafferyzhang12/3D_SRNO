# exp3 单次训练与同权重在线图册对照

> 最新状态（2026-11-27 23:15 北京时间）：用户要求优化效率后，5393973 已于18:19:13取消，5393974同时取消且从未启动。exp3最后日志为epoch1/batch512，只有TRAIN几何缓存，没有正式训练checkpoint或完整MRI验证结果。下文的提交/启动记录是历史记录。完整最新报告见 [算法与集群报告](MRI_ALGORITHM_AND_CLUSTER_REPORT_20261127.md)。exp1保持运行。

用户于2026-11-27授权exp3独立使用4×A800/high QOS，与正在运行的exp1并行；取消尚未启动的exp2–exp4旧任务，exp2暂缓、exp4不再单独训练。

已逐一校验5362249–5362254均为PENDING且RunTime=00:00:00，并仅取消这六项。exp1训练5362247和后续VAL5362248保留，原源码、运行配置和输出目录不改动。

新部署目录 `/gpfs/share/home/t2504041104/3D_SRNO_4/jetatlas_exp3_shared_20261127`。复制已通过完整尺寸4卡预检的exp3配置，仅更换run_dir；维持每卡batch48/全局192、60轮、相同学习率/种子/损失/训练图册。新任务排除exp1当前节点gpua800n04。

训练仍以offline整病例VAL选模。训练完成后选择best.pt（无合格best时明确回退last.pt），复制为一个冻结checkpoint；offline和online两个评估都加载这同一组EMA/model和router权重、同一病例清单、相同随机种子/步数/求解器。两个配置仅online_atlas开关不同，在线结果不再反向选择另一份权重。

评估产物包括冻结权重SHA256、训练epoch、选择方式、两组PSNR/SSIM/LPIPS/FID/局部方差/纹理SSIM以及online-minus-offline差值。固定TEST不参与。

本地已用实际合成PNG训练所得exp3 checkpoint跑通两组完整Hann四指标评估，输出同一SHA256；预检还将在新A800四卡分配内执行全量pytest及四进程同权重评估冒烟。训练代码与已通过预检版本逐字节比较。

入口：设置SHARED_STUDY_DIR、EXP1_PROJECT_ROOT、MRI_PERCEPTUAL_WEIGHTS、PROTECTED_EXP1_NODE后，运行 `bash scripts/submit_mri_exp3_shared.sh`。仅提交新预检→exp3训练一次→同权重双VAL，无exp1依赖、无exp2或exp4训练任务。

## 实际提交与保护核验

代码41bb23f已同步集群并推送main。预检5393972以COMPLETED/0:0完成，2分18秒；集群363项全部通过，实际合成PNG训练后使用同一个冻结checkpoint完成四进程offline/online四指标评估。

新exp3训练5393973已在gpua800n15运行（high、4×A800、fundiff），已观察到TRAIN准备601/3919，尚未宣称完成训练轮次。训练后双评估5393974等待afterok:5393973，输出目录paired_val_5393974。保持每卡48/全局192、60轮。

旧5362249–5362254均在尚未启动时取消；未重提exp2或exp4训练。exp1训练5362247仍在gpua800n04推进至epoch24/batch1152，后续VAL5362248仍保留。exp1源码SHA256 `3e0da1af78f0280a2b0ae8439b526fbbcc160d3ed1f48974274e4865eaa43448`、运行配置SHA256 `3fd9252e2c18201a256d00a80d077ea51dd2a3737ec647dd7882ad9c9141ac33` 操作前后相同。

详情见reports/mri/exp3_shared_submission_20261127.json、jobs TSV及GPU pytest XML。训练前冒烟结果仅证明同权重评估接口可运行，不是MRI收敛效果。
