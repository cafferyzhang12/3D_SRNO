# 3D MRI 超分辨率

将 MLS 图册提供的几何用于配对 LR→HR 的 MRI 重建。支持配对漂移匹配、动量桥，以及由真实 HR 监督的确定性端点迭代重建。

## 安装

从本目录运行。先从几何学习仓库安装共享图册包，再安装超分项目。

```powershell
python -m pip install -e "<geometry-repository>/neural_atlas_lab"
python -m pip install -e ".[atlas,test]"
python -m pytest -q
```

集群的 `fundiff` 环境也需要安装同版本图册包。可以在登录节点部署源码后安装对应路径，或先构建 `neural_atlas_lab` wheel，再在集群环境中安装该 wheel。

## 运行

- [配对 SR 与二阶 MLS](docs/MRI_SR_MLS.md)说明当前重建目标：从部署起点及模型自身轨迹监督 HR，使用原始强度单位的像素、高频、边缘与可微曲面约束；包含四卡 A800、high QoS 的训练及固定 TEST 入口。

- [单向 x1 Bridge Matching](docs/MRI_X1_BRIDGE.md)说明新版白化、无噪声端点 ODE、EMA、训练期 MLS，以及四卡训练后自动验证的运行命令。

- [MRI MLS 桥](docs/MRI_MLS.md)说明训练 HR 如何构建图册、配置和推断入口。
- [集群命令](docs/CLUSTER.md)说明数据路径、表征训练、桥训练和留出评估。
- [动量桥](docs/MOMENTUM.md)说明加速度匹配。
- [历史配对 MRI 汇总](reports/mri/summary.json)记录同一批 84 个病例的 endpoint-safe 和动量桥结果。

配对病例清单和本地数据放在 `data/private/`。源码发布入口见 [发布说明](docs/RELEASE.md)。
