# 源码发布准备

从仓库根目录生成源码压缩包。

```text
python -B tools/package_source.py
```

输出为 `dist/vem_gsbm-source.tar.gz`。包中带有 `SOURCE_SHA256.txt`，在解压目录可用 `sha256sum -c SOURCE_SHA256.txt` 核对。发布包包含配对 MRI 超分的源码、运行配置、测试、方法文档和精选实验汇总。MLS 图册依赖单独发布的 `neural-atlas-lab` 包。

MRI 原始样本、病例清单、逐病例结果、训练权重和工作缓存留在本地目录。

公开上传前，由仓库权利人确定项目级授权。上游 GSBM 与 MFM 的来源和授权分别记录，发布时随附适用的许可和署名文本。 [第三方来源与许可证](THIRD_PARTY.md)
