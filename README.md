# Flat-Top Light GitHub Release

这个文件夹是从原始工程中整理出来的“适合上传到 GitHub 的精简版”，不会修改原工程内容。

## 目录说明

- `src/ai_holography/`
  - 当前主程序的核心实现，包括传播、损失、训练、hybrid 路线、runner 和 benchmark 相关代码。
- `src/lin2025_holography/`
  - Lin 路线的学习式初始化器实现，包括数据、编码、模型、训练和推理代码。
- `notebooks/`
  - 当前推荐的 AutoDL/Colab notebook：
  - `train_lin2025_autodl_lineflat_highres_export_v58.ipynb`
- `analysis/`
  - 本地分析脚本，例如 checkpoint 分析。
- `docs/`
  - 说明文档、详细报告、PPT 和讲稿 PDF。
- `legacy_references/ftl_gen/`
  - 历史参考程序和 795 相关原始参考文件，用于说明来源和对照。
- `assets/`
  - 当前整理时保留的关键参考相位数据文件。

## 这份整理版适合做什么

- 作为 GitHub 仓库的第一版公开整理目录
- 给老师、同学或合作者看核心代码和文档
- 保留 795 / Bowman / Lin / 当前混合路线之间的对照材料

## 这份整理版没有放什么

- 大量实验中间产物
- 大体积 checkpoint
- 临时 notebook 版本链
- 原工程中的所有杂项脚本

这样做的目的是让 GitHub 仓库更清晰，别人一眼能看懂主线，而不是被大量实验碎片淹没。

## 推荐上传内容

如果你想先做一个干净的 GitHub 仓库，建议直接上传这个文件夹的全部内容。

如果你想更轻量一些，最少保留这些目录即可：

- `src/`
- `notebooks/`
- `docs/`
- `legacy_references/ftl_gen/`
- `assets/`

## 上传说明

具体步骤见：

- `UPLOAD_TO_GITHUB.md`
