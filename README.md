# Flat-Top Light: Codex-Assisted Neural Network Holography Project

本仓库用于归档我使用 **Codex** 辅助构建和优化的神经网络全息光场生成项目。项目目标是构建一个面向 **线型平顶光场生成（line flat-top light-field generation）** 的神经网络训练与评估流程，并通过 AI Coding Agent 提高代码开发、训练调试和实验迭代效率。

This repository archives my Codex-assisted neural network project for line flat-top holographic light-field generation and optimization.

---

## 项目背景 / Project Background

线型平顶全息光场生成涉及多个复杂环节，包括参考相位生成、神经网络初始化、物理传播模拟、混合优化后处理、多路线 benchmark 对比和结果导出。

这类项目的核心痛点是链路长、参数多、调试成本高。一次实验通常需要反复修改代码、运行训练、检查日志、分析结果并继续优化。

Line flat-top holography involves multiple tightly coupled components, including reference phase generation, neural network initialization, physical propagation simulation, hybrid optimization, benchmark comparison, and result export.

The main challenge is that the workflow is long and error-prone. A single experiment may require repeated code modification, training runs, log inspection, benchmark comparison, and parameter adjustment.

---

## Codex / AI Agent 的作用

在该项目中，我使用 **Codex** 作为 AI Coding Agent，辅助完成：

- 神经网络代码理解与重构
- 训练循环和 loss 配置检查
- AutoDL / Jupyter notebook 流程整理
- 运行报错和日志分析
- benchmark 路由设计与结果对比
- checkpoint、summary 和 plots 等结果导出流程优化

Codex is used as an AI coding agent to assist with:

- Neural network code analysis and refactoring
- Training loop debugging and optimization
- Loss function and benchmark logic review
- AutoDL / Jupyter notebook workflow organization
- Multi-route experiment comparison
- Result export and visualization

---

## AI Coding Agent Workflow

该流程不是单次问答，而是一个多轮闭环：

```text
代码 / notebook / 日志输入
→ Codex 分析模型结构和训练逻辑
→ 生成修改建议或代码优化方案
→ 在 AutoDL / Jupyter 中运行训练
→ 分析 loss、benchmark 和导出结果
→ 继续反馈给 Codex 进行下一轮优化
```

This is not a single prompt-response use case. It is a multi-round AI coding workflow that connects code understanding, training/debugging, benchmark comparison, result analysis, and further optimization.

General workflow:

1. Define the holography optimization goal and experiment constraints.
2. Provide Codex with the notebook, code snippets, runtime logs, or error messages.
3. Let Codex analyze model structure, data flow, training loop, loss configuration, and benchmark logic.
4. Apply the suggested code changes or parameter adjustments.
5. Run the experiment in AutoDL / Jupyter.
6. Feed logs, plots, benchmark results, or errors back to Codex.
7. Iterate until the training and evaluation pipeline becomes stable.
8. Export checkpoints, summary files, plots, and versioned project bundles.

---

## 仓库结构 / Repository Structure

后续我可能会继续上传不同版本的完整项目，因此本仓库采用版本文件夹归档方式。

Each version folder contains a relatively complete project snapshot.

```text
flat-top-light/
├── README.md
├── github_release_20260418/
│   ├── core notebook / training files
│   ├── results / summary files
│   ├── plots / screenshots
│   └── project evidence
└── future_version_folders/
```

---

## 版本记录 / Version History

| Version Folder | 内容说明 / Description | Status |
|---|---|---|
| `github_release_20260418` | 当前上传的第一版完整项目归档，包含 Codex 辅助神经网络全息光场优化的 notebook、截图、结果文件和项目说明。Initial public project archive for the Codex-assisted line flat-top holography training workflow. | Uploaded |
| Future versions | 后续会继续上传改进后的训练流程、benchmark 结果和可视化材料。Later experiment snapshots with improved training logic, benchmark routes, or result visualization. | Planned |

---

## 当前项目成果 / Current Highlight

当前版本已经形成一个相对完整的 AutoDL / Jupyter 工作流，用于神经网络线型平顶全息光场生成实验。项目材料包括：

- 主 notebook
- 神经网络训练流程
- 参考相位生成与优化流程
- 多路线 benchmark 对比
- 结果导出文件
- Codex 辅助优化过程截图
- 运行或结果证明材料

The current version demonstrates a neural-network-based workflow for line flat-top holography, including:

- Reference phase generation
- Neural network initialization
- Training workflow in AutoDL / Jupyter
- Benchmark route comparison
- Result export
- Codex-assisted code analysis and optimization evidence

---

## 用于申请的说明 / Application Evidence

本仓库主要作为 AI 创作者激励 / token plan 申请的项目证明材料。

它用于证明我已经将 Codex 应用于真实神经网络研发流程中，包括代码理解、问题定位、训练优化、benchmark 对比和结果导出。虽然项目仍处于早期整理阶段，但已经包含核心代码、运行流程、结果材料和 AI Coding Agent 参与开发的证据。

This repository is mainly used as supporting material for an AI creator / token-plan application.

It provides evidence that Codex was used as an AI coding agent in a real technical project, rather than only for simple Q&A. The uploaded materials show the development loop from code analysis, training/debugging, benchmark comparison, to result export.

The project is still in an early research and organization stage. Future updates will improve documentation, reproducibility, benchmark clarity, and result visualization.

---

## Notes

Some large intermediate training artifacts may be omitted to keep the repository lightweight.

Version folders are intended to preserve reproducible project snapshots and key evidence materials rather than every temporary runtime file.

部分大型中间训练文件可能不会上传，以避免仓库过大。版本文件夹主要用于保存可复现的项目快照、关键结果和申请证明材料，而不是保存所有临时运行文件。
