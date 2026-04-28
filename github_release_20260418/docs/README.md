# 平顶光全息整合汇报包

## 1. 目标

这份目录用于汇报当前平顶光全息程序的完整实现方案、关键代码、远程训练流程、本地分析流程，以及当前最有代表性的实验结果。

当前主目标是同时优化三项指标，其中优先级为：

1. 核心区均匀性
2. 核心区等相位
3. 光利用效率

当前系统不是单一算法，而是一个混合框架：

- `795` 提供实验经验参考相位
- `Bowman / hybrid` 提供物理优化和最终 polish
- `Lin2025` 提供学习型初始化器

---

## 2. 目录说明

### `code/ai_holography`

当前物理优化主线代码，负责：

- 构造目标场
- 生成初始化候选
- 多分辨率 refine
- Bowman 风格 CG polish
- 统一 benchmark

重点文件：

- `config.py`：物理优化、初始化候选、polish 权重配置
- `runner.py`：初始化候选构建与整条 route 的执行入口
- `pipeline.py`：传播模型、目标构建、AI refine
- `hybrid.py`：Bowman 风格物理 polish
- `losses.py`：均匀性、等相位、效率等损失定义
- `references.py`：`795` 参考相位加载与元数据解析
- `scripts/benchmark_lin2025_hybrid.py`：统一对比入口

### `code/lin2025_holography`

Lin-2025 风格训练与推理代码，负责：

- 数据集构造
- teacher 选择
- CNN 模型训练
- 位置域输出到 hologram 的变换

重点文件：

- `config.py`：训练配置与 teacher 权重
- `dataset.py`：训练样本与 teacher 池逻辑
- `model.py`：Lin2025HologramNet
- `training.py`：训练主循环
- `predictor.py`：把训练好的 checkpoint 变成初始化相位
- `metrics.py`：训练时的质量评价函数

### `notebooks`

- `train_lin2025_autodl_lineflat_highres_export_v58.ipynb`

这是当前推荐的远程训练 notebook：

- 适合 AutoDL
- 自包含
- 自动写出源码
- 面向 line-flat / 高分辨率导出主线
- 训练后自动导出 checkpoint bundle
- 自动跑 benchmark 并给出建议
- 版本演进说明见 `method_comparison_detailed.*` 与 `ppt_new_program_vs_795_detailed.*`

### `analysis`

- `analyze_lin2025_checkpoint_local.py`

本地分析脚本，负责：

- 读取远程训练带回的 checkpoint
- 在本地统一 benchmark
- 自动对比官方 best
- 自动输出调参建议

### `results`

- `official_benchmark_summary.json`：当前官方主线 benchmark 结果
- `autodl_checkpoint_analysis_summary.json`：AutoDL checkpoint 的本地分析结果

### `assets`

- `795_flat_top_d=160_dx=67_dy=-173.npy`
- `795_round_top_d=160_dx=67_dy=-173.npy`

这是当前最核心的两个参考相位文件。

---

## 3. 当前程序的实现策略

### 3.1 总体流程

当前完整流程可以概括为：

`目标场 -> 初始化候选 -> Lin2025 / 795 / warm start 选优 -> phase-only correction -> AI refine -> Bowman-style hybrid polish -> 指标评估`

### 3.2 初始化候选来源

在 `runner.py` 中，初始化候选主要有：

- `reference`
- `lin2025`
- `warm_start`
- 早期版本中还有 `neural`

当前主线上最重要的两个是：

- `reference`：来自 `795`
- `lin2025`：来自训练得到的 checkpoint

然后程序会根据初始化评分选择更好的候选，再进入后续物理优化。

### 3.3 为什么不能只用神经网络

从当前结果看：

- `lin2025_only` 单独使用时效果明显不够好
- 真正有效的是 `lin2025_plus_hybrid`

这说明神经网络更适合作为：

- 高质量初始化器

而不是：

- 完全替代物理优化器

---

## 4. `795`、`Bowman`、`Lin2025` 三者关系

### 4.1 `795`

`795` 的作用是提供实验经验参考相位。

要注意：

当前 benchmark 里的 `795_only` 不是“原始 795 不经处理直接使用”，而是：

- 加载 `795` 参考相位
- 按当前 `slm_size` 插值缩放
- 按文件名中的 `d / dx / dy / style` 重建目标参数
- 在当前代码实现里，还会与少量初始化相位做混合评估

所以 benchmark 里的 `795_only` 更准确地说是：

- 当前仿真尺寸和目标定义下的 `795` 参考基线

### 4.2 `Bowman / hybrid`

`hybrid_only` 是当前最强的纯物理基线。

它的核心是：

- 参考相位 / warm start
- 多分辨率 refine
- Bowman 风格 CG polish

它的优点：

- 物理上稳定
- 效率通常更高
- 最终结果比较稳

### 4.3 `Lin2025`

`Lin2025` 的作用不是直接输出最终最优相位，而是：

- 学习一个更好的初始化器

训练时：

- 用 `795` 与 `hybrid` 结果做 teacher
- 在位置域学习 amplitude / phase
- 最后通过 `predictor.py` 输出 hologram 初相位

它的价值主要体现在：

- 改善初始化质量
- 帮后面的 `hybrid polish` 找到更好的局部最优

---

## 5. 当前最重要的结果

### 5.1 官方主线 benchmark

见：

- `results/official_benchmark_summary.json`

其中官方主线的 best route 是：

- `lin2025_plus_hybrid_quality`

它表明：

- 相比 `795`，整体已经明显提升
- 相比 `hybrid_only`，可以找到核心区更优的解

### 5.2 AutoDL checkpoint 的本地分析

见：

- `results/autodl_checkpoint_analysis_summary.json`

这份结果的意义是：

- 训练在远端完成
- checkpoint 带回本地分析
- 分析脚本自动比较官方 best

在这份分析里，最好的 route 是：

- `lin2025_best_supervised_quality`

核心指标为：

- `core_uniformity_loss = 0.0263995`
- `core_phase_flatness = 0.0013418`
- `efficiency = 0.7894875`

相对当前官方 best：

- 核心区均匀性更好
- 核心区等相位更好
- 效率略低

这说明当前系统已经能在前两项上超过官方主线。

---

## 6. 相比 `795` 和 `Bowman`，提升有多大

### 相比 `795`

提升非常明显，尤其体现在：

- 核心区等相位
- 光利用效率

`795` 现在更适合作为：

- 实验参考相位
- teacher 来源

而不是：

- 最终最优解

### 相比 `Bowman / hybrid_only`

当前系统还没有做到三项同时完全优于 `hybrid_only`，但已经能找到：

- 核心区均匀性更优
- 核心区等相位更优

的结果点。

代价通常是：

- 效率略低

所以当前结论更准确地说是：

- 找到了比 Bowman 更好的 Pareto 点
- 但不是无代价全赢

---

## 7. 当前还有多少优化空间

当前仍然有优化空间，但已经不是“数量级提升”阶段，而更像：

- 在前两项上继续压 10% 到 30%
- 尽量把效率往回拉

目前最现实的优化方向是：

1. 继续用 AutoDL 训练 `Lin2025`
2. 本地统一分析 checkpoint
3. 不盲目加大 epoch
4. 更重视 checkpoint 选择和 route 选择

当前经验表明：

- epoch 过大时，`core_phase_flatness` 容易变坏
- 训练方向不能过度激进，否则会牺牲相位泛化

---

## 8. 推荐的汇报顺序

建议汇报时按下面顺序讲：

1. 研究目标
   - 平顶光三指标优化
2. 三条参考路线
   - `795`
   - `Bowman / hybrid`
   - `Lin2025`
3. 当前混合系统架构
   - `Lin2025 + hybrid`
4. 训练与分析分离流程
   - AutoDL 训练
   - 本地分析
5. 结果对比
   - `795_only`
   - `hybrid_only`
   - `lin2025_plus_hybrid_balanced`
   - `lin2025_plus_hybrid_quality`
6. 当前结论
   - 前两项已能超过基线
   - 效率仍需折中
7. 下一步
   - 继续远程训练
   - 本地筛选更好的 checkpoint

---

## 9. 推荐的使用方式

### 远程训练

使用：

- `notebooks/train_lin2025_autodl_lineflat_highres_export_v58.ipynb`

### 本地分析

使用：

- `analysis/analyze_lin2025_checkpoint_local.py`

流程是：

1. 在远端训练并导出 checkpoint
2. 把 checkpoint 解压到本地
3. 用分析脚本统一评估
4. 再和官方 best 比较

---

## 10. 当前最重要的判断

当前系统已经证明：

- 相比 `795`，提升很大
- 相比 `Bowman`，可以在前两项上取得实质提升
- 但效率通常会略有下降

因此在实验上最稳的策略不是只保留一条路线，而是同时保留：

- `hybrid_only`
- `lin2025_plus_hybrid_balanced`
- `lin2025_plus_hybrid_quality`

再按实验目标选择：

- 前两项优先：先看 `balanced` 或最优 `supervised_quality`
- 效率优先：回退到 `hybrid_only`
