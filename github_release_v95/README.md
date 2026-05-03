# Flat-Top Light Holography — v95 Release

在 Holoeye 1024×1272 相位型 SLM 上生成 100 µm × 20 µm 线状平顶光斑的全息算法。

## 核心结果

v95 是首个同时达成 phase_flatness < 0.3 rad + flat_rms < 0.25 的版本，在 6/7 个有效指标上优于 Bowman CG 基线。

| 指标 | v95 Pipeline | Bowman CG | 795 CG |
|---|---|---|---|
| overlap | **0.713** | 0.685 | 0.606 |
| phase_flatness | **0.289 rad** | 0.283 rad | 0.868 rad |
| flat_rms | **0.211** | 0.256 | 0.076 |
| efficiency | 0.559 | 0.526 | **0.829** |
| cost_se | 10867 | 13126 | **597** |
| spillover | 0.415 | 0.453 | **0.143** |

## 目录结构

```
github_release_v95/
├── notebooks/
│   └── train_lin2025_autodl_lineflat_highres_export_v95.ipynb  ← 主 notebook
├── src/
│   ├── ai_holography/          ← 核心算法包
│   │   ├── propagation.py      ← Fraunhofer 衍射
│   │   ├── losses.py           ← 损失函数
│   │   ├── pipeline.py         ← v95 五阶段 Pipeline
│   │   ├── config.py           ← 参数配置
│   │   ├── references.py       ← 参考相位加载
│   │   ├── runner.py           ← benchmark runner
│   │   └── scripts/            ← 独立运行脚本
│   └── lin2025_holography/     ← 学习式初始化器
│       ├── model.py            ← 神经网络模型
│       ├── training.py         ← 训练逻辑
│       └── scripts/
├── legacy_references/ftl_gen/  ← 795 原始程序及依赖
│   ├── 795.py                  ← 原 SOP (scipy CG + 相机闭环)
│   ├── SLM_1X.py               ← SLM 物理模型
│   ├── CG_new.py               ← CG 优化器
│   └── CG_2.py                 ← CG 优化器 v2
├── assets/                     ← 参考相位 .npy 文件
├── analysis/                   ← 分析和对比脚本
└── docs/                       ← 详细报告 PDF
```

## 快速开始

### 环境要求

```bash
pip install torch numpy scipy matplotlib
```

GPU 推荐但非必须。CPU 可运行但较慢。

### 运行 notebook

1. 打开 `notebooks/train_lin2025_autodl_lineflat_highres_export_v95.ipynb`
2. 按顺序运行所有 cell
3. Cell 16 是核心算法（~2000 行），包含三算法对比和 Pareto 选择

### 独立运行 benchmark

```bash
cd src
python -m ai_holography.scripts.benchmark_lin2025_hybrid
```

## 算法架构

v95 Pipeline 是 5 个 stage 串联，每个 stage 保存候选，最后 Pareto 选择：

```
LG init → Stage 0 (Bowman warm) → Stage A (anchored lift) → Stage B (multi-snap fmin_cg)
        → Stage C (Adam trust-region) → Stage D (damped WGS + TV-prox) → Pareto front
```

- **Stage 0**: 复刻 Bowman CG (LG vortex + scipy fmin_cg)，锁定 phase ≈ 0.283 rad
- **Stage A**: 锚点约束下推 efficiency/overlap，λ schedule 从 5000 降到 300
- **Stage B**: 无锚点 overlap/eff 推压，多角度 snapshot (cost_min, phase_min, constrained)
- **Stage C**: 小步 Adam 同时下压 phase 和 flat_rms，双段 phase cap (0.32/0.40)
- **Stage D**: Damped WGS 振幅匹配 + TV-prox 抑制相位毛刺，alpha backtracking

## 物理参数

| 参数 | 值 |
|---|---|
| 波长 | 795 nm |
| 焦距 | 0.2 m |
| SLM 像素 | 12.5 µm × (1024 × 1272) |
| 焦平面采样 | 6876 × 6876 |
| 入射光束 | 高斯, 1/e² 半径 3.5 mm |
| 目标平顶 | 100 µm × 20 µm, 偏移 (-100, -100) µm |

## 与 795 原程序的关系

- `legacy_references/ftl_gen/795.py` 是实验室原始 SOP
- 795 使用 scipy newton-CG + 相机闭环，纯仿真中 phase_flatness 差 (0.87 rad)
- v95 使用多阶段 Pipeline + Pareto 选择，纯仿真中 phase_flatness = 0.289 rad
- v95 尚未集成相机闭环，实测需移植 795.py 的 IDS_Camera 反馈块

## 文档

- `docs/method_comparison_detailed.pdf` — 方法对比详细报告
- `docs/ppt_new_program_vs_795_detailed.pdf` — 新旧程序对比 PPT

## 引用

如果使用本代码，请引用：
- Optics Express 25, 11692 (2017) — 原始 CG 方法
- 本仓库 — v95 Pipeline 实现
