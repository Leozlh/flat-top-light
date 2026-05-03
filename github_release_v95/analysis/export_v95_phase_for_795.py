"""
导出 v95 Pipeline 相位图为 795.py 可加载的 .npy 文件。

用法：
  1. 在 v95 notebook 中运行 Pipeline 拿到 phase_sD (或任意候选)
  2. 运行本脚本保存相位图
  3. 在 795.py 中用 init_phi = torch.from_numpy(np.load('v95_phase_export.npy')).cuda() 加载

导出内容：
  - v95_phase_export.npy          : 最终选中相位 (SLM 平面, 1024×1272, [0, 2π])
  - v95_phase_all_candidates.npz  : 所有 Pareto 候选的相位图
  - v95_eval_metrics.json         : 仿真评估指标 (与 795 可直接对比)
"""

import torch
import numpy as np
import json
from pathlib import Path


def export_phase(phase_tensor, path):
    """导出相位图为 .npy (1024×1272, [0, 2π], float64)。"""
    if phase_tensor.dim() == 1:
        phase_tensor = phase_tensor.reshape(1024, 1272)
    phase_np = torch.remainder(phase_tensor.detach().cpu(), 2 * np.pi).numpy()
    np.save(path, phase_np)
    print(f"已保存: {path}  shape={phase_np.shape}  range=[{phase_np.min():.4f}, {phase_np.max():.4f}]")
    return phase_np


def export_all_candidates(candidates_dict, path):
    """导出所有候选相位图为 .npz。"""
    np_dict = {}
    for name, phase_tensor in candidates_dict.items():
        if phase_tensor.dim() == 1:
            phase_tensor = phase_tensor.reshape(1024, 1272)
        np_dict[name] = torch.remainder(phase_tensor.detach().cpu(), 2 * np.pi).numpy()
    np.savez(path, **np_dict)
    print(f"已保存 {len(np_dict)} 个候选到: {path}")


def export_metrics(metrics_dict, path):
    """导出评估指标为 JSON。"""
    # 移除不可序列化的字段
    clean = {}
    for k, v in metrics_dict.items():
        if isinstance(v, (int, float, str, bool, list)):
            clean[k] = v
        elif isinstance(v, np.floating):
            clean[k] = float(v)
        elif isinstance(v, np.integer):
            clean[k] = int(v)
    with open(path, 'w') as f:
        json.dump(clean, f, indent=2)
    print(f"已保存指标到: {path}")


# ===========================================================================
# 主入口 — 在 v95 notebook 中 import 本模块后调用
# ===========================================================================
if __name__ == "__main__":
    print("本脚本需要在 v95 notebook 中 import 后使用。")
    print()
    print("用法示例 (在 v95 notebook 的最后一个 cell 中):")
    print()
    print("  from export_v95_phase_for_795 import export_phase, export_all_candidates, export_metrics")
    print()
    print("  # 导出最终选中的相位")
    print("  export_phase(selected_phase, 'v95_phase_export.npy')")
    print()
    print("  # 导出所有候选")
    print("  candidates = {")
    print("      'stage0_bowman': phase_s0,")
    print("      'stageA_anchor': phase_sA,")
    print("      'stageB_constrained': phase_sB,")
    print("      'stageC_polish': phase_sC,")
    print("      'stageD_wgs': phase_sD,")
    print("  }")
    print("  export_all_candidates(candidates, 'v95_phase_all_candidates.npz')")
    print()
    print("  # 导出指标")
    print("  export_metrics(selected_metrics, 'v95_eval_metrics.json')")
    print()
    print("#" + "=" * 60)
    print("# 然后在 795.py 中加载:")
    print("#   init_phi = torch.from_numpy(np.load('v95_phase_export.npy')).flatten().to('cuda')")
    print("#   slm_opt = slm.SLM(NT=NT, N=N, numb=numb, initial_phi=init_phi, profile_s=L)")
    print("#" + "=" * 60)
