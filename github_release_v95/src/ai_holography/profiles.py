from __future__ import annotations

from .config import HolographyConfig


def apply_profile(cfg: HolographyConfig, name: str) -> HolographyConfig:
    if name == "balanced":
        return cfg
    if name == "uniformity":
        cfg.uniformity_weight = 0.4
        cfg.efficiency_weight = 0.08
        cfg.phase_weight = 0.18
        cfg.polish_uniformity_weight = 0.4
        cfg.polish_efficiency_weight = 0.08
        cfg.polish_phase_weight = 0.2
        return cfg
    if name == "efficiency":
        cfg.uniformity_weight = 0.1
        cfg.efficiency_weight = 0.3
        cfg.phase_weight = 0.12
        cfg.polish_uniformity_weight = 0.1
        cfg.polish_efficiency_weight = 0.35
        cfg.polish_phase_weight = 0.12
        return cfg
    if name == "phase":
        cfg.uniformity_weight = 0.15
        cfg.efficiency_weight = 0.1
        cfg.phase_weight = 0.3
        cfg.polish_uniformity_weight = 0.15
        cfg.polish_efficiency_weight = 0.1
        cfg.polish_phase_weight = 0.3
        return cfg
    if name == "experiment":
        cfg.uniformity_weight = 0.15
        cfg.efficiency_weight = 0.25
        cfg.phase_weight = 0.12
        cfg.core_uniformity_weight = 0.15
        cfg.core_phase_weight = 0.12
        cfg.smoothness_weight = 5e-4
        cfg.polish_uniformity_weight = 0.12
        cfg.polish_efficiency_weight = 0.3
        cfg.polish_phase_weight = 0.1
        cfg.polish_smoothness_weight = 5e-4
        cfg.target_overlap = 0.9988
        cfg.hybrid_cg_maxiter = 25
        cfg.polish_stage1_maxiter = 12
        cfg.polish_stage2_maxiter = 8
        return cfg
    raise ValueError(f"Unknown profile: {name}")
