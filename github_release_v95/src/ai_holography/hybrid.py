from __future__ import annotations

import time

import torch
from torch import nn

from .config import HolographyConfig
from .losses import composite_loss, efficiency_weighted_overlap_loss, normalized_overlap
from .propagation import FourierSLM, wrap_phase


def bowman_cg_refine(
    cfg: HolographyConfig,
    problem: dict[str, torch.Tensor],
    init_phase: torch.Tensor,
    maxiter: int = 120,
    target_overlap: float | None = None,
) -> dict[str, object]:
    device = torch.device(cfg.device)
    propagator = FourierSLM(cfg.slm_size, cfg.output_size).to(device)
    beam = problem["beam"]
    target_amp = problem["target_amp"]
    target_phase = problem["target_phase"]
    weight = problem["weight"]

    def run_cg_stage(
        phase0: torch.Tensor,
        maxiter_stage: int,
        objective_mode: str,
        target_overlap_stage: float | None,
    ) -> tuple[torch.Tensor, float, int, float]:
        best: dict[str, object] = {
            "phase": phase0.detach().clone(),
            "loss": float("inf"),
            "overlap": 0.0,
        }

        phase_param = nn.Parameter(phase0.detach().clone().to(device))
        optimizer = torch.optim.LBFGS(
            [phase_param],
            lr=1.0,
            max_iter=1,
            history_size=20,
            line_search_fn="strong_wolfe",
        )

        def objective_and_overlap(phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            out_amp, out_phase = propagator(beam, phase)
            overlap = normalized_overlap(target_amp, target_phase, out_amp, out_phase, weight)
            if objective_mode == "overlap":
                loss = 1e9 * (1.0 - overlap).square()
            elif objective_mode == "efficiency_overlap":
                loss = 1e9 * efficiency_weighted_overlap_loss(
                    target_amp=target_amp,
                    target_phase=target_phase,
                    out_amp=out_amp,
                    out_phase=out_phase,
                    weight=weight,
                    steepness=2.0,
                )
            elif objective_mode == "phase_priority":
                loss, _ = composite_loss(
                    target_amp=target_amp,
                    target_phase=target_phase,
                    out_amp=out_amp,
                    out_phase=out_phase,
                    weight=weight,
                    slm_phase=phase,
                    efficiency_floor=cfg.efficiency_floor,
                    overlap_weight=cfg.polish_overlap_weight,
                    intensity_weight=0.0,
                    phase_weight=cfg.polish_phase_priority_phase_weight,
                    uniformity_weight=cfg.polish_phase_priority_uniformity_weight,
                    efficiency_weight=cfg.polish_phase_priority_efficiency_weight,
                    core_uniformity_weight=cfg.polish_phase_priority_core_uniformity_weight,
                    core_phase_weight=cfg.polish_phase_priority_core_phase_weight,
                    core_region_threshold=cfg.core_region_threshold,
                    smoothness_weight=cfg.polish_smoothness_weight,
                    intensity_tv_weight=cfg.polish_intensity_tv_weight,
                )
            else:
                loss, _ = composite_loss(
                    target_amp=target_amp,
                    target_phase=target_phase,
                    out_amp=out_amp,
                    out_phase=out_phase,
                    weight=weight,
                    slm_phase=phase,
                    efficiency_floor=cfg.efficiency_floor,
                    overlap_weight=cfg.polish_overlap_weight,
                    intensity_weight=cfg.polish_intensity_weight,
                    phase_weight=cfg.polish_phase_weight,
                    uniformity_weight=cfg.polish_uniformity_weight,
                    efficiency_weight=cfg.polish_efficiency_weight,
                    core_uniformity_weight=cfg.core_uniformity_weight,
                    core_phase_weight=cfg.core_phase_weight,
                    core_region_threshold=cfg.core_region_threshold,
                    smoothness_weight=cfg.polish_smoothness_weight,
                    intensity_tv_weight=cfg.polish_intensity_tv_weight,
                )
            return loss, overlap

        stage_start = time.perf_counter()
        steps_taken = 0
        for _ in range(maxiter_stage):
            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                phase = wrap_phase(phase_param)
                loss, _ = objective_and_overlap(phase)
                loss.backward()
                return loss

            optimizer.step(closure)
            steps_taken += 1
            with torch.no_grad():
                phase = wrap_phase(phase_param)
                loss, overlap = objective_and_overlap(phase)
                loss_value = float(loss.detach().cpu())
                overlap_value = float(overlap.detach().cpu())
                if loss_value < best["loss"]:
                    best["loss"] = loss_value
                    best["overlap"] = overlap_value
                    best["phase"] = phase.detach().clone()
                phase_param.copy_(phase)
            if target_overlap_stage is not None and best["overlap"] >= target_overlap_stage:
                break
        stage_runtime = time.perf_counter() - stage_start
        result_phase = best["phase"].detach().clone()
        result_cost = float(best["loss"])
        return result_phase, result_cost, steps_taken, stage_runtime

    x0 = wrap_phase(init_phase.detach().to(device))
    start = time.perf_counter()
    stage1_phase, stage1_cost, stage1_steps, stage1_runtime = run_cg_stage(
        phase0=x0,
        maxiter_stage=min(maxiter, cfg.polish_stage1_maxiter),
        objective_mode="efficiency_overlap",
        target_overlap_stage=target_overlap,
    )
    remaining = max(
        0,
        min(maxiter, cfg.polish_stage1_maxiter + cfg.polish_stage2_maxiter + cfg.polish_stage3_maxiter) - stage1_steps,
    )
    if remaining > 0:
        stage2_phase, stage2_cost, stage2_steps, stage2_runtime = run_cg_stage(
            phase0=stage1_phase,
            maxiter_stage=min(remaining, cfg.polish_stage2_maxiter),
            objective_mode="composite",
            target_overlap_stage=target_overlap,
        )
        result_phase = stage2_phase
        result_cost = stage2_cost
        total_steps = stage1_steps + stage2_steps
        stage2_used = True
        stage2_time = stage2_runtime
        remaining_after_stage2 = max(0, remaining - stage2_steps)
    else:
        result_phase = stage1_phase
        result_cost = stage1_cost
        total_steps = stage1_steps
        stage2_used = False
        stage2_time = 0.0
        remaining_after_stage2 = 0
    stage3_used = False
    stage3_time = 0.0
    if cfg.polish_enable_phase_priority and remaining_after_stage2 > 0:
        stage3_phase, stage3_cost, stage3_steps, stage3_runtime = run_cg_stage(
            phase0=result_phase,
            maxiter_stage=min(remaining_after_stage2, cfg.polish_stage3_maxiter),
            objective_mode="phase_priority",
            target_overlap_stage=target_overlap,
        )
        result_phase = stage3_phase
        result_cost = stage3_cost
        total_steps += stage3_steps
        stage3_used = True
        stage3_time = stage3_runtime
    runtime = time.perf_counter() - start

    final_phase = wrap_phase(
        result_phase.detach().to(device)
    )
    out_amp, out_phase = propagator(beam, final_phase)
    overlap = normalized_overlap(target_amp, target_phase, out_amp, out_phase, weight)
    loss, aux = composite_loss(
        target_amp=target_amp,
        target_phase=target_phase,
        out_amp=out_amp,
        out_phase=out_phase,
        weight=weight,
        slm_phase=final_phase,
        efficiency_floor=cfg.efficiency_floor,
        overlap_weight=cfg.polish_overlap_weight,
        intensity_weight=cfg.polish_intensity_weight,
        phase_weight=cfg.polish_phase_weight,
        uniformity_weight=cfg.polish_uniformity_weight,
        efficiency_weight=cfg.polish_efficiency_weight,
        core_uniformity_weight=cfg.core_uniformity_weight,
        core_phase_weight=cfg.core_phase_weight,
        core_region_threshold=cfg.core_region_threshold,
        smoothness_weight=cfg.polish_smoothness_weight,
        intensity_tv_weight=cfg.polish_intensity_tv_weight,
    )
    metrics = {
        "runtime_sec": runtime,
        "bowman_cost": result_cost,
        "overlap": float(overlap.detach().cpu()),
        "composite_loss": float(loss.detach().cpu()),
        "cg_steps": total_steps,
        "stage1_steps": stage1_steps,
        "stage2_used": stage2_used,
        "stage2_runtime_sec": stage2_time,
        "stage3_used": stage3_used,
        "stage3_runtime_sec": stage3_time,
        **aux,
    }
    return {
        "phase": final_phase.detach().cpu(),
        "out_amp": out_amp.detach().cpu(),
        "out_phase": out_phase.detach().cpu(),
        "metrics": metrics,
    }
