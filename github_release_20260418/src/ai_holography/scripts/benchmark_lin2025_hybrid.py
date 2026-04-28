import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_holography.config import HolographyConfig
from ai_holography.losses import composite_loss
from ai_holography.profiles import apply_profile
from ai_holography.references import apply_795_reference_metadata, find_795_reference_files, parse_795_reference_name
from ai_holography.runner import HybridHolographyRunner
from ai_holography.visualization import save_field_visualizations, save_linecuts
from lin2025_holography.config import Lin2025Config


def _select_reference(refs: list[Path], style: str) -> Path:
    for ref in refs:
        if parse_795_reference_name(ref).get("style") == style:
            return ref
    raise FileNotFoundError(f"Missing 795 reference for style={style}")


def _base_cfg(ref: Path, run_name: str) -> HolographyConfig:
    cfg = apply_profile(HolographyConfig(run_name=run_name), "experiment")
    info = parse_795_reference_name(ref)
    cfg.reference_phase_path = ref
    cfg = apply_795_reference_metadata(cfg, info)
    cfg.phase_type = "phase_flat"
    cfg.uniformity_weight = 0.2
    cfg.efficiency_weight = 0.1
    cfg.core_uniformity_weight = 0.45
    cfg.core_phase_weight = 0.2
    cfg.core_region_threshold = 0.78
    cfg.polish_uniformity_weight = 0.18
    cfg.polish_efficiency_weight = 0.1
    cfg.polish_phase_weight = 0.1
    cfg.target_overlap = 0.9990
    cfg.hybrid_cg_maxiter = 35
    cfg.warm_start_candidate_threshold = 0.0
    cfg.warm_start_similarity_threshold = 0.0
    return cfg


def _evaluate_direct(cfg: HolographyConfig, phase: torch.Tensor, label: str) -> dict[str, float]:
    pipeline = HybridHolographyRunner(cfg).pipeline
    problem = pipeline.build_problem()
    out_amp, out_phase = pipeline.propagator(problem["beam"], phase.to(pipeline.device))
    loss, aux = composite_loss(
        target_amp=problem["target_amp"],
        target_phase=problem["target_phase"],
        out_amp=out_amp,
        out_phase=out_phase,
        weight=problem["weight"],
        slm_phase=phase.to(pipeline.device),
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
    )
    output_dir = cfg.output_dir / label
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "slm_phase.npy", phase.detach().cpu().numpy())
    metrics = {"composite_loss": float(loss.detach().cpu()), **aux}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_field_visualizations(output_dir, problem["target_amp"], problem["target_phase"], out_amp, out_phase, phase)
    save_linecuts(output_dir, problem["target_amp"], out_amp)
    return metrics


def _route_score(metrics: dict[str, float]) -> float:
    efficiency_floor = float(metrics.get("efficiency_floor", 0.985))
    efficiency_penalty = max(0.0, efficiency_floor - metrics.get("efficiency", 0.0)) ** 2
    return (
        1.0 * metrics.get("core_uniformity_loss", float("inf"))
        + 1.0 * metrics.get("core_phase_flatness", float("inf"))
        + 0.5 * efficiency_penalty
        + 0.2 * metrics.get("uniformity_loss", float("inf"))
    )


def _rank_key(metrics: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        _route_score(metrics),
        metrics.get("core_uniformity_loss", float("inf")),
        metrics.get("core_phase_flatness", float("inf")),
        -metrics.get("efficiency", 0.0),
    )


def _best_reason(metrics: dict[str, float]) -> str:
    return (
        f"score={_route_score(metrics):.6f}, "
        f"core_uniformity={metrics.get('core_uniformity_loss', float('inf')):.6f}, "
        f"core_phase_flatness={metrics.get('core_phase_flatness', float('inf')):.6f}, "
        f"efficiency={metrics.get('efficiency', 0.0):.6f}"
    )


def _configure_lin_hybrid_route(
    cfg: HolographyConfig,
    mode: str,
) -> HolographyConfig:
    cfg.init_candidate_sources = ("lin2025", "reference", "warm_start")
    cfg.lin2025_primary = True
    cfg.reference_phase_mix = 0.35
    cfg.lin2025_reference_weak_mix = 0.35
    cfg.warm_start_allow_direct = False
    cfg.warm_start_external_mix_floor = 0.45
    cfg.polish_overlap_weight = 0.85
    cfg.polish_phase_weight = 0.08
    cfg.polish_uniformity_weight = 0.1
    cfg.polish_efficiency_weight = 0.05
    cfg.hybrid_cg_maxiter = 20
    cfg.polish_stage1_maxiter = 8
    cfg.polish_stage2_maxiter = 8
    cfg.polish_stage3_maxiter = 4
    cfg.polish_phase_priority_uniformity_weight = 0.04
    cfg.polish_phase_priority_efficiency_weight = 0.01
    cfg.polish_phase_priority_phase_weight = 0.12

    if mode == "balanced":
        cfg.core_uniformity_weight = 0.45
        cfg.core_phase_weight = 0.35
        cfg.efficiency_floor = 0.985
        cfg.polish_phase_priority_core_uniformity_weight = 0.4
        cfg.polish_phase_priority_core_phase_weight = 0.45
    elif mode == "quality":
        wider_ckpt_dir = Path("D:/Trae products/flat_top light/lin2025_holography/checkpoints/wider_net")
        wider_ckpt = wider_ckpt_dir / "lin2025_best_hybrid_polish.pt"
        if wider_ckpt.exists():
            cfg.lin2025_checkpoint_dir = wider_ckpt_dir
            cfg.lin2025_checkpoint = wider_ckpt
        cfg.core_uniformity_weight = 0.70
        cfg.core_phase_weight = 0.40
        cfg.efficiency_floor = 0.84
        cfg.polish_efficiency_weight = 0.12
        cfg.polish_phase_priority_core_uniformity_weight = 0.55
        cfg.polish_phase_priority_core_phase_weight = 0.65
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return cfg


def _run_lin2025_hybrid_route(
    round_ref: Path,
    flat_ref: Path,
    mode: str,
    summary: dict[str, object],
) -> None:
    route_name = f"lin2025_plus_hybrid_{mode}"
    round_cfg = _base_cfg(round_ref, f"benchmark_{mode}_round")
    round_cfg.init_candidate_sources = ("lin2025", "reference")
    round_cfg.lin2025_primary = True
    round_cfg.reference_phase_mix = 0.35
    round_cfg.lin2025_reference_weak_mix = 0.35
    round_runner = HybridHolographyRunner(round_cfg)
    round_phase, _ = round_runner.run_once(use_warm_start=False, save_subdir=f"round_{route_name}")

    flat_cfg = _base_cfg(flat_ref, f"benchmark_{mode}_flat")
    flat_cfg = _configure_lin_hybrid_route(flat_cfg, mode=mode)
    flat_runner = HybridHolographyRunner(flat_cfg)
    flat_runner.previous_phase = round_phase
    flat_runner.previous_target_amp = round_runner.previous_target_amp
    flat_runner.previous_target_phase = round_runner.previous_target_phase
    _, metrics = flat_runner.run_once(use_warm_start=True, save_subdir=f"flat_{route_name}")
    metrics["quality_priority_mode"] = bool(mode == "quality")
    metrics["efficiency_floor"] = flat_cfg.efficiency_floor
    metrics["below_efficiency_floor"] = bool(metrics.get("efficiency", 0.0) < flat_cfg.efficiency_floor)
    summary["routes"][route_name] = metrics


def main() -> None:
    ref_dir = Path("D:/Trae products/flat_top light/ftl_gen")
    refs = find_795_reference_files(ref_dir)
    flat_ref = _select_reference(refs, "flat_top")
    round_ref = _select_reference(refs, "round_top")

    summary: dict[str, object] = {"routes": {}}

    # 795 only
    round_cfg = _base_cfg(round_ref, "benchmark_795_round")
    round_cfg.init_candidate_sources = ("reference",)
    round_runner = HybridHolographyRunner(round_cfg)
    round_phase, _, _ = round_runner._build_init_phase(round_runner.pipeline.build_problem(), use_warm_start=False)
    _ = _evaluate_direct(round_cfg, round_phase, "round_795_only")

    flat_cfg = _base_cfg(flat_ref, "benchmark_795_flat")
    flat_cfg.init_candidate_sources = ("reference",)
    flat_runner = HybridHolographyRunner(flat_cfg)
    flat_phase, _, _ = flat_runner._build_init_phase(flat_runner.pipeline.build_problem(), use_warm_start=False)
    summary["routes"]["795_only"] = _evaluate_direct(flat_cfg, flat_phase, "flat_795_only")

    # hybrid only
    round_cfg = _base_cfg(round_ref, "benchmark_hybrid_round")
    round_cfg.init_candidate_sources = ("neural", "reference")
    round_runner = HybridHolographyRunner(round_cfg)
    round_phase, round_metrics = round_runner.run_once(use_warm_start=False, save_subdir="round_hybrid")

    flat_cfg = _base_cfg(flat_ref, "benchmark_hybrid_flat")
    flat_cfg.init_candidate_sources = ("neural", "reference", "warm_start")
    flat_runner = HybridHolographyRunner(flat_cfg)
    flat_runner.previous_phase = round_phase
    flat_runner.previous_target_amp = round_runner.previous_target_amp
    flat_runner.previous_target_phase = round_runner.previous_target_phase
    _, hybrid_metrics = flat_runner.run_once(use_warm_start=True, save_subdir="flat_hybrid")
    summary["routes"]["hybrid_only"] = hybrid_metrics

    # lin2025 only
    lin_cfg = Lin2025Config(target_mode="flat_top", slm_size=flat_cfg.slm_size, device=flat_cfg.device)
    round_cfg = _base_cfg(round_ref, "benchmark_lin_round")
    round_cfg.init_candidate_sources = ("lin2025",)
    round_runner = HybridHolographyRunner(round_cfg)
    round_phase, _, _ = round_runner._build_init_phase(round_runner.pipeline.build_problem(), use_warm_start=False)
    _ = _evaluate_direct(round_cfg, round_phase, "round_lin_only")

    flat_cfg = _base_cfg(flat_ref, "benchmark_lin_flat")
    flat_cfg.init_candidate_sources = ("lin2025",)
    flat_runner = HybridHolographyRunner(flat_cfg)
    flat_phase, _, _ = flat_runner._build_init_phase(flat_runner.pipeline.build_problem(), use_warm_start=False)
    summary["routes"]["lin2025_only"] = _evaluate_direct(flat_cfg, flat_phase, "flat_lin_only")

    _run_lin2025_hybrid_route(round_ref, flat_ref, mode="balanced", summary=summary)
    _run_lin2025_hybrid_route(round_ref, flat_ref, mode="quality", summary=summary)

    ranked = sorted(summary["routes"].items(), key=lambda item: _rank_key(item[1]))
    summary["ranking"] = [{"route": name, "score": _route_score(metrics), "metrics": metrics} for name, metrics in ranked]
    summary["best_route"] = ranked[0][0]
    summary["best_route_reason"] = _best_reason(ranked[0][1])
    benchmark_dir = Path("D:/Trae products/flat_top light/ai_holography/runs/benchmark_lin2025_hybrid")
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    (benchmark_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
