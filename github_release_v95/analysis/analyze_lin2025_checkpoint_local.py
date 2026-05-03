from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_holography.config import HolographyConfig
from ai_holography.losses import composite_loss
from ai_holography.profiles import apply_profile
from ai_holography.references import apply_795_reference_metadata, find_795_reference_files, parse_795_reference_name
from ai_holography.runner import HybridHolographyRunner
from ai_holography.visualization import save_field_visualizations, save_linecuts


def select_reference(refs: Iterable[Path], style: str) -> Path:
    for ref in refs:
        if parse_795_reference_name(ref).get("style") == style:
            return ref
    raise FileNotFoundError(f"Missing 795 reference for style={style}")


def base_cfg(ref: Path, run_name: str) -> HolographyConfig:
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


def evaluate_direct(cfg: HolographyConfig, phase, label: str) -> dict[str, float]:
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
    metrics = {"composite_loss": float(loss.detach().cpu()), **aux}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_field_visualizations(output_dir, problem["target_amp"], problem["target_phase"], out_amp, out_phase, phase)
    save_linecuts(output_dir, problem["target_amp"], out_amp)
    return metrics


def route_score(metrics: dict[str, float]) -> float:
    efficiency_floor = float(metrics.get("efficiency_floor", 0.985))
    efficiency_penalty = max(0.0, efficiency_floor - metrics.get("efficiency", 0.0)) ** 2
    return (
        1.0 * metrics.get("core_uniformity_loss", float("inf"))
        + 1.0 * metrics.get("core_phase_flatness", float("inf"))
        + 0.5 * efficiency_penalty
        + 0.2 * metrics.get("uniformity_loss", float("inf"))
    )


def rank_key(metrics: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        route_score(metrics),
        metrics.get("core_uniformity_loss", float("inf")),
        metrics.get("core_phase_flatness", float("inf")),
        -metrics.get("efficiency", 0.0),
    )


def build_tuning_recommendations(
    best_route: str,
    best_metrics: dict[str, float],
    baseline_best_route: str | None,
    baseline_best_metrics: dict[str, float],
) -> dict[str, object]:
    suggestions: list[dict[str, object]] = []
    observations: list[str] = []

    best_eff = float(best_metrics.get("efficiency", 0.0))
    best_core_u = float(best_metrics.get("core_uniformity_loss", float("inf")))
    best_core_p = float(best_metrics.get("core_phase_flatness", float("inf")))
    baseline_eff = float(baseline_best_metrics.get("efficiency", 0.0))
    baseline_core_u = float(baseline_best_metrics.get("core_uniformity_loss", float("inf")))
    baseline_core_p = float(baseline_best_metrics.get("core_phase_flatness", float("inf")))

    if baseline_best_route:
        observations.append(
            f"best route is {best_route}; official baseline is {baseline_best_route}."
        )
        observations.append(
            "delta vs baseline: "
            f"core_uniformity={best_core_u - baseline_core_u:+.6f}, "
            f"core_phase_flatness={best_core_p - baseline_core_p:+.6f}, "
            f"efficiency={best_eff - baseline_eff:+.6f}."
        )

    if best_route.endswith("_quality"):
        observations.append("quality-priority route is currently dominant.")
    elif best_route.endswith("_balanced"):
        observations.append("balanced route is currently dominant.")
    else:
        observations.append("current best route is outside the standard balanced/quality pair.")

    if baseline_best_route and best_eff < baseline_eff - 0.003:
        suggestions.append(
            {
                "priority": 1,
                "focus": "recover efficiency while preserving the new best core metrics",
                "action": "scan polish_efficiency_weight",
                "target_routes": [best_route],
                "recommended_values": [0.12, 0.14, 0.16, 0.18],
                "why": (
                    "efficiency improved less than the core metrics; this is the most likely single-variable recovery path "
                    "without changing the training setup."
                ),
            }
        )

    if baseline_best_route and best_core_p > baseline_core_p:
        suggestions.append(
            {
                "priority": 2,
                "focus": "close the remaining core-phase gap",
                "action": "increase phase-priority training bias",
                "target_files": [
                    str(ROOT / "lin2025_holography" / "config.py"),
                    str(ROOT / "colab_local_split" / "train_lin2025_colab.ipynb"),
                ],
                "recommended_changes": {
                    "flat_core_phase_weight": [1.8, 2.0],
                    "hybrid_teacher_phase_gate": [0.0018, 0.0015],
                    "teacher_weight_hybrid": [0.9, 1.0],
                },
                "why": "core_phase_flatness is still above the baseline best; teacher and phase-loss bias remain the most direct levers.",
            }
        )

    if baseline_best_route and best_core_u > baseline_core_u:
        suggestions.append(
            {
                "priority": 2,
                "focus": "reduce core uniformity loss",
                "action": "scan core_uniformity_weight on the quality route",
                "target_routes": [best_route],
                "recommended_values": [0.65, 0.70, 0.75],
                "why": "core uniformity remains worse than the baseline best, so route-side quality weights are still the first thing to sweep.",
            }
        )

    if best_route.startswith("lin2025_best_supervised"):
        suggestions.append(
            {
                "priority": 1,
                "focus": "promote the winning checkpoint into the main benchmark path",
                "action": "test the current best_supervised checkpoint as the new official wider_net candidate",
                "checkpoint": str(best_metrics.get("checkpoint", "")),
                "why": "the supervised checkpoint currently beats the quality and hybrid-polish checkpoints in downstream route scoring.",
            }
        )

    if best_eff < 0.80:
        suggestions.append(
            {
                "priority": 3,
                "focus": "set expectations for the current quality regime",
                "action": "keep quality mode for core metrics, but retain hybrid_only as the efficiency-fallback profile",
                "recommended_routes": [best_route, "hybrid_only"],
                "why": "current best route is core-quality dominant but stays below 0.80 efficiency.",
            }
        )

    if not suggestions:
        suggestions.append(
            {
                "priority": 3,
                "focus": "stability",
                "action": "no immediate scan required",
                "why": "current candidate already beats the baseline without a clear single weak metric.",
            }
        )

    suggestions.sort(key=lambda item: int(item.get("priority", 99)))
    return {
        "observations": observations,
        "suggestions": suggestions,
    }


def configure_lin_hybrid_route(
    cfg: HolographyConfig,
    checkpoint_dir: Path,
    checkpoint_file: Path,
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
    cfg.lin2025_checkpoint_dir = checkpoint_dir
    cfg.lin2025_checkpoint = checkpoint_file
    if mode == "balanced":
        cfg.core_uniformity_weight = 0.45
        cfg.core_phase_weight = 0.35
        cfg.efficiency_floor = 0.985
        cfg.polish_phase_priority_core_uniformity_weight = 0.4
        cfg.polish_phase_priority_core_phase_weight = 0.45
    elif mode == "quality":
        cfg.core_uniformity_weight = 0.70
        cfg.core_phase_weight = 0.40
        cfg.efficiency_floor = 0.84
        cfg.polish_efficiency_weight = 0.12
        cfg.polish_phase_priority_core_uniformity_weight = 0.55
        cfg.polish_phase_priority_core_phase_weight = 0.65
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return cfg


def run_lin2025_hybrid_route(
    round_ref: Path,
    flat_ref: Path,
    checkpoint_dir: Path,
    checkpoint_file: Path,
    mode: str,
    summary: dict[str, object],
) -> None:
    route_name = f"{checkpoint_file.stem}_{mode}"
    round_cfg = base_cfg(round_ref, f"analysis_{route_name}_round")
    round_cfg.init_candidate_sources = ("lin2025", "reference")
    round_cfg.lin2025_primary = True
    round_cfg.reference_phase_mix = 0.35
    round_cfg.lin2025_reference_weak_mix = 0.35
    round_cfg.lin2025_checkpoint_dir = checkpoint_dir
    round_cfg.lin2025_checkpoint = checkpoint_file
    round_runner = HybridHolographyRunner(round_cfg)
    round_phase, _ = round_runner.run_once(use_warm_start=False, save_subdir=f"round_{route_name}")

    flat_cfg = base_cfg(flat_ref, f"analysis_{route_name}_flat")
    flat_cfg = configure_lin_hybrid_route(flat_cfg, checkpoint_dir, checkpoint_file, mode=mode)
    flat_runner = HybridHolographyRunner(flat_cfg)
    flat_runner.previous_phase = round_phase
    flat_runner.previous_target_amp = round_runner.previous_target_amp
    flat_runner.previous_target_phase = round_runner.previous_target_phase
    _, metrics = flat_runner.run_once(use_warm_start=True, save_subdir=f"flat_{route_name}")
    metrics["efficiency_floor"] = flat_cfg.efficiency_floor
    metrics["below_efficiency_floor"] = bool(metrics.get("efficiency", 0.0) < flat_cfg.efficiency_floor)
    metrics["analysis_checkpoint"] = str(checkpoint_file)
    summary["routes"][route_name] = metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a Colab-trained Lin2025 checkpoint locally.")
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing lin2025_best_supervised.pt / lin2025_best_quality.pt / lin2025_best_hybrid_polish.pt",
    )
    parser.add_argument(
        "--project-root",
        default=str(ROOT),
        help="Project root containing ai_holography, lin2025_holography, and ftl_gen",
    )
    parser.add_argument(
        "--output-tag",
        default=f"local_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Run tag for analysis outputs",
    )
    parser.add_argument(
        "--baseline-summary",
        default=str(ROOT / "ai_holography" / "runs" / "benchmark_lin2025_hybrid" / "summary.json"),
        help="Existing benchmark summary.json used as the current official best reference",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")

    refs = find_795_reference_files(project_root / "ftl_gen")
    flat_ref = select_reference(refs, "flat_top")
    round_ref = select_reference(refs, "round_top")

    summary: dict[str, object] = {"routes": {}, "checkpoint_dir": str(checkpoint_dir)}

    # Current hybrid baseline.
    round_cfg = base_cfg(round_ref, f"{args.output_tag}_hybrid_round")
    round_cfg.init_candidate_sources = ("neural", "reference")
    round_runner = HybridHolographyRunner(round_cfg)
    round_phase, _ = round_runner.run_once(use_warm_start=False, save_subdir=f"round_hybrid_only_{args.output_tag}")

    flat_cfg = base_cfg(flat_ref, f"{args.output_tag}_hybrid_flat")
    flat_cfg.init_candidate_sources = ("neural", "reference", "warm_start")
    flat_runner = HybridHolographyRunner(flat_cfg)
    flat_runner.previous_phase = round_phase
    flat_runner.previous_target_amp = round_runner.previous_target_amp
    flat_runner.previous_target_phase = round_runner.previous_target_phase
    _, hybrid_metrics = flat_runner.run_once(use_warm_start=True, save_subdir=f"flat_hybrid_only_{args.output_tag}")
    summary["routes"]["hybrid_only"] = hybrid_metrics

    # Evaluate all standard Lin checkpoints if present.
    for filename in (
        "lin2025_best_supervised.pt",
        "lin2025_best_quality.pt",
        "lin2025_best_hybrid_polish.pt",
    ):
        checkpoint_file = checkpoint_dir / filename
        if not checkpoint_file.exists():
            continue
        run_lin2025_hybrid_route(round_ref, flat_ref, checkpoint_dir, checkpoint_file, mode="balanced", summary=summary)
        run_lin2025_hybrid_route(round_ref, flat_ref, checkpoint_dir, checkpoint_file, mode="quality", summary=summary)

    ranked = sorted(summary["routes"].items(), key=lambda item: rank_key(item[1]))
    summary["ranking"] = [{"route": name, "score": route_score(metrics), "metrics": metrics} for name, metrics in ranked]
    summary["best_route"] = ranked[0][0]
    summary["best_route_reason"] = (
        f"score={route_score(ranked[0][1]):.6f}, "
        f"core_uniformity={ranked[0][1].get('core_uniformity_loss', float('inf')):.6f}, "
        f"core_phase_flatness={ranked[0][1].get('core_phase_flatness', float('inf')):.6f}, "
        f"efficiency={ranked[0][1].get('efficiency', 0.0):.6f}"
    )

    baseline_summary_path = Path(args.baseline_summary).resolve()
    if baseline_summary_path.exists():
        baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
        baseline_best_route = baseline_summary.get("best_route")
        baseline_best_metrics = baseline_summary.get("routes", {}).get(baseline_best_route, {})
        candidate_best_metrics = ranked[0][1]
        candidate_score = route_score(candidate_best_metrics)
        baseline_score = route_score(baseline_best_metrics) if baseline_best_metrics else float("inf")
        comparison = {
            "baseline_summary": str(baseline_summary_path),
            "baseline_best_route": baseline_best_route,
            "baseline_score": baseline_score,
            "candidate_best_route": ranked[0][0],
            "candidate_score": candidate_score,
            "beats_baseline": candidate_score < baseline_score,
            "delta": {
                "score": candidate_score - baseline_score,
                "core_uniformity_loss": float(candidate_best_metrics.get("core_uniformity_loss", float("inf")))
                - float(baseline_best_metrics.get("core_uniformity_loss", float("inf"))),
                "core_phase_flatness": float(candidate_best_metrics.get("core_phase_flatness", float("inf")))
                - float(baseline_best_metrics.get("core_phase_flatness", float("inf"))),
                "efficiency": float(candidate_best_metrics.get("efficiency", 0.0))
                - float(baseline_best_metrics.get("efficiency", 0.0)),
            },
        }
        if comparison["beats_baseline"]:
            comparison["verdict"] = "Candidate checkpoint analysis exceeds the current official best."
        else:
            deltas = comparison["delta"]
            if deltas["core_phase_flatness"] > 0:
                drag = "core_phase_flatness is still worse than the current official best"
            elif deltas["core_uniformity_loss"] > 0:
                drag = "core_uniformity_loss is still worse than the current official best"
            else:
                drag = "score remains higher overall despite mixed metric changes"
            comparison["verdict"] = f"Candidate checkpoint analysis does not exceed the current official best; {drag}."
        summary["comparison_to_official_best"] = comparison
        summary["tuning_recommendations"] = build_tuning_recommendations(
            best_route=ranked[0][0],
            best_metrics=candidate_best_metrics,
            baseline_best_route=baseline_best_route,
            baseline_best_metrics=baseline_best_metrics,
        )
    else:
        summary["comparison_to_official_best"] = {
            "baseline_summary": str(baseline_summary_path),
            "available": False,
            "verdict": "Baseline summary not found; skipped comparison to official best.",
        }
        summary["tuning_recommendations"] = build_tuning_recommendations(
            best_route=ranked[0][0],
            best_metrics=ranked[0][1],
            baseline_best_route=None,
            baseline_best_metrics={},
        )

    out_dir = project_root / "ai_holography" / "runs" / args.output_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote analysis summary to: {out_path}")


if __name__ == "__main__":
    main()
