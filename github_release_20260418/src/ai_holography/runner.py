from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lin2025_holography.config import Lin2025Config
from lin2025_holography.predictor import predict_hologram_phase

from .config import HolographyConfig
from .camera_loop import (
    CameraLoopConfig,
    camera_feedback_weight,
    load_measured_intensity,
    save_measured_intensity,
    update_target_from_measurement,
)
from .hybrid import bowman_cg_refine
from .losses import composite_loss, normalized_overlap
from .pipeline import AIHolographyPipeline
from .propagation import wrap_phase
from .references import load_reference_phase, parse_795_reference_name
from .visualization import save_field_visualizations, save_linecuts


class HybridHolographyRunner:
    """
    Production-oriented runner:
    1. neural initialization
    2. fast multiscale differentiable refinement
    3. short CG polish
    4. optional warm start from the previous generated phase
    """

    def __init__(self, config: HolographyConfig):
        self.cfg = config
        self.pipeline = AIHolographyPipeline(config)
        self.previous_phase: torch.Tensor | None = None
        self.previous_target_amp: torch.Tensor | None = None
        self.previous_target_phase: torch.Tensor | None = None
        self.camera_loop = CameraLoopConfig()
        self.last_init_source: str | None = None

    def _target_similarity(self, problem: dict[str, torch.Tensor]) -> float:
        if self.previous_target_amp is None or self.previous_target_phase is None:
            return 0.0

        amp_a = problem["target_amp"]
        amp_b = self.previous_target_amp.to(amp_a.device)
        phase_a = problem["target_phase"]
        phase_b = self.previous_target_phase.to(phase_a.device)

        amp_num = torch.sum(amp_a * amp_b)
        amp_den = torch.sqrt(torch.sum(amp_a.square()) * torch.sum(amp_b.square())).clamp_min(1e-9)
        amp_sim = amp_num / amp_den

        phase_diff = torch.atan2(torch.sin(phase_a - phase_b), torch.cos(phase_a - phase_b))
        phase_sim = 1.0 - torch.mean(torch.abs(phase_diff)) / torch.pi
        score = 0.7 * amp_sim + 0.3 * phase_sim
        return float(score.detach().cpu())

    def _mix_ratio(self, similarity: float) -> float:
        if similarity < self.cfg.warm_start_similarity_threshold:
            return 0.0
        span = max(1e-9, 1.0 - self.cfg.warm_start_similarity_threshold)
        norm = min(1.0, max(0.0, (similarity - self.cfg.warm_start_similarity_threshold) / span))
        return self.cfg.warm_start_min_mix + norm * (self.cfg.warm_start_max_mix - self.cfg.warm_start_min_mix)

    def _estimate_overlap(self, problem: dict[str, torch.Tensor], phase: torch.Tensor) -> float:
        out_amp, out_phase = self.pipeline.propagator(problem["beam"], phase)
        overlap = normalized_overlap(
            problem["target_amp"],
            problem["target_phase"],
            out_amp,
            out_phase,
            problem["weight"],
        )
        return float(overlap.detach().cpu())

    def _estimate_init_score(self, problem: dict[str, torch.Tensor], phase: torch.Tensor) -> float:
        out_amp, out_phase = self.pipeline.propagator(problem["beam"], phase)
        _, metrics = composite_loss(
            target_amp=problem["target_amp"],
            target_phase=problem["target_phase"],
            out_amp=out_amp,
            out_phase=out_phase,
            weight=problem["weight"],
            slm_phase=phase,
            overlap_weight=0.0,
            intensity_weight=0.0,
            phase_weight=0.0,
            uniformity_weight=0.0,
            efficiency_weight=0.0,
            core_uniformity_weight=0.0,
            core_phase_weight=0.0,
            core_region_threshold=self.cfg.core_region_threshold,
            smoothness_weight=0.0,
        )
        score = (
            self.cfg.init_score_core_uniformity_weight * metrics["core_uniformity_loss"]
            + self.cfg.init_score_core_phase_weight * metrics["core_phase_flatness"]
            + self.cfg.init_score_efficiency_weight * (1.0 - metrics["efficiency"])
            + self.cfg.init_score_uniformity_weight * metrics["uniformity_loss"]
        )
        return float(score)

    def _build_init_phase(self, problem: dict[str, torch.Tensor], use_warm_start: bool) -> tuple[torch.Tensor, float, float]:
        sources = set(self.cfg.init_candidate_sources)
        neural_init = self.pipeline.predict_initial_phase(
            problem["target_amp"], problem["target_phase"], problem["weight"]
        )
        candidate_phases: dict[str, torch.Tensor] = {}
        best_phase: torch.Tensor | None = None
        best_overlap = float("-inf")
        best_score = float("inf")
        best_mix = 0.0
        best_source = "neural"
        if "neural" in sources:
            candidate_phases["neural"] = neural_init
            best_phase = neural_init
            best_overlap = self._estimate_overlap(problem, neural_init)
            best_score = self._estimate_init_score(problem, neural_init)

        if "reference" in sources and self.cfg.reference_phase_path is not None and Path(self.cfg.reference_phase_path).exists():
            ref_phase = load_reference_phase(
                self.cfg.reference_phase_path,
                size=(self.cfg.slm_size, self.cfg.slm_size),
                device=self.cfg.device,
            )
            ref_blend = self.cfg.reference_phase_mix * ref_phase + (1.0 - self.cfg.reference_phase_mix) * neural_init
            candidate_phases["reference_blend"] = ref_blend
            ref_overlap = self._estimate_overlap(problem, ref_blend)
            ref_score = self._estimate_init_score(problem, ref_blend)
            if ref_score < best_score:
                best_phase = ref_blend
                best_overlap = ref_overlap
                best_score = ref_score
                best_mix = self.cfg.reference_phase_mix
                best_source = "reference_blend"

        ref_phase = None
        ref_blend = None
        if self.cfg.reference_phase_path is not None and Path(self.cfg.reference_phase_path).exists():
            ref_phase = load_reference_phase(
                self.cfg.reference_phase_path,
                size=(self.cfg.slm_size, self.cfg.slm_size),
                device=self.cfg.device,
            )
            ref_blend = self.cfg.reference_phase_mix * ref_phase + (1.0 - self.cfg.reference_phase_mix) * neural_init

        if "lin2025" in sources and self.cfg.target_type in {"target_flat_top", "target_round_top"}:
            lin_cfg = Lin2025Config(
                target_mode="flat_top",
                slm_size=self.cfg.slm_size,
                device=self.cfg.device,
                checkpoint_dir=self.cfg.lin2025_checkpoint_dir,
            )
            lin_phase, lin_loaded = predict_hologram_phase(
                lin_cfg,
                target_amp=problem["target_amp"],
                target_phase=problem["target_phase"],
                checkpoint=self.cfg.lin2025_checkpoint,
            )
            candidate_phases["lin2025"] = lin_phase
            lin_overlap = self._estimate_overlap(problem, lin_phase)
            lin_score = self._estimate_init_score(problem, lin_phase)
            if lin_score < best_score:
                best_phase = lin_phase
                best_overlap = lin_overlap
                best_score = lin_score
                best_mix = 0.0
                best_source = "lin2025"
            if self.cfg.lin2025_primary and ref_phase is not None:
                primary_ref_mix = self.cfg.lin2025_reference_weak_mix * ref_phase + (1.0 - self.cfg.lin2025_reference_weak_mix) * lin_phase
                candidate_phases["lin2025_reference_weak_blend"] = primary_ref_mix
                primary_ref_overlap = self._estimate_overlap(problem, primary_ref_mix)
                primary_ref_score = self._estimate_init_score(problem, primary_ref_mix)
                if primary_ref_score <= best_score:
                    best_phase = primary_ref_mix
                    best_overlap = primary_ref_overlap
                    best_score = primary_ref_score
                    best_mix = self.cfg.lin2025_reference_weak_mix
                    best_source = "lin2025_reference_weak_blend"
            lin_blend = self.cfg.lin2025_mix * lin_phase + (1.0 - self.cfg.lin2025_mix) * neural_init
            candidate_phases["lin2025_neural_blend"] = lin_blend
            lin_blend_overlap = self._estimate_overlap(problem, lin_blend)
            lin_blend_score = self._estimate_init_score(problem, lin_blend)
            if not self.cfg.lin2025_primary and lin_blend_score < best_score:
                best_phase = lin_blend
                best_overlap = lin_blend_overlap
                best_score = lin_blend_score
                best_mix = self.cfg.lin2025_mix
                best_source = "lin2025_neural_blend"
            if ref_phase is not None:
                ref_lin_blend = 0.5 * ref_phase + 0.5 * lin_phase
                candidate_phases["reference_lin2025_blend"] = ref_lin_blend
                ref_lin_overlap = self._estimate_overlap(problem, ref_lin_blend)
                ref_lin_score = self._estimate_init_score(problem, ref_lin_blend)
                if not self.cfg.lin2025_primary and ref_lin_score < best_score:
                    best_phase = ref_lin_blend
                    best_overlap = ref_lin_overlap
                    best_score = ref_lin_score
                    best_mix = 0.5
                    best_source = "reference_lin2025_blend"
            if lin_loaded is not None:
                self.pipeline.loaded_checkpoint = Path(lin_loaded)

        if use_warm_start and "warm_start" in sources and self.previous_phase is not None:
            similarity = self._target_similarity(problem)
            if similarity >= self.cfg.warm_start_candidate_threshold:
                prev_phase = self.previous_phase.to(neural_init.device)
                candidate_phases["warm_start"] = prev_phase
                if self.cfg.warm_start_allow_direct:
                    prev_overlap = self._estimate_overlap(problem, prev_phase)
                    prev_score = self._estimate_init_score(problem, prev_phase)
                    if prev_score < best_score:
                        best_phase = prev_phase
                        best_overlap = prev_overlap
                        best_score = prev_score
                        best_mix = 1.0
                        best_source = "warm_start"

                mix = self._mix_ratio(similarity)
                if mix > 0.0:
                    mix = max(mix, self.cfg.warm_start_external_mix_floor)
                    for source_name, source_phase in list(candidate_phases.items()):
                        if source_name == "warm_start":
                            continue
                        blended = mix * prev_phase + (1.0 - mix) * source_phase
                        blend_overlap = self._estimate_overlap(problem, blended)
                        blend_score = self._estimate_init_score(problem, blended)
                        if blend_score < best_score:
                            best_phase = blended
                            best_overlap = blend_overlap
                            best_score = blend_score
                            best_mix = mix
                            best_source = f"warm_start_{source_name}_blend"
                self.last_init_source = best_source
                return best_phase, best_overlap, best_mix

        if best_phase is None:
            best_phase = neural_init
            best_overlap = self._estimate_overlap(problem, neural_init)
            best_source = "neural_fallback"
        self.last_init_source = best_source
        return best_phase, best_overlap, best_mix

    def _phase_only_correct(
        self,
        problem: dict[str, torch.Tensor],
        init_phase: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if not self.cfg.phase_post_correction_enabled or "lin2025" not in self.cfg.init_candidate_sources:
            return init_phase, {
                "phase_post_correction_used": False,
                "phase_post_correction_steps": 0,
                "phase_post_correction_delta": 0.0,
            }

        base_phase = init_phase.to(self.pipeline.device)
        residual = nn.Parameter(torch.zeros_like(base_phase))
        optimizer = torch.optim.Adam([residual], lr=self.cfg.phase_post_correction_lr)
        best_phase = base_phase.detach().clone()
        best_loss = float("inf")
        best_metrics: dict[str, float] = {}

        for _ in range(self.cfg.phase_post_correction_steps):
            optimizer.zero_grad(set_to_none=True)
            phase = wrap_phase(base_phase + self.cfg.phase_post_correction_max_delta * torch.tanh(residual))
            out_amp, out_phase = self.pipeline.propagator(problem["beam"], phase)
            loss, metrics = composite_loss(
                target_amp=problem["target_amp"],
                target_phase=problem["target_phase"],
                out_amp=out_amp,
                out_phase=out_phase,
                weight=problem["weight"],
                slm_phase=phase,
                overlap_weight=0.0,
                intensity_weight=0.0,
                phase_weight=0.0,
                uniformity_weight=self.cfg.phase_post_correction_uniformity_weight,
                efficiency_weight=self.cfg.phase_post_correction_efficiency_weight,
                core_uniformity_weight=self.cfg.phase_post_correction_core_uniformity_weight,
                core_phase_weight=self.cfg.phase_post_correction_core_phase_weight,
                core_region_threshold=self.cfg.core_region_threshold,
                smoothness_weight=self.cfg.polish_smoothness_weight,
                efficiency_floor=self.cfg.efficiency_floor,
            )
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            if loss_value < best_loss:
                best_loss = loss_value
                best_phase = phase.detach().clone()
                best_metrics = metrics

        delta = torch.atan2(torch.sin(best_phase - base_phase), torch.cos(best_phase - base_phase))
        correction_metrics = {
            "phase_post_correction_used": True,
            "phase_post_correction_steps": self.cfg.phase_post_correction_steps,
            "phase_post_correction_delta": float(delta.abs().mean().detach().cpu()),
            "phase_post_correction_core_uniformity_loss": best_metrics.get("core_uniformity_loss", 0.0),
            "phase_post_correction_core_phase_flatness": best_metrics.get("core_phase_flatness", 0.0),
            "phase_post_correction_efficiency": best_metrics.get("efficiency", 0.0),
        }
        return best_phase, correction_metrics

    def run_once(
        self,
        use_warm_start: bool = True,
        save_subdir: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        start = time.perf_counter()
        problem = self.pipeline.build_problem()
        similarity = self._target_similarity(problem) if use_warm_start else 0.0
        init_phase, init_overlap, warm_mix = self._build_init_phase(problem, use_warm_start=use_warm_start)
        corrected_phase, correction_metrics = self._phase_only_correct(problem, init_phase)
        corrected_overlap = self._estimate_overlap(problem, corrected_phase)
        if self._estimate_init_score(problem, corrected_phase) <= self._estimate_init_score(problem, init_phase):
            init_phase = corrected_phase
            init_overlap = corrected_overlap

        if init_overlap >= self.cfg.fast_init_overlap_threshold:
            refined_phase = init_phase
            ai_metrics = {"overlap": init_overlap}
        else:
            refined_phase, ai_metrics, _, _ = self.pipeline.refine(
                problem["beam"],
                problem["target_amp"],
                problem["target_phase"],
                problem["weight"],
                init_phase,
            )

        if self.camera_loop.enabled:
            measured = load_measured_intensity(
                self.camera_loop.measured_intensity_path,
                expected_shape=tuple(problem["target_amp"].shape),
            )
            if measured is not None:
                correction = camera_feedback_weight(measured, problem["target_amp"].square())
                correction = correction.to(problem["weight"].device)
                problem["weight"] = problem["weight"] * correction / correction.max().clamp_min(1e-9)
                problem["target_amp"] = update_target_from_measurement(
                    previous_target_amp=problem["target_amp"],
                    measured_intensity=measured.to(problem["target_amp"].device),
                    gain=self.cfg.feedback_target_gain,
                )

        hybrid = bowman_cg_refine(
            self.cfg,
            problem,
            refined_phase,
            maxiter=self.cfg.hybrid_cg_maxiter,
            target_overlap=self.cfg.target_overlap,
        )
        hybrid["metrics"]["checkpoint"] = (
            str(self.pipeline.loaded_checkpoint) if self.pipeline.loaded_checkpoint is not None else None
        )
        hybrid["metrics"]["ai_overlap_before_cg"] = ai_metrics.get("overlap")
        hybrid["metrics"]["init_overlap"] = init_overlap
        hybrid["metrics"]["runtime_total_sec"] = time.perf_counter() - start
        hybrid["metrics"]["used_warm_start"] = bool(use_warm_start and self.previous_phase is not None and warm_mix > 0.0)
        hybrid["metrics"]["warm_start_similarity"] = similarity
        hybrid["metrics"]["warm_start_mix"] = warm_mix
        hybrid["metrics"]["init_source"] = self.last_init_source
        hybrid["metrics"]["skipped_ai_refine"] = bool(init_overlap >= self.cfg.fast_init_overlap_threshold)
        hybrid["metrics"]["camera_loop_enabled"] = self.camera_loop.enabled
        hybrid["metrics"].update(correction_metrics)
        if self.cfg.reference_phase_path is not None:
            hybrid["metrics"]["reference_phase_path"] = str(self.cfg.reference_phase_path)
            hybrid["metrics"]["reference_info"] = parse_795_reference_name(self.cfg.reference_phase_path)

        output_dir = self.cfg.output_dir if save_subdir is None else self.cfg.output_dir / save_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "slm_phase.npy", hybrid["phase"].numpy())
        (output_dir / "metrics.json").write_text(json.dumps(hybrid["metrics"], indent=2), encoding="utf-8")
        save_field_visualizations(
            output_dir,
            problem["target_amp"],
            problem["target_phase"],
            hybrid["out_amp"],
            hybrid["out_phase"],
            hybrid["phase"],
        )
        save_linecuts(output_dir, problem["target_amp"], hybrid["out_amp"])

        self.previous_phase = hybrid["phase"].clone()
        self.previous_target_amp = problem["target_amp"].detach().cpu().clone()
        self.previous_target_phase = problem["target_phase"].detach().cpu().clone()
        return hybrid["phase"], hybrid["metrics"]

    def run_closed_loop(
        self,
        measured_intensity_path: str | None = None,
        save_root: str = "closed_loop",
        synthesize_measurement_from_first_pass: bool = True,
    ) -> list[dict[str, float]]:
        records: list[dict[str, float]] = []

        # Round 1: open-loop generation
        phase0, metrics0 = self.run_once(use_warm_start=False, save_subdir=f"{save_root}_round0")
        records.append({"round": 0.0, **metrics0})

        output_dir0 = self.cfg.output_dir / f"{save_root}_round0"
        round0_metrics_path = output_dir0 / "metrics.json"
        round0_phase_path = output_dir0 / "slm_phase.npy"
        _ = phase0, round0_metrics_path, round0_phase_path

        # Use provided measured image or synthesize one from the first-pass simulation.
        measurement_file = measured_intensity_path
        if measurement_file is None and synthesize_measurement_from_first_pass:
            problem = self.pipeline.build_problem()
            phase_tensor = torch.tensor(np.load(output_dir0 / "slm_phase.npy"), dtype=torch.float32, device=self.pipeline.device)
            out_amp, _ = self.pipeline.propagator(problem["beam"], phase_tensor)
            measured_i = out_amp.square()
            measurement_file = str(self.cfg.output_dir / f"{save_root}_measured.npy")
            save_measured_intensity(measurement_file, measured_i)

        # Round 2: camera-feedback generation
        self.camera_loop.enabled = True
        self.camera_loop.measured_intensity_path = measurement_file
        phase1, metrics1 = self.run_once(use_warm_start=False, save_subdir=f"{save_root}_round1")
        records.append({"round": 1.0, **metrics1})
        self.camera_loop.enabled = False
        self.camera_loop.measured_intensity_path = None

        summary = {
            "round0": metrics0,
            "round1": metrics1,
            "delta": {
                "overlap": metrics1.get("overlap", 0.0) - metrics0.get("overlap", 0.0),
                "efficiency": metrics1.get("efficiency", 0.0) - metrics0.get("efficiency", 0.0),
                "uniformity_loss": metrics1.get("uniformity_loss", 0.0) - metrics0.get("uniformity_loss", 0.0),
                "phase_flatness": metrics1.get("phase_flatness", 0.0) - metrics0.get("phase_flatness", 0.0),
            },
            "measurement_file": measurement_file,
        }
        summary_path = self.cfg.output_dir / f"{save_root}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return records

    def run_sequence(self, count: int, save_root: str = "sequence") -> list[dict[str, float]]:
        metrics_list: list[dict[str, float]] = []
        for index in range(count):
            _, metrics = self.run_once(use_warm_start=True, save_subdir=f"{save_root}_{index:03d}")
            metrics_list.append(metrics)
        summary_path = self.cfg.output_dir / f"{save_root}_summary.json"
        summary_path.write_text(json.dumps(metrics_list, indent=2), encoding="utf-8")
        return metrics_list
