from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import HolographyConfig
from .calibration import load_phase_correction
from .experimental_heuristics import physical_phase_guess, target_shift_pixels
from .losses import composite_loss
from .models import PhaseInitNet
from .propagation import FourierSLM, wrap_phase
from .targets import (
    build_phase,
    build_target,
    build_weight,
    gaussian_beam,
    quadratic_phase_guess,
    threshold_weight,
)
from .visualization import save_field_visualizations, save_linecuts


class AIHolographyPipeline:
    def __init__(self, config: HolographyConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        self.propagator = FourierSLM(config.slm_size, config.output_size).to(self.device)
        self.phase_net = PhaseInitNet(residual_scale=config.phase_residual_scale).to(self.device)
        self.phase_correction = load_phase_correction(
            config.slm_phase_correction_path,
            size=config.slm_size,
            device=config.device,
        )
        checkpoint_path = self._resolve_checkpoint()
        self.loaded_checkpoint = checkpoint_path
        if checkpoint_path is not None:
            try:
                state = torch.load(checkpoint_path, map_location=self.device)
                self.phase_net.load_state_dict(state)
            except Exception:
                self.loaded_checkpoint = None

    def _resolve_checkpoint(self) -> Path | None:
        if self.cfg.checkpoint and Path(self.cfg.checkpoint).exists():
            return Path(self.cfg.checkpoint)
        if self.cfg.auto_load_best_checkpoint:
            preferred: list[Path] = []
            if self.cfg.target_type == "target_round_top":
                preferred.append(Path(self.cfg.checkpoint_dir) / "phase_init_best_round.pt")
            elif self.cfg.target_type == "target_flat_top":
                preferred.append(Path(self.cfg.checkpoint_dir) / "phase_init_best_flat.pt")
            preferred.extend(
                [
                    Path(self.cfg.checkpoint_dir) / "phase_init_best_metrics.pt",
                    Path(self.cfg.checkpoint_dir) / "phase_init_best.pt",
                ]
            )
            for best in preferred:
                if best.exists():
                    return best
        return None

    def build_problem(self) -> dict[str, torch.Tensor]:
        shift_x_px, shift_y_px = target_shift_pixels(
            dx_m=self.cfg.target_shift_x_m,
            dy_m=self.cfg.target_shift_y_m,
            wavelength_m=self.cfg.wavelength_m,
            focal_length_m=self.cfg.focal_length_m,
            output_shape=(self.cfg.output_size, self.cfg.output_size),
            slm_pixel_pitch_m=self.cfg.slm_pixel_pitch_m,
            magnification=self.cfg.magnification,
        )
        center = (
            self.cfg.target_center_y if self.cfg.target_center_y is not None else self.cfg.output_size / 2.0 + shift_y_px,
            self.cfg.target_center_x if self.cfg.target_center_x is not None else self.cfg.output_size / 2.0 + shift_x_px,
        )
        beam = gaussian_beam(self.cfg.slm_size, self.cfg.beam_sigma_x, self.cfg.beam_sigma_y, self.cfg.device)
        target_amp = build_target(
            target_type=self.cfg.target_type,
            size=self.cfg.output_size,
            center=center,
            sigma=self.cfg.target_sigma,
            charge=self.cfg.vortex_charge,
            device=self.cfg.device,
        )
        target_phase = build_phase(
            phase_type=self.cfg.phase_type,
            size=self.cfg.output_size,
            center=center,
            device=self.cfg.device,
        )
        roi = build_weight(
            weight_type=self.cfg.weight_type,
            size=self.cfg.output_size,
            center=center,
            diameter=self.cfg.roi_diameter,
            softness=self.cfg.roi_softness,
            device=self.cfg.device,
        )
        weight = threshold_weight(roi, fraction=self.cfg.weight_threshold)
        target_amp = target_amp * weight
        target_amp = target_amp * torch.sqrt(beam.square().sum() / target_amp.square().sum().clamp_min(1e-9))
        return {
            "beam": beam,
            "target_amp": target_amp,
            "target_phase": target_phase,
            "weight": weight,
        }

    def predict_initial_phase(self, target_amp: torch.Tensor, target_phase: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        heuristic = physical_phase_guess(
            slm_shape=(self.cfg.slm_size, self.cfg.slm_size),
            dx_m=self.cfg.target_shift_x_m,
            dy_m=self.cfg.target_shift_y_m,
            wavelength_m=self.cfg.wavelength_m,
            focal_length_m=self.cfg.focal_length_m,
            slm_pixel_pitch_m=self.cfg.slm_pixel_pitch_m,
            magnification=self.cfg.magnification,
            aspect=self.cfg.init_aspect,
            curvature=self.cfg.init_curvature,
            cone=self.cfg.init_cone,
            device=self.cfg.device,
        )
        heuristic = heuristic + quadratic_phase_guess(
            self.cfg.slm_size,
            tilt=self.cfg.init_tilt,
            aspect=self.cfg.init_aspect,
            curvature=self.cfg.init_curvature,
            angle=self.cfg.init_angle,
            cone=self.cfg.init_cone,
            device=self.cfg.device,
        )
        target_amp_small = self._resize_2d(target_amp, self.cfg.slm_size)
        target_phase_small = self._resize_2d(target_phase, self.cfg.slm_size)
        weight_small = self._resize_2d(weight, self.cfg.slm_size)
        with torch.no_grad():
            predicted = self.phase_net(target_amp_small, target_phase_small, weight_small, heuristic)
        if predicted.shape != heuristic.shape:
            predicted = F.interpolate(
                predicted.unsqueeze(0).unsqueeze(0),
                size=(self.cfg.slm_size, self.cfg.slm_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        return wrap_phase(predicted + self.phase_correction)

    def _scheduled_weights(self, progress: float) -> dict[str, float]:
        ramp_start = self.cfg.schedule_ramp_start
        ramp_end = self.cfg.schedule_ramp_end
        if progress <= ramp_start:
            ramp = 0.0
        elif progress >= ramp_end:
            ramp = 1.0
        else:
            ramp = (progress - ramp_start) / max(1e-9, ramp_end - ramp_start)
        return {
            "overlap_weight": self.cfg.overlap_weight,
            "intensity_weight": self.cfg.intensity_weight,
            "phase_weight": self.cfg.phase_weight * (0.4 + 0.6 * ramp),
            "uniformity_weight": self.cfg.uniformity_weight * ramp,
            "efficiency_weight": self.cfg.efficiency_weight * ramp,
            "core_uniformity_weight": self.cfg.core_uniformity_weight * ramp,
            "core_phase_weight": self.cfg.core_phase_weight * ramp,
            "core_region_threshold": self.cfg.core_region_threshold,
            "smoothness_weight": self.cfg.smoothness_weight,
        }

    def _resize_2d(self, field: torch.Tensor, size: int) -> torch.Tensor:
        return F.interpolate(
            field.unsqueeze(0).unsqueeze(0),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    def _resize_phase(self, phase: torch.Tensor, size: int) -> torch.Tensor:
        resized = self._resize_2d(phase, size)
        return wrap_phase(resized)

    def _run_adam_stage(
        self,
        beam: torch.Tensor,
        target_amp: torch.Tensor,
        target_phase: torch.Tensor,
        weight: torch.Tensor,
        init_phase: torch.Tensor,
        size: int,
        num_steps: int,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor]:
        phase = torch.nn.Parameter(self._resize_phase(init_phase, size))
        stage_beam = beam if beam.shape == (size, size) else self._resize_2d(beam, size)
        if size == self.cfg.slm_size:
            stage_output_size = self.cfg.output_size
            propagator = self.propagator
        else:
            stage_output_size = 2 * size
            propagator = FourierSLM(size, stage_output_size).to(self.device)
        stage_target_amp = target_amp if target_amp.shape == (stage_output_size, stage_output_size) else self._resize_2d(target_amp, stage_output_size)
        stage_target_phase = target_phase if target_phase.shape == (stage_output_size, stage_output_size) else self._resize_2d(target_phase, stage_output_size)
        stage_weight = weight if weight.shape == (stage_output_size, stage_output_size) else self._resize_2d(weight, stage_output_size)

        optimizer = torch.optim.AdamW([phase], lr=self.cfg.learning_rate)
        last_metrics: dict[str, float] = {}
        best_loss = float("inf")
        best_phase = phase.detach().clone()
        stale_steps = 0

        for step in range(num_steps):
            optimizer.zero_grad(set_to_none=True)
            out_amp, out_phase = propagator(stage_beam, phase)
            weights = self._scheduled_weights((step + 1) / max(num_steps, 1))
            loss, last_metrics = composite_loss(
                target_amp=stage_target_amp,
                target_phase=stage_target_phase,
                out_amp=out_amp,
                out_phase=out_phase,
                weight=stage_weight,
                slm_phase=phase,
                efficiency_floor=self.cfg.efficiency_floor,
                **weights,
            )
            loss.backward()
            optimizer.step()
            if self.cfg.adaptive_weighting:
                with torch.no_grad():
                    target_i = stage_target_amp.square().clamp_min(1e-9)
                    out_i = out_amp.square().clamp_min(1e-9)
                    ratio = torch.pow(target_i / out_i, self.cfg.adaptive_weighting_alpha)
                    ratio = torch.clamp(
                        ratio,
                        min=self.cfg.adaptive_weight_clip_min,
                        max=self.cfg.adaptive_weight_clip_max,
                    )
                    stage_weight = stage_weight * ratio
                    stage_weight = stage_weight / stage_weight.max().clamp_min(1e-9)
            current_loss = float(loss.detach().cpu())
            if current_loss + self.cfg.early_stop_min_delta < best_loss:
                best_loss = current_loss
                best_phase = phase.detach().clone()
                stale_steps = 0
            else:
                stale_steps += 1
            if stale_steps >= self.cfg.early_stop_patience:
                break

        final_phase = wrap_phase(best_phase)
        out_amp, out_phase = propagator(stage_beam, final_phase)
        _, last_metrics = composite_loss(
            target_amp=stage_target_amp,
            target_phase=stage_target_phase,
            out_amp=out_amp,
            out_phase=out_phase,
            weight=stage_weight,
            slm_phase=final_phase,
            efficiency_floor=self.cfg.efficiency_floor,
            **self._scheduled_weights(1.0),
        )
        return final_phase, last_metrics, out_amp.detach(), out_phase.detach()

    def refine(self, beam: torch.Tensor, target_amp: torch.Tensor, target_phase: torch.Tensor, weight: torch.Tensor, init_phase: torch.Tensor) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor]:
        phase = init_phase.clone()
        last_metrics: dict[str, float] = {}
        out_amp = out_phase = None

        for level, steps in zip(self.cfg.multiscale_levels, self.cfg.stage_iterations):
            phase, last_metrics, out_amp, out_phase = self._run_adam_stage(
                beam=beam,
                target_amp=target_amp,
                target_phase=target_phase,
                weight=weight,
                init_phase=phase,
                size=level,
                num_steps=steps,
            )

        if out_amp is None or out_phase is None:
            raise RuntimeError("Refinement failed to produce output fields.")

        if self.cfg.refine_with_lbfgs and not (
            self.cfg.skip_lbfgs_if_target_met and last_metrics.get("overlap", 0.0) >= self.cfg.target_overlap
        ):
            phase = torch.nn.Parameter(phase.clone())
            lbfgs = torch.optim.LBFGS([phase], max_iter=self.cfg.lbfgs_steps, line_search_fn="strong_wolfe")

            def closure() -> torch.Tensor:
                lbfgs.zero_grad(set_to_none=True)
                out_amp, out_phase = self.propagator(beam, phase)
                loss, _ = composite_loss(
                    target_amp=target_amp,
                    target_phase=target_phase,
                    out_amp=out_amp,
                    out_phase=out_phase,
                    weight=weight,
                    slm_phase=phase,
                    efficiency_floor=self.cfg.efficiency_floor,
                    **self._scheduled_weights(1.0),
                )
                loss.backward()
                return loss

            lbfgs.step(closure)
            out_amp, out_phase = self.propagator(beam, phase)
            _, last_metrics = composite_loss(
                target_amp=target_amp,
                target_phase=target_phase,
                out_amp=out_amp,
                out_phase=out_phase,
                weight=weight,
                slm_phase=phase,
                efficiency_floor=self.cfg.efficiency_floor,
                **self._scheduled_weights(1.0),
            )
            final_phase = wrap_phase(phase.detach())
        else:
            final_phase = wrap_phase(phase.detach() if isinstance(phase, torch.nn.Parameter) else phase.detach())
        out_amp, out_phase = self.propagator(beam, final_phase)
        return final_phase, last_metrics, out_amp.detach(), out_phase.detach()

    def save_outputs(
        self,
        phase: torch.Tensor,
        metrics: dict[str, float],
        target_amp: torch.Tensor,
        target_phase: torch.Tensor,
        out_amp: torch.Tensor,
        out_phase: torch.Tensor,
    ) -> None:
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.cfg.output_dir / "slm_phase.npy", phase.cpu().numpy())
        (self.cfg.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        save_field_visualizations(self.cfg.output_dir, target_amp, target_phase, out_amp, out_phase, phase)
        save_linecuts(self.cfg.output_dir, target_amp, out_amp)

    def run(self) -> tuple[torch.Tensor, dict[str, float]]:
        start = time.perf_counter()
        problem = self.build_problem()
        init_phase = self.predict_initial_phase(problem["target_amp"], problem["target_phase"], problem["weight"])
        phase, metrics, out_amp, out_phase = self.refine(
            problem["beam"],
            problem["target_amp"],
            problem["target_phase"],
            problem["weight"],
            init_phase,
        )
        metrics["runtime_sec"] = time.perf_counter() - start
        metrics["checkpoint"] = str(self.loaded_checkpoint) if self.loaded_checkpoint is not None else None
        self.save_outputs(phase, metrics, problem["target_amp"], problem["target_phase"], out_amp, out_phase)
        return phase, metrics


def main() -> None:
    cfg = HolographyConfig()
    pipeline = AIHolographyPipeline(cfg)
    _, metrics = pipeline.run()
    print("Saved AI holography outputs to", cfg.output_dir)
    print(metrics)


if __name__ == "__main__":
    main()
