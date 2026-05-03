from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import HolographyConfig
from .experimental_heuristics import physical_phase_guess
from .losses import composite_loss
from .models import PhaseInitNet
from .propagation import FourierSLM, wrap_phase
from .references import (
    apply_795_reference_metadata,
    find_795_reference_files,
    load_reference_phase,
    parse_795_reference_name,
)
from .targets import build_phase, build_target, build_weight, gaussian_beam, quadratic_phase_guess, threshold_weight


class SyntheticBowmanDataset:
    """
    Generates synthetic Bowman-style LG targets on the fly.
    We stay inside the current target family instead of adding new target types.
    """

    def __init__(self, cfg: HolographyConfig, num_samples: int, seed: int = 42):
        self.cfg = cfg
        self.num_samples = num_samples
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_samples

    def _sample_center(self) -> tuple[float, float]:
        base = self.cfg.output_size / 3.0
        jitter = self.cfg.output_size * 0.04
        cy = base + self.rng.uniform(-jitter, jitter)
        cx = base + self.rng.uniform(-jitter, jitter)
        return cy, cx

    def _sample_sigma(self) -> float:
        return self.rng.uniform(max(3.0, self.cfg.target_sigma * 0.7), self.cfg.target_sigma * 1.3)

    def _sample_roi(self) -> tuple[float, float]:
        diameter = self.rng.uniform(max(16.0, self.cfg.roi_diameter * 0.8), self.cfg.roi_diameter * 1.2)
        softness = self.rng.uniform(max(1.0, self.cfg.roi_softness * 0.8), self.cfg.roi_softness * 1.3)
        return diameter, softness

    def _sample_target_type(self) -> str:
        target_types = self.cfg.training_target_types or (self.cfg.target_type,)
        return self.rng.choice(tuple(target_types))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        del idx
        device = self.cfg.device
        center = self._sample_center()
        sigma = self._sample_sigma()
        diameter, softness = self._sample_roi()
        target_type = self._sample_target_type()
        phase_type = "phase_flat" if target_type in {"target_flat_top", "target_round_top", "target_gaussian"} else self.cfg.phase_type

        beam = gaussian_beam(self.cfg.slm_size, self.cfg.beam_sigma_x, self.cfg.beam_sigma_y, device)
        target_amp = build_target(
            target_type=target_type,
            size=self.cfg.output_size,
            center=center,
            sigma=sigma,
            charge=self.cfg.vortex_charge,
            device=device,
        )
        target_phase = build_phase(
            phase_type=phase_type,
            size=self.cfg.output_size,
            center=center,
            device=device,
        )
        roi = build_weight(
            weight_type=self.cfg.weight_type,
            size=self.cfg.output_size,
            center=center,
            diameter=diameter,
            softness=softness,
            device=device,
        )
        weight = threshold_weight(roi, fraction=self.cfg.weight_threshold)
        target_amp = target_amp * weight
        target_amp = target_amp * torch.sqrt(beam.square().sum() / target_amp.square().sum().clamp_min(1e-9))
        return {
            "beam": beam,
            "target_amp": target_amp,
            "target_phase": target_phase,
            "weight": weight,
            "target_type": target_type,
        }


class ReferencePhaseDataset:
    """
    Builds supervised initialization samples from 795-style experimental references.
    These samples teach the network to imitate the reference phase before physics fine-tuning.
    """

    def __init__(self, cfg: HolographyConfig, seed: int = 42):
        self.cfg = cfg
        self.rng = random.Random(seed)
        ref_dir = Path(cfg.reference_dir)
        refs = find_795_reference_files(ref_dir)
        self.refs = [
            ref for ref in refs
            if parse_795_reference_name(ref).get("style") in {"round_top", "flat_top"}
        ]
        self.num_samples = max(len(self.refs), cfg.reference_augmented_samples)

    def __len__(self) -> int:
        return self.num_samples

    def _sample_augmentation(self) -> tuple[float, float, float, float]:
        shift_x = self.rng.uniform(-self.cfg.reference_shift_px, self.cfg.reference_shift_px)
        shift_y = self.rng.uniform(-self.cfg.reference_shift_px, self.cfg.reference_shift_px)
        scale = 1.0 + self.rng.uniform(-self.cfg.reference_scale_jitter, self.cfg.reference_scale_jitter)
        beam_scale = 1.0 + self.rng.uniform(-self.cfg.reference_beam_sigma_jitter, self.cfg.reference_beam_sigma_jitter)
        return shift_x, shift_y, scale, beam_scale

    def _warp_phase(self, phase: torch.Tensor, shift_x: float, shift_y: float, scale: float) -> torch.Tensor:
        theta = torch.tensor(
            [
                [1.0 / max(scale, 1e-6), 0.0, -2.0 * shift_x / max(self.cfg.slm_size, 1)],
                [0.0, 1.0 / max(scale, 1e-6), -2.0 * shift_y / max(self.cfg.slm_size, 1)],
            ],
            dtype=torch.float32,
            device=phase.device,
        ).unsqueeze(0)
        grid = F.affine_grid(theta, size=(1, 1, phase.shape[0], phase.shape[1]), align_corners=False)
        cos_phase = torch.cos(phase).unsqueeze(0).unsqueeze(0)
        sin_phase = torch.sin(phase).unsqueeze(0).unsqueeze(0)
        warped_cos = F.grid_sample(cos_phase, grid, mode="bilinear", padding_mode="border", align_corners=False)
        warped_sin = F.grid_sample(sin_phase, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return torch.atan2(warped_sin.squeeze(0).squeeze(0), warped_cos.squeeze(0).squeeze(0))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ref = self.refs[idx % len(self.refs)]
        info = parse_795_reference_name(ref)
        local_cfg = copy.deepcopy(self.cfg)
        local_cfg = apply_795_reference_metadata(local_cfg, info)
        local_cfg.phase_type = "phase_flat"
        shift_x, shift_y, scale, beam_scale = self._sample_augmentation()
        if local_cfg.target_center_x is None:
            local_cfg.target_center_x = local_cfg.output_size / 2.0
        if local_cfg.target_center_y is None:
            local_cfg.target_center_y = local_cfg.output_size / 2.0
        local_cfg.target_center_x += shift_x
        local_cfg.target_center_y += shift_y
        local_cfg.target_sigma = max(4.0, local_cfg.target_sigma * scale)
        local_cfg.roi_diameter = max(4.0, local_cfg.roi_diameter * scale)
        local_cfg.beam_sigma_x = max(4.0, local_cfg.beam_sigma_x * beam_scale)
        local_cfg.beam_sigma_y = max(4.0, local_cfg.beam_sigma_y * beam_scale)

        beam = gaussian_beam(local_cfg.slm_size, local_cfg.beam_sigma_x, local_cfg.beam_sigma_y, local_cfg.device)
        center = (
            local_cfg.target_center_y if local_cfg.target_center_y is not None else local_cfg.output_size / 2.0,
            local_cfg.target_center_x if local_cfg.target_center_x is not None else local_cfg.output_size / 2.0,
        )
        target_amp = build_target(
            target_type=local_cfg.target_type,
            size=local_cfg.output_size,
            center=center,
            sigma=local_cfg.target_sigma,
            charge=local_cfg.vortex_charge,
            device=local_cfg.device,
        )
        target_phase = build_phase(
            phase_type=local_cfg.phase_type,
            size=local_cfg.output_size,
            center=center,
            device=local_cfg.device,
        )
        roi = build_weight(
            weight_type=local_cfg.weight_type,
            size=local_cfg.output_size,
            center=center,
            diameter=local_cfg.roi_diameter,
            softness=local_cfg.roi_softness,
            device=local_cfg.device,
        )
        weight = threshold_weight(roi, fraction=local_cfg.weight_threshold)
        target_amp = target_amp * weight
        target_amp = target_amp * torch.sqrt(beam.square().sum() / target_amp.square().sum().clamp_min(1e-9))
        reference_phase = self._warp_phase(
            load_reference_phase(ref, size=(local_cfg.slm_size, local_cfg.slm_size), device=local_cfg.device),
            shift_x=shift_x * (local_cfg.slm_size / max(local_cfg.output_size, 1)),
            shift_y=shift_y * (local_cfg.slm_size / max(local_cfg.output_size, 1)),
            scale=scale,
        )
        return {
            "beam": beam,
            "target_amp": target_amp,
            "target_phase": target_phase,
            "weight": weight,
            "reference_phase": reference_phase,
            "reference_path": str(ref),
            "target_type": local_cfg.target_type,
        }


def downsample_field(field: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(
        field.unsqueeze(0).unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)


def build_base_phase(cfg: HolographyConfig) -> torch.Tensor:
    return physical_phase_guess(
        slm_shape=(cfg.slm_size, cfg.slm_size),
        dx_m=cfg.target_shift_x_m,
        dy_m=cfg.target_shift_y_m,
        wavelength_m=cfg.wavelength_m,
        focal_length_m=cfg.focal_length_m,
        slm_pixel_pitch_m=cfg.slm_pixel_pitch_m,
        magnification=cfg.magnification,
        aspect=cfg.init_aspect,
        curvature=cfg.init_curvature,
        cone=cfg.init_cone,
        device=cfg.device,
    ) + quadratic_phase_guess(
        cfg.slm_size,
        tilt=cfg.init_tilt,
        aspect=cfg.init_aspect,
        curvature=cfg.init_curvature,
        angle=cfg.init_angle,
        cone=cfg.init_cone,
        device=cfg.device,
    )


def wrapped_phase_difference(pred_phase: torch.Tensor, ref_phase: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(pred_phase - ref_phase), torch.cos(pred_phase - ref_phase))


def resize_phase_field(field: torch.Tensor, size: int) -> torch.Tensor:
    resized = downsample_field(field, size)
    return wrap_phase(resized)


def train_phase_init_net(
    cfg: HolographyConfig,
    train_samples: int = 256,
    val_samples: int = 32,
    epochs: int = 8,
    lr: float = 1e-3,
    checkpoint_dir: str | Path | None = None,
    seed: int = 42,
) -> Path:
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device(cfg.device)
    checkpoint_dir = Path(checkpoint_dir or cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = PhaseInitNet(residual_scale=cfg.phase_residual_scale).to(device)
    propagator = FourierSLM(cfg.slm_size, cfg.output_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_set = SyntheticBowmanDataset(cfg, num_samples=train_samples, seed=seed)
    val_set = SyntheticBowmanDataset(cfg, num_samples=val_samples, seed=seed + 1)
    reference_set = ReferencePhaseDataset(cfg, seed=seed + 99) if cfg.enable_reference_pretrain else None

    best_val = float("inf")
    best_metric_score = float("inf")
    best_path = checkpoint_dir / "phase_init_best.pt"
    best_loss_path = checkpoint_dir / "phase_init_best_loss.pt"
    best_metric_path = checkpoint_dir / "phase_init_best_metrics.pt"
    best_round_metric_score = float("inf")
    best_flat_metric_score = float("inf")
    best_round_path = checkpoint_dir / "phase_init_best_round.pt"
    best_flat_path = checkpoint_dir / "phase_init_best_flat.pt"
    history: list[dict[str, float]] = []
    stale_epochs = 0

    def scheduled_weights(progress: float) -> dict[str, float]:
        ramp_start = cfg.schedule_ramp_start
        ramp_end = cfg.schedule_ramp_end
        if progress <= ramp_start:
            ramp = 0.0
        elif progress >= ramp_end:
            ramp = 1.0
        else:
            ramp = (progress - ramp_start) / max(1e-9, ramp_end - ramp_start)
        return {
            "overlap_weight": cfg.overlap_weight,
            "intensity_weight": cfg.intensity_weight,
            "phase_weight": cfg.phase_weight * (0.4 + 0.6 * ramp),
            "uniformity_weight": cfg.uniformity_weight * ramp,
            "efficiency_weight": cfg.efficiency_weight * ramp,
            "core_uniformity_weight": cfg.core_uniformity_weight * ramp,
            "core_phase_weight": cfg.core_phase_weight * ramp,
            "core_region_threshold": cfg.core_region_threshold,
            "smoothness_weight": cfg.smoothness_weight,
        }

    if reference_set is not None and len(reference_set) > 0 and cfg.reference_pretrain_epochs > 0:
        lowres_size = min(cfg.reference_pretrain_lowres_size, cfg.slm_size)
        lowres_output = 2 * lowres_size
        lowres_propagator = FourierSLM(lowres_size, lowres_output).to(device)
        for epoch in range(1, cfg.reference_pretrain_epochs + 1):
            model.train()
            phase_loss_sum = 0.0
            start = time.perf_counter()
            for i in range(len(reference_set)):
                sample = reference_set[i]
                target_amp_small = downsample_field(sample["target_amp"], cfg.slm_size)
                target_phase_small = downsample_field(sample["target_phase"], cfg.slm_size)
                weight_small = downsample_field(sample["weight"], cfg.slm_size)
                base_phase = build_base_phase(cfg)
                pred_phase = wrap_phase(model(target_amp_small, target_phase_small, weight_small, base_phase))
                phase_residual = wrapped_phase_difference(pred_phase, sample["reference_phase"])
                phase_loss = phase_residual.square().mean()
                tv_loss = torch.mean(torch.abs(pred_phase[:, 1:] - pred_phase[:, :-1])) + torch.mean(
                    torch.abs(pred_phase[1:, :] - pred_phase[:-1, :])
                )
                lowres_beam = downsample_field(sample["beam"], lowres_size)
                lowres_target_amp = downsample_field(sample["target_amp"], lowres_output)
                lowres_target_phase = downsample_field(sample["target_phase"], lowres_output)
                lowres_weight = downsample_field(sample["weight"], lowres_output)
                lowres_phase = resize_phase_field(pred_phase, lowres_size)
                lowres_out_amp, lowres_out_phase = lowres_propagator(lowres_beam, lowres_phase)
                physics_loss, _ = composite_loss(
                    target_amp=lowres_target_amp,
                    target_phase=lowres_target_phase,
                    out_amp=lowres_out_amp,
                    out_phase=lowres_out_phase,
                    weight=lowres_weight,
                    slm_phase=lowres_phase,
                    overlap_weight=0.6,
                    intensity_weight=0.15,
                    phase_weight=0.05,
                    uniformity_weight=0.1,
                    efficiency_weight=0.05,
                    core_uniformity_weight=0.2,
                    core_phase_weight=0.1,
                    core_region_threshold=cfg.core_region_threshold,
                    smoothness_weight=cfg.reference_tv_loss_weight,
                )
                loss = (
                    cfg.reference_phase_loss_weight * phase_loss
                    + cfg.reference_physics_loss_weight * physics_loss
                    + cfg.reference_tv_loss_weight * tv_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                phase_loss_sum += float(loss.detach().cpu())

            avg_loss = phase_loss_sum / max(len(reference_set), 1)
            epoch_time = time.perf_counter() - start
            history.append(
                {
                    "epoch": float(-epoch),
                    "stage": "reference_pretrain",
                    "train_loss": avg_loss,
                    "val_loss": avg_loss,
                    "epoch_sec": epoch_time,
                }
            )
            print(f"pretrain {epoch:02d} | loss={avg_loss:.6f} | time={epoch_time:.2f}s")
            latest_path = checkpoint_dir / "phase_init_latest.pt"
            torch.save(model.state_dict(), latest_path)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        start = time.perf_counter()

        for i in range(len(train_set)):
            sample = train_set[i]
            target_amp_small = downsample_field(sample["target_amp"], cfg.slm_size)
            target_phase_small = downsample_field(sample["target_phase"], cfg.slm_size)
            weight_small = downsample_field(sample["weight"], cfg.slm_size)
            base_phase = build_base_phase(cfg)

            pred_phase = model(target_amp_small, target_phase_small, weight_small, base_phase)
            out_amp, out_phase = propagator(sample["beam"], pred_phase)
            loss, train_metrics = composite_loss(
                target_amp=sample["target_amp"],
                target_phase=sample["target_phase"],
                out_amp=out_amp,
                out_phase=out_phase,
                weight=sample["weight"],
                slm_phase=pred_phase,
                **scheduled_weights(epoch / max(epochs, 1)),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu())

        model.eval()
        val_loss_sum = 0.0
        val_metrics_sum = {
            "uniformity_loss": 0.0,
            "core_uniformity_loss": 0.0,
            "core_phase_flatness": 0.0,
            "efficiency": 0.0,
            "overlap": 0.0,
        }
        val_by_type: dict[str, dict[str, float]] = {}
        with torch.no_grad():
            for i in range(len(val_set)):
                sample = val_set[i]
                target_amp_small = downsample_field(sample["target_amp"], cfg.slm_size)
                target_phase_small = downsample_field(sample["target_phase"], cfg.slm_size)
                weight_small = downsample_field(sample["weight"], cfg.slm_size)
                base_phase = build_base_phase(cfg)
                pred_phase = wrap_phase(model(target_amp_small, target_phase_small, weight_small, base_phase))
                out_amp, out_phase = propagator(sample["beam"], pred_phase)
                loss, val_metrics = composite_loss(
                    target_amp=sample["target_amp"],
                    target_phase=sample["target_phase"],
                    out_amp=out_amp,
                    out_phase=out_phase,
                    weight=sample["weight"],
                    slm_phase=pred_phase,
                    **scheduled_weights(epoch / max(epochs, 1)),
                )
                val_loss_sum += float(loss.detach().cpu())
                for key in val_metrics_sum:
                    val_metrics_sum[key] += float(val_metrics.get(key, 0.0))
                sample_type = str(sample.get("target_type", "unknown"))
                bucket = val_by_type.setdefault(sample_type, {"count": 0.0, **{k: 0.0 for k in val_metrics_sum}})
                bucket["count"] += 1.0
                for key in val_metrics_sum:
                    bucket[key] += float(val_metrics.get(key, 0.0))

        train_loss = train_loss_sum / max(len(train_set), 1)
        val_loss = val_loss_sum / max(len(val_set), 1)
        avg_val_metrics = {k: v / max(len(val_set), 1) for k, v in val_metrics_sum.items()}
        avg_val_by_type: dict[str, dict[str, float]] = {}
        for target_type, bucket in val_by_type.items():
            count = max(bucket["count"], 1.0)
            avg_val_by_type[target_type] = {
                key: bucket[key] / count for key in val_metrics_sum
            }

        def metric_from(metrics: dict[str, float], target_type: str | None = None) -> float:
            if target_type == "target_round_top":
                uniformity_weight = cfg.training_round_uniformity_metric_weight
                core_uniformity_weight = cfg.training_round_core_uniformity_metric_weight
                core_phase_weight = cfg.training_round_core_phase_metric_weight
                efficiency_weight = cfg.training_round_efficiency_metric_weight
            elif target_type == "target_flat_top":
                uniformity_weight = cfg.training_flat_uniformity_metric_weight
                core_uniformity_weight = cfg.training_flat_core_uniformity_metric_weight
                core_phase_weight = cfg.training_flat_core_phase_metric_weight
                efficiency_weight = cfg.training_flat_efficiency_metric_weight
            else:
                uniformity_weight = cfg.training_uniformity_metric_weight
                core_uniformity_weight = cfg.training_core_uniformity_metric_weight
                core_phase_weight = cfg.training_core_phase_metric_weight
                efficiency_weight = cfg.training_efficiency_metric_weight
            return (
                uniformity_weight * metrics["uniformity_loss"]
                + core_uniformity_weight * metrics["core_uniformity_loss"]
                + core_phase_weight * metrics["core_phase_flatness"]
                + efficiency_weight * (1.0 - metrics["efficiency"])
            )

        metric_score = (
            metric_from(avg_val_metrics)
        )
        round_metric_score = (
            metric_from(avg_val_by_type["target_round_top"], "target_round_top")
            if "target_round_top" in avg_val_by_type
            else None
        )
        flat_metric_score = (
            metric_from(avg_val_by_type["target_flat_top"], "target_flat_top")
            if "target_flat_top" in avg_val_by_type
            else None
        )
        epoch_time = time.perf_counter() - start
        history.append(
            {
                "epoch": float(epoch),
                "stage": "physics_finetune",
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metric_score": metric_score,
                "val_round_metric_score": round_metric_score,
                "val_flat_metric_score": flat_metric_score,
                **avg_val_metrics,
                "epoch_sec": epoch_time,
            }
        )
        print(
            f"epoch {epoch:02d} | train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | metric={metric_score:.6f} | time={epoch_time:.2f}s"
        )

        latest_path = checkpoint_dir / "phase_init_latest.pt"
        torch.save(model.state_dict(), latest_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_loss_path)
        if metric_score + cfg.training_metric_min_delta < best_metric_score:
            best_metric_score = metric_score
            stale_epochs = 0
            torch.save(model.state_dict(), best_metric_path)
            torch.save(model.state_dict(), best_path)
        else:
            stale_epochs += 1
        if round_metric_score is not None and round_metric_score + cfg.training_metric_min_delta < best_round_metric_score:
            best_round_metric_score = round_metric_score
            torch.save(model.state_dict(), best_round_path)
        if flat_metric_score is not None and flat_metric_score + cfg.training_metric_min_delta < best_flat_metric_score:
            best_flat_metric_score = flat_metric_score
            torch.save(model.state_dict(), best_flat_path)
        if stale_epochs >= cfg.training_metric_patience:
            break

    metadata = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "train_samples": train_samples,
        "val_samples": val_samples,
        "epochs": epochs,
        "lr": lr,
        "best_val_loss": best_val,
        "best_metric_score": best_metric_score,
        "best_round_metric_score": best_round_metric_score,
        "best_flat_metric_score": best_flat_metric_score,
        "best_checkpoint": str(best_path),
        "best_loss_checkpoint": str(best_loss_path),
        "best_metric_checkpoint": str(best_metric_path),
        "best_round_checkpoint": str(best_round_path),
        "best_flat_checkpoint": str(best_flat_path),
        "history": history,
    }
    (checkpoint_dir / "training_history.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return best_path
