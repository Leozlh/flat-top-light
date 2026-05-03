from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ai_holography.losses import (
    core_phase_flatness_metric,
    core_phase_loss,
    core_uniformity_loss,
    efficiency_metric,
    normalized_overlap,
    phase_flatness_metric,
    target_core_weight,
    uniformity_loss,
)
from ai_holography.propagation import FourierSLM
from ai_holography.targets import gaussian_beam

from .model import position_to_hologram
from .wgs import crop_center


def amplitude_l1(pred_amp: torch.Tensor, target_amp: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred_amp - target_amp))


def phase_l2(pred_phase: torch.Tensor, target_phase: torch.Tensor) -> torch.Tensor:
    delta = torch.atan2(
        torch.sin((pred_phase - target_phase) * math.pi),
        torch.cos((pred_phase - target_phase) * math.pi),
    )
    return torch.mean(delta.square())


def wrapped_phase_l1(pred_phase_rad: torch.Tensor, target_phase_rad: torch.Tensor) -> torch.Tensor:
    delta = torch.atan2(
        torch.sin(pred_phase_rad - target_phase_rad),
        torch.cos(pred_phase_rad - target_phase_rad),
    )
    return torch.mean(torch.abs(delta))


def weighted_wrapped_phase_l1(
    pred_phase_rad: torch.Tensor,
    target_phase_rad: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    delta = torch.atan2(
        torch.sin(pred_phase_rad - target_phase_rad),
        torch.cos(pred_phase_rad - target_phase_rad),
    )
    weighted = torch.abs(delta) * weight
    denom = weight.sum().clamp_min(1e-9)
    return weighted.sum() / denom


def hologram_field_error(pred_amp: torch.Tensor, pred_phase: torch.Tensor, target_amp: torch.Tensor, target_phase: torch.Tensor) -> torch.Tensor:
    pred_holo = position_to_hologram(pred_amp, pred_phase)
    target_holo = position_to_hologram(target_amp, target_phase)
    pred_crop = crop_center(pred_holo, pred_amp.shape[-1])
    target_crop = crop_center(target_holo, target_amp.shape[-1])
    return torch.mean(torch.abs(pred_crop - target_crop))


def soft_efficiency_constraint(efficiency: torch.Tensor, floor: float) -> torch.Tensor:
    margin = torch.relu(torch.tensor(floor, device=efficiency.device, dtype=efficiency.dtype) - efficiency)
    return margin.square()


def masked_intensity_tv_loss(out_amp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    out_intensity = out_amp.square()
    norm = out_intensity.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-9)
    out_intensity = out_intensity / norm
    masked = out_intensity * weight
    dy = (masked[..., 1:, :] - masked[..., :-1, :]) * weight[..., 1:, :] * weight[..., :-1, :]
    dx = (masked[..., :, 1:] - masked[..., :, :-1]) * weight[..., :, 1:] * weight[..., :, :-1]
    return dx.abs().mean() + dy.abs().mean()


def flat_top_metric_loss(
    pred_amp: torch.Tensor,
    pred_phase: torch.Tensor,
    target_amp: torch.Tensor,
    target_phase: torch.Tensor,
    weight: torch.Tensor,
    overlap_weight: float,
    uniformity_weight: float,
    core_uniformity_weight: float,
    efficiency_weight: float,
    phase_weight: float,
    core_phase_weight: float,
    intensity_tv_weight: float,
    core_threshold: float,
    efficiency_floor: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_phase_rad = target_phase * math.pi
    pred_phase_rad = pred_phase * math.pi
    overlap = normalized_overlap(target_amp, target_phase_rad, pred_amp, pred_phase_rad, weight)
    overlap_loss = (1.0 - overlap).square()
    uniformity = uniformity_loss(target_amp, pred_amp, weight)
    core_uniformity = core_uniformity_loss(target_amp, pred_amp, weight, core_threshold)
    core_weight = target_core_weight(target_amp, weight, core_threshold)
    efficiency = efficiency_metric(weight, pred_amp)
    efficiency_constraint = soft_efficiency_constraint(efficiency, efficiency_floor)
    phase_loss = core_phase_loss(target_phase_rad, pred_phase_rad, target_amp, weight, core_threshold)
    phase_flatness = phase_flatness_metric(target_phase_rad, pred_phase_rad, weight)
    core_phase_flatness = core_phase_flatness_metric(target_phase_rad, pred_phase_rad, target_amp, weight, core_threshold)
    intensity_tv = masked_intensity_tv_loss(pred_amp, core_weight)
    total = (
        overlap_weight * overlap_loss
        + uniformity_weight * uniformity
        + core_uniformity_weight * core_uniformity
        + efficiency_weight * efficiency_constraint
        + phase_weight * phase_flatness
        + core_phase_weight * phase_loss
        + intensity_tv_weight * intensity_tv
    )
    metrics = {
        "overlap": float(overlap.detach().cpu()),
        "uniformity_loss": float(uniformity.detach().cpu()),
        "core_uniformity_loss": float(core_uniformity.detach().cpu()),
        "efficiency": float(efficiency.detach().cpu()),
        "efficiency_constraint": float(efficiency_constraint.detach().cpu()),
        "phase_flatness": float(phase_flatness.detach().cpu()),
        "core_phase_flatness": float(core_phase_flatness.detach().cpu()),
        "intensity_tv_loss": float(intensity_tv.detach().cpu()),
    }
    return total, metrics


def hybrid_init_score(
    pred_hologram_phase: torch.Tensor,
    target_amp_small: torch.Tensor,
    target_phase_small: torch.Tensor,
    beam_sigma_px: float,
    overlap_weight: float,
    uniformity_weight: float,
    core_uniformity_weight: float,
    efficiency_weight: float,
    core_phase_weight: float,
    core_threshold: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch = pred_hologram_phase.shape[0]
    slm_size = pred_hologram_phase.shape[-1]
    out_size = 2 * slm_size
    device = pred_hologram_phase.device
    propagator = FourierSLM(slm_size, out_size).to(device)
    beam_2d = gaussian_beam(slm_size, beam_sigma_px, beam_sigma_px, str(device))
    beam = beam_2d.unsqueeze(0).expand(batch, -1, -1)
    target_amp = F.interpolate(target_amp_small.unsqueeze(1), size=(out_size, out_size), mode="bilinear", align_corners=False).squeeze(1)
    target_phase = F.interpolate(target_phase_small.unsqueeze(1), size=(out_size, out_size), mode="bilinear", align_corners=False).squeeze(1) * math.pi
    weight = (target_amp > 0.05 * target_amp.amax(dim=(-2, -1), keepdim=True)).to(target_amp.dtype)
    out_amp, out_phase = propagator(beam, pred_hologram_phase)
    score, metrics = flat_top_metric_loss(
        pred_amp=out_amp,
        pred_phase=out_phase / math.pi,
        target_amp=target_amp,
        target_phase=target_phase / math.pi,
        weight=weight,
        overlap_weight=overlap_weight,
        uniformity_weight=uniformity_weight,
        core_uniformity_weight=core_uniformity_weight,
        efficiency_weight=efficiency_weight,
        phase_weight=0.0,
        core_phase_weight=core_phase_weight,
        intensity_tv_weight=0.0,
        core_threshold=core_threshold,
    )
    return score, metrics
