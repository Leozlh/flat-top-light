import math
import torch


def normalized_overlap(
    target_amp: torch.Tensor,
    target_phase: torch.Tensor,
    out_amp: torch.Tensor,
    out_phase: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    numerator = torch.sum(target_amp * out_amp * weight * torch.cos(out_phase - target_phase))
    denominator = torch.sqrt(torch.sum(target_amp.square()) * torch.sum((out_amp * weight).square())).clamp_min(1e-9)
    return numerator / denominator


def weighted_intensity_loss(target_amp: torch.Tensor, out_amp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    target_intensity = target_amp.square()
    out_intensity = out_amp.square()
    diff = (target_intensity - out_intensity) * weight
    return diff.square().mean()


def target_core_weight(target_amp: torch.Tensor, weight: torch.Tensor, threshold: float) -> torch.Tensor:
    target_intensity = target_amp.square()
    if torch.max(target_intensity) <= 0:
        return torch.zeros_like(weight)
    core = (target_intensity / torch.max(target_intensity).clamp_min(1e-9)) >= threshold
    return weight * core.to(weight.dtype)


def wrapped_phase_loss(target_phase: torch.Tensor, out_phase: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    phase_error = torch.atan2(torch.sin(out_phase - target_phase), torch.cos(out_phase - target_phase))
    return (phase_error.square() * weight).mean()


def phase_flatness_metric(target_phase: torch.Tensor, out_phase: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    phase_error = torch.atan2(torch.sin(out_phase - target_phase), torch.cos(out_phase - target_phase))
    weighted = phase_error[weight > 0]
    if weighted.numel() == 0:
        return torch.tensor(0.0, device=out_phase.device)
    return weighted.std()


def uniformity_loss(target_amp: torch.Tensor, out_amp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    target_intensity = target_amp.square()
    out_intensity = out_amp.square()
    mask = weight > 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=out_amp.device)
    target_vals = target_intensity[mask]
    out_vals = out_intensity[mask]
    # Compare normalized local intensity to improve uniformity inside the active region.
    target_norm = target_vals / target_vals.mean().clamp_min(1e-9)
    out_norm = out_vals / out_vals.mean().clamp_min(1e-9)
    return (out_norm - target_norm).square().mean()


def core_uniformity_loss(
    target_amp: torch.Tensor,
    out_amp: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    return uniformity_loss(target_amp, out_amp, target_core_weight(target_amp, weight, threshold))


def core_phase_flatness_metric(
    target_phase: torch.Tensor,
    out_phase: torch.Tensor,
    target_amp: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    return phase_flatness_metric(target_phase, out_phase, target_core_weight(target_amp, weight, threshold))


def core_phase_loss(
    target_phase: torch.Tensor,
    out_phase: torch.Tensor,
    target_amp: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    return wrapped_phase_loss(target_phase, out_phase, target_core_weight(target_amp, weight, threshold))


def efficiency_metric(weight: torch.Tensor, out_amp: torch.Tensor) -> torch.Tensor:
    out_intensity = out_amp.square()
    useful = torch.sum(out_intensity * (weight > 0).to(out_intensity.dtype))
    total = torch.sum(out_intensity).clamp_min(1e-9)
    return useful / total


def efficiency_loss(weight: torch.Tensor, out_amp: torch.Tensor) -> torch.Tensor:
    return 1.0 - efficiency_metric(weight, out_amp)


def soft_efficiency_constraint(efficiency: torch.Tensor, floor: float) -> torch.Tensor:
    margin = torch.relu(torch.tensor(floor, device=efficiency.device, dtype=efficiency.dtype) - efficiency)
    return margin.square()


def efficiency_weighted_overlap_loss(
    target_amp: torch.Tensor,
    target_phase: torch.Tensor,
    out_amp: torch.Tensor,
    out_phase: torch.Tensor,
    weight: torch.Tensor,
    steepness: float = 2.0,
) -> torch.Tensor:
    overlap = normalized_overlap(target_amp, target_phase, out_amp, out_phase, weight)
    eff = efficiency_metric(weight, out_amp)
    return (1.0 - overlap).square() * torch.pow(1.0 / eff.clamp_min(1e-6), steepness)


def total_variation_loss(phase: torch.Tensor) -> torch.Tensor:
    dy = phase[1:, :] - phase[:-1, :]
    dx = phase[:, 1:] - phase[:, :-1]
    return dx.abs().mean() + dy.abs().mean()


def masked_intensity_tv_loss(
    target_amp: torch.Tensor,
    out_amp: torch.Tensor,
    weight: torch.Tensor,
    core_region_threshold: float,
) -> torch.Tensor:
    core_weight = target_core_weight(target_amp=target_amp, weight=weight, threshold=core_region_threshold)
    norm = out_amp.square()
    norm = norm / norm.max().clamp_min(1e-9)
    masked = norm * core_weight
    dy = (masked[1:, :] - masked[:-1, :]) * core_weight[1:, :] * core_weight[:-1, :]
    dx = (masked[:, 1:] - masked[:, :-1]) * core_weight[:, 1:] * core_weight[:, :-1]
    return dx.abs().mean() + dy.abs().mean()


def composite_loss(
    target_amp: torch.Tensor,
    target_phase: torch.Tensor,
    out_amp: torch.Tensor,
    out_phase: torch.Tensor,
    weight: torch.Tensor,
    slm_phase: torch.Tensor,
    overlap_weight: float,
    intensity_weight: float,
    phase_weight: float,
    smoothness_weight: float,
    uniformity_weight: float = 0.0,
    efficiency_weight: float = 0.0,
    core_uniformity_weight: float = 0.0,
    core_phase_weight: float = 0.0,
    core_region_threshold: float = 0.7,
    efficiency_floor: float | None = None,
    intensity_tv_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    overlap = normalized_overlap(target_amp, target_phase, out_amp, out_phase, weight)
    overlap_term = (1.0 - overlap).square()
    intensity_term = weighted_intensity_loss(target_amp, out_amp, weight)
    phase_term = wrapped_phase_loss(target_phase, out_phase, weight)
    flatness_term = phase_flatness_metric(target_phase, out_phase, weight)
    uniformity_term = uniformity_loss(target_amp, out_amp, weight)
    core_uniformity_term = core_uniformity_loss(target_amp, out_amp, weight, core_region_threshold)
    core_phase_term = core_phase_loss(target_phase, out_phase, target_amp, weight, core_region_threshold)
    core_flatness_term = core_phase_flatness_metric(target_phase, out_phase, target_amp, weight, core_region_threshold)
    efficiency = efficiency_metric(weight, out_amp)
    efficiency_term = 1.0 - efficiency
    efficiency_constraint = (
        soft_efficiency_constraint(efficiency, efficiency_floor)
        if efficiency_floor is not None
        else efficiency_term
    )
    smoothness_term = total_variation_loss(slm_phase)
    intensity_tv_term = masked_intensity_tv_loss(target_amp, out_amp, weight, core_region_threshold)

    total = (
        overlap_weight * overlap_term
        + intensity_weight * intensity_term
        + phase_weight * phase_term
        + uniformity_weight * uniformity_term
        + efficiency_weight * efficiency_constraint
        + core_uniformity_weight * core_uniformity_term
        + core_phase_weight * core_phase_term
        + smoothness_weight * smoothness_term
        + intensity_tv_weight * intensity_tv_term
    )
    metrics = {
        "overlap": float(overlap.detach().cpu()),
        "overlap_loss": float(overlap_term.detach().cpu()),
        "intensity_loss": float(intensity_term.detach().cpu()),
        "phase_loss": float(phase_term.detach().cpu()),
        "phase_flatness": float(flatness_term.detach().cpu()),
        "uniformity_loss": float(uniformity_term.detach().cpu()),
        "core_uniformity_loss": float(core_uniformity_term.detach().cpu()),
        "efficiency": float(efficiency.detach().cpu()),
        "efficiency_loss": float(efficiency_term.detach().cpu()),
        "efficiency_constraint": float(efficiency_constraint.detach().cpu()),
        "core_phase_loss": float(core_phase_term.detach().cpu()),
        "core_phase_flatness": float(core_flatness_term.detach().cpu()),
        "tv_loss": float(smoothness_term.detach().cpu()),
        "intensity_tv_loss": float(intensity_tv_term.detach().cpu()),
    }
    return total, metrics
