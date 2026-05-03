import math
import torch


def _meshgrid(size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.arange(size, dtype=torch.float32, device=device)
    return torch.meshgrid(axis, axis, indexing="ij")


def gaussian_beam(size: int, sigma_x: float, sigma_y: float, device: str) -> torch.Tensor:
    y, x = _meshgrid(size, device)
    cx = cy = size / 2.0
    beam = torch.exp(
        -2.0 * (((x - cx) / sigma_x) ** 2 + ((y - cy) / sigma_y) ** 2)
    )
    return beam / beam.square().sum().sqrt().clamp_min(1e-9)


def laguerre_gaussian_target(
    size: int,
    center: tuple[float, float],
    width: float,
    charge: int,
    device: str,
) -> torch.Tensor:
    y, x = _meshgrid(size, device)
    cy, cx = center
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    radial = torch.pow((r * math.sqrt(2.0) / width).clamp_min(1e-9), abs(charge))
    amplitude = (radial * torch.exp(-((r / width) ** 2)) * 2.0 * (r / width) ** 2) / width
    return amplitude


def gaussian_target(
    size: int,
    center: tuple[float, float],
    sigma_x: float,
    sigma_y: float,
    device: str,
) -> torch.Tensor:
    y, x = _meshgrid(size, device)
    cy, cx = center
    return torch.exp(-2.0 * (((x - cx) / sigma_x) ** 2 + ((y - cy) / sigma_y) ** 2))


def round_top_target(
    size: int,
    center: tuple[float, float],
    diameter: float,
    edge_softness: float,
    device: str,
) -> torch.Tensor:
    y, x = _meshgrid(size, device)
    cy, cx = center
    radius = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    transition = (radius - diameter / 2.0) / max(edge_softness, 1e-6)
    return torch.sigmoid(-transition)


def flat_top_target(
    size: int,
    center: tuple[float, float],
    diameter: float,
    order: float,
    edge_softness: float,
    device: str,
) -> torch.Tensor:
    y, x = _meshgrid(size, device)
    cy, cx = center
    radius = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    norm_r = radius / max(diameter / 2.0, 1e-6)
    soft_transition = torch.sigmoid(-(norm_r - 1.0) / max(edge_softness, 1e-6))
    super_gaussian = torch.exp(-torch.pow(norm_r, order))
    return soft_transition * super_gaussian


def vortex_phase(size: int, center: tuple[float, float], device: str) -> torch.Tensor:
    y, x = _meshgrid(size, device)
    cy, cx = center
    phase = torch.atan2(x - cx, y - cy)
    return torch.remainder(phase + math.pi, 2.0 * math.pi) - math.pi


def circular_weight(size: int, center: tuple[float, float], diameter: float, softness: float, device: str) -> torch.Tensor:
    y, x = _meshgrid(size, device)
    cy, cx = center
    radius = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    shell = 0.5 * (torch.abs(radius - diameter / 2.0) + torch.abs(radius + diameter / 2.0) - diameter)
    mask = torch.exp(-((shell / softness) ** 2))
    return torch.where(mask >= 1e-5, mask, torch.zeros_like(mask))


def threshold_weight(mask: torch.Tensor, fraction: float, fill: float = 0.0) -> torch.Tensor:
    threshold = fraction * mask.max().clamp_min(1e-9)
    return torch.where(mask.abs() < threshold, torch.full_like(mask, fill), torch.ones_like(mask))


def quadratic_phase_guess(size: int, tilt: float, aspect: float, curvature: float, angle: float, cone: float, device: str) -> torch.Tensor:
    y, x = _meshgrid(size, device)
    x = x - size / 2.0
    y = y - size / 2.0
    linear = tilt * (x * torch.cos(torch.tensor(angle, device=device)) + y * torch.sin(torch.tensor(angle, device=device)))
    quad = 3.0 * curvature * (aspect * x.square() + (1.0 - aspect) * y.square())
    radial = cone * torch.sqrt(x.square() + y.square())
    return linear + quad + radial


def flat_phase(size: int, device: str) -> torch.Tensor:
    return torch.zeros((size, size), dtype=torch.float32, device=device)


def build_target(target_type: str, size: int, center: tuple[float, float], sigma: float, charge: int, device: str) -> torch.Tensor:
    if target_type == "target_lg":
        return laguerre_gaussian_target(size=size, center=center, width=sigma, charge=charge, device=device)
    if target_type == "target_gaussian":
        return gaussian_target(size=size, center=center, sigma_x=sigma, sigma_y=sigma, device=device)
    if target_type == "target_round_top":
        return round_top_target(size=size, center=center, diameter=sigma, edge_softness=max(2.0, sigma * 0.08), device=device)
    if target_type == "target_flat_top":
        return flat_top_target(
            size=size,
            center=center,
            diameter=sigma,
            order=8.0,
            edge_softness=max(0.08, 0.12),
            device=device,
        )
    raise ValueError(f"Unsupported target_type: {target_type}")


def build_phase(phase_type: str, size: int, center: tuple[float, float], device: str) -> torch.Tensor:
    if phase_type == "phase_spinning_continuous":
        return vortex_phase(size=size, center=center, device=device)
    if phase_type == "phase_flat":
        return flat_phase(size=size, device=device)
    raise ValueError(f"Unsupported phase_type: {phase_type}")


def build_weight(weight_type: str, size: int, center: tuple[float, float], diameter: float, softness: float, device: str) -> torch.Tensor:
    if weight_type == "gaussian_top_round":
        return circular_weight(size=size, center=center, diameter=diameter, softness=softness, device=device)
    raise ValueError(f"Unsupported weight_type: {weight_type}")
