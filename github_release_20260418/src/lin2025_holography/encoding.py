from __future__ import annotations

import math

import torch


def bilinear_scatter(
    coords: torch.Tensor,
    values: torch.Tensor,
    size: int,
    device: torch.device,
) -> torch.Tensor:
    image = torch.zeros((size, size), dtype=torch.float32, device=device)
    x = coords[:, 0].clamp(0, size - 1 - 1e-6)
    y = coords[:, 1].clamp(0, size - 1 - 1e-6)
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = torch.clamp(x0 + 1, max=size - 1)
    y1 = torch.clamp(y0 + 1, max=size - 1)
    wx = x - x0.float()
    wy = y - y0.float()

    image.index_put_((y0, x0), values * (1 - wx) * (1 - wy), accumulate=True)
    image.index_put_((y0, x1), values * wx * (1 - wy), accumulate=True)
    image.index_put_((y1, x0), values * (1 - wx) * wy, accumulate=True)
    image.index_put_((y1, x1), values * wx * wy, accumulate=True)
    return image


def encode_inputs(
    coords: torch.Tensor,
    phases: torch.Tensor,
    size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    amp = bilinear_scatter(coords, torch.ones_like(phases), size=size, device=device)
    phase_cos = bilinear_scatter(coords, torch.cos(phases), size=size, device=device)
    phase_sin = bilinear_scatter(coords, torch.sin(phases), size=size, device=device)
    phase = torch.atan2(phase_sin, phase_cos + 1e-9)
    if amp.max() > 0:
        amp = amp / amp.max()
    phase = phase / math.pi
    return amp, phase

