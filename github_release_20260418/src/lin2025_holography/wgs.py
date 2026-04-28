from __future__ import annotations

import math

import torch


def gaussian_beam(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    axis = torch.arange(size, dtype=torch.float32, device=device)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    c = size / 2.0
    beam = torch.exp(-2.0 * (((x - c) / sigma) ** 2 + ((y - c) / sigma) ** 2))
    return beam / beam.max().clamp_min(1e-9)


def make_target_field(
    coords: torch.Tensor,
    phases: torch.Tensor,
    size: int,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    axis = torch.arange(size, dtype=torch.float32, device=device)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    field = torch.zeros((size, size), dtype=torch.complex64, device=device)
    for idx in range(coords.shape[0]):
        cx, cy = coords[idx]
        amp = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2))
        field = field + amp * torch.exp(1j * phases[idx])
    return field


def weighted_gs_hologram(
    coords: torch.Tensor,
    phases: torch.Tensor,
    slm_size: int,
    oversampled_size: int,
    trap_sigma: float,
    beam_sigma: float,
    iterations: int,
    device: torch.device,
) -> torch.Tensor:
    scale = oversampled_size / slm_size
    coords_os = coords * scale
    target = make_target_field(coords_os, phases, oversampled_size, trap_sigma * scale, device)
    target_amp = torch.abs(target)
    target_phase = torch.angle(target)
    beam = gaussian_beam(oversampled_size, beam_sigma * scale, device)

    slm_phase = 2.0 * math.pi * torch.rand((oversampled_size, oversampled_size), device=device)
    weights = torch.ones((coords.shape[0],), dtype=torch.float32, device=device)

    for _ in range(iterations):
        slm_field = beam * torch.exp(1j * slm_phase)
        out_field = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(slm_field)))
        out_amp = torch.abs(out_field)
        out_phase = torch.angle(out_field)

        # Update per-trap weights from sampled trap amplitudes.
        sample_x = coords_os[:, 0].round().long().clamp(0, oversampled_size - 1)
        sample_y = coords_os[:, 1].round().long().clamp(0, oversampled_size - 1)
        measured = out_amp[sample_y, sample_x].clamp_min(1e-6)
        weights = weights * (measured.mean() / measured)

        weighted_target_amp = torch.zeros_like(target_amp)
        weighted_target_amp[sample_y, sample_x] = weights
        weighted_target_amp = torch.maximum(weighted_target_amp, target_amp * 0.05)

        constrained = weighted_target_amp * torch.exp(1j * target_phase)
        back = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(constrained)))
        slm_phase = torch.angle(back)

    final_field = beam * torch.exp(1j * slm_phase)
    return final_field


def crop_center(field: torch.Tensor, size: int) -> torch.Tensor:
    current = field.shape[-1]
    start = (current - size) // 2
    end = start + size
    return field[start:end, start:end]


def hologram_to_position_labels(hologram: torch.Tensor, label_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    cropped = crop_center(hologram, label_size)
    pos_field = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(cropped)))
    amp = torch.abs(pos_field)
    phase = torch.angle(pos_field)
    amp = amp / amp.max().clamp_min(1e-9)
    phase = phase / math.pi
    return amp.float(), phase.float()

