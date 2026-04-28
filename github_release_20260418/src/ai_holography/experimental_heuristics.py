from __future__ import annotations

import math

import torch


def focal_plane_pixel_pitch(
    wavelength_m: float,
    focal_length_m: float,
    output_pixels: int,
    slm_pixel_pitch_m: float,
    magnification: float = 1.0,
) -> float:
    return wavelength_m * focal_length_m / (output_pixels * slm_pixel_pitch_m * magnification)


def target_shift_pixels(
    dx_m: float,
    dy_m: float,
    wavelength_m: float,
    focal_length_m: float,
    output_shape: tuple[int, int],
    slm_pixel_pitch_m: float,
    magnification: float = 1.0,
) -> tuple[float, float]:
    pitch_y = focal_plane_pixel_pitch(
        wavelength_m=wavelength_m,
        focal_length_m=focal_length_m,
        output_pixels=output_shape[0],
        slm_pixel_pitch_m=slm_pixel_pitch_m,
        magnification=magnification,
    )
    pitch_x = focal_plane_pixel_pitch(
        wavelength_m=wavelength_m,
        focal_length_m=focal_length_m,
        output_pixels=output_shape[1],
        slm_pixel_pitch_m=slm_pixel_pitch_m,
        magnification=magnification,
    )
    return dx_m / pitch_x, dy_m / pitch_y


def physical_phase_guess(
    slm_shape: tuple[int, int],
    dx_m: float,
    dy_m: float,
    wavelength_m: float,
    focal_length_m: float,
    slm_pixel_pitch_m: float,
    magnification: float,
    aspect: float,
    curvature: float,
    cone: float,
    device: str,
) -> torch.Tensor:
    rows, cols = slm_shape
    x = torch.arange(cols, dtype=torch.float32, device=device) - cols / 2.0
    y = torch.arange(rows, dtype=torch.float32, device=device) - rows / 2.0
    X, Y = torch.meshgrid(x, y, indexing="xy")

    dx_px = dx_m / slm_pixel_pitch_m
    dy_px = dy_m / slm_pixel_pitch_m
    mu = math.atan2(dy_px, dx_px if abs(dx_px) > 1e-12 else 1e-12)
    distance_px = math.sqrt(dx_px * dx_px + dy_px * dy_px)
    tilt_scale = 2.0 * math.pi * slm_pixel_pitch_m / max(wavelength_m * focal_length_m, 1e-12)
    D = 4.0 * tilt_scale * distance_px
    ang = torch.tensor(mu, dtype=torch.float32, device=device)

    Xn = X / max(rows, cols)
    Yn = Y / max(rows, cols)
    linear = D * (Xn * torch.cos(ang) + Yn * torch.sin(ang))
    quad = 3.0 * curvature * (aspect * Xn.square() + (1.0 - aspect) * Yn.square())
    radial = cone * torch.sqrt(Xn.square() + Yn.square())
    return linear + quad + radial

