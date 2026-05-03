from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REFERENCE_RE = re.compile(
    r"^795_(?P<style>[a-z_]+)_d=(?P<d>-?\d+)_dx=(?P<dx>-?\d+)_dy=(?P<dy>-?\d+)\.npy$"
)


def parse_795_reference_name(path: str | Path) -> dict[str, int | str | None]:
    name = Path(path).name
    match = REFERENCE_RE.match(name)
    if not match:
        return {"style": None, "d": None, "dx": None, "dy": None}
    out: dict[str, int | str | None] = match.groupdict()
    for key in ("d", "dx", "dy"):
        out[key] = int(out[key]) if out[key] is not None else None
    return out


def load_reference_phase(path: str | Path, size: tuple[int, int], device: str) -> torch.Tensor:
    arr = np.load(Path(path))
    phase = torch.tensor(arr, dtype=torch.float32, device=device)
    if phase.shape != size:
        complex_field = torch.exp(1j * phase)
        resized = F.interpolate(
            complex_field.unsqueeze(0).unsqueeze(0).real,
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        resized_imag = F.interpolate(
            complex_field.unsqueeze(0).unsqueeze(0).imag,
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        phase = torch.angle(torch.complex(resized, resized_imag))
    return phase


def find_795_reference_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(folder.glob("795*.npy"))


def apply_795_reference_metadata(cfg, info: dict[str, int | str | None]):
    style = info.get("style")
    d = info.get("d")
    dx = info.get("dx")
    dy = info.get("dy")
    scale = cfg.output_size / max(cfg.reference_output_size, 1)

    if dx is not None:
        cfg.target_center_x = cfg.output_size / 2.0 + dx * scale
        cfg.target_shift_x_m = 0.0
    if dy is not None:
        cfg.target_center_y = cfg.output_size / 2.0 + dy * scale
        cfg.target_shift_y_m = 0.0
    if d is not None:
        scaled_d = max(4.0, d * scale)
        cfg.roi_diameter = scaled_d
        cfg.target_sigma = scaled_d

    if style == "gaussian":
        cfg.target_type = "target_gaussian"
        cfg.weight_type = "gaussian_top_round"
        cfg.phase_type = "phase_flat"
        cfg.uniformity_weight = 0.05
        cfg.efficiency_weight = 0.2
    elif style == "round_top":
        cfg.target_type = "target_round_top"
        cfg.weight_type = "gaussian_top_round"
        cfg.phase_type = "phase_flat"
        cfg.uniformity_weight = 0.2
        cfg.efficiency_weight = 0.2
    elif style == "flat_top":
        cfg.target_type = "target_flat_top"
        cfg.weight_type = "gaussian_top_round"
        cfg.phase_type = "phase_flat"
        cfg.uniformity_weight = 0.35
        cfg.efficiency_weight = 0.15
    return cfg
