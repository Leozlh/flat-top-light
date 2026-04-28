from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class CameraLoopConfig:
    enabled: bool = False
    feedback_gain: float = 0.2
    max_rounds: int = 3
    measured_intensity_path: str | None = None


def load_measured_intensity(path: str | None, expected_shape: tuple[int, int]) -> torch.Tensor | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    if p.suffix == ".npy":
        arr = np.load(p)
    else:
        arr = np.loadtxt(p)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape != expected_shape:
        raise ValueError(f"Measured intensity shape {arr.shape} does not match expected {expected_shape}")
    return torch.tensor(arr, dtype=torch.float32)


def save_measured_intensity(path: str | Path, intensity: torch.Tensor) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, intensity.detach().cpu().numpy())


def camera_feedback_weight(measured_intensity: torch.Tensor, target_intensity: torch.Tensor) -> torch.Tensor:
    measured = measured_intensity / measured_intensity.mean().clamp_min(1e-9)
    target = target_intensity / target_intensity.mean().clamp_min(1e-9)
    ratio = target / measured.clamp_min(1e-6)
    return ratio


def update_target_from_measurement(
    previous_target_amp: torch.Tensor,
    measured_intensity: torch.Tensor,
    gain: float,
) -> torch.Tensor:
    prev_i = previous_target_amp.square()
    measured = measured_intensity / measured_intensity.max().clamp_min(1e-9)
    prev_norm = prev_i / prev_i.max().clamp_min(1e-9)
    correction = prev_norm - measured
    new_goal = torch.clamp(prev_norm + gain * correction, min=0.0)
    return torch.sqrt(new_goal)
