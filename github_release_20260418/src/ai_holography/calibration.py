from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_phase_correction(path: str | Path | None, size: int, device: str) -> torch.Tensor:
    if path is None:
        return torch.zeros((size, size), dtype=torch.float32, device=device)
    path = Path(path)
    if not path.exists():
        return torch.zeros((size, size), dtype=torch.float32, device=device)
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        arr = np.loadtxt(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape != (size, size):
        raise ValueError(f"Correction phase shape {arr.shape} does not match expected {(size, size)}")
    return torch.tensor(arr, dtype=torch.float32, device=device)

