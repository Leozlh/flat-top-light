from __future__ import annotations

import math

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class Lin2025HologramNet(nn.Module):
    """
    Position-domain predictor inspired by Lin et al. 2025.
    Input: amplitude/phase images in the position domain.
    Output: amplitude/phase labels in the position domain.
    """

    def __init__(
        self,
        input_channels: int = 4,
        hidden_channels: int = 48,
        num_blocks: int = 4,
        phase_representation: str = "phasor",
    ):
        super().__init__()
        if phase_representation not in {"phasor", "scalar"}:
            raise ValueError(f"Unsupported phase_representation: {phase_representation}")
        self.phase_representation = phase_representation
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_channels) for _ in range(num_blocks)])
        self.amp_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.phase_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2 if phase_representation == "phasor" else 1, kernel_size=1),
        )

    def forward(
        self,
        a_input: torch.Tensor,
        phi_input: torch.Tensor,
        core_mask: torch.Tensor | None = None,
        roi_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if core_mask is None:
            core_mask = torch.zeros_like(a_input)
        if roi_mask is None:
            roi_mask = (a_input > 0).to(a_input.dtype)
        x = torch.stack([a_input, phi_input, core_mask, roi_mask], dim=1)
        features = self.blocks(self.stem(x))
        amp = torch.sigmoid(self.amp_head(features).squeeze(1))
        phase_features = self.phase_head(features)
        if self.phase_representation == "phasor":
            phase = torch.atan2(phase_features[:, 1], phase_features[:, 0]) / math.pi
        else:
            phase = torch.tanh(phase_features.squeeze(1))
        return amp, phase


def position_to_hologram(pred_amp: torch.Tensor, pred_phase: torch.Tensor) -> torch.Tensor:
    field = pred_amp * torch.exp(1j * (pred_phase * math.pi))
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field)))
