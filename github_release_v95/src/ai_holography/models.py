import math
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PhaseInitNet(nn.Module):
    """
    Predicts an initial SLM phase from target amplitude, target phase, and weighting.
    This is the easiest place to add "more AI" without giving up physical interpretability.
    """

    def __init__(self, in_channels: int = 4, hidden: int = 32, residual_scale: float = math.pi / 2.0):
        super().__init__()
        self.residual_scale = residual_scale
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, hidden),
            nn.AvgPool2d(2),
            ConvBlock(hidden, hidden * 2),
            nn.AvgPool2d(2),
            ConvBlock(hidden * 2, hidden * 4),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden * 4, hidden * 2, kernel_size=2, stride=2),
            nn.GELU(),
            ConvBlock(hidden * 2, hidden * 2),
            nn.ConvTranspose2d(hidden * 2, hidden, kernel_size=2, stride=2),
            nn.GELU(),
            ConvBlock(hidden, hidden),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(
        self,
        target_amp: torch.Tensor,
        target_phase: torch.Tensor,
        weight: torch.Tensor,
        base_phase: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.stack([target_amp, target_phase / math.pi, weight, base_phase / math.pi], dim=0).unsqueeze(0)
        residual = self.decoder(self.encoder(x)).squeeze(0).squeeze(0)
        return base_phase + self.residual_scale * torch.tanh(residual)
