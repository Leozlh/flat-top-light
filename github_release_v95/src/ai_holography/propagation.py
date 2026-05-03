import math
import torch
import torch.nn.functional as F


class FourierSLM(torch.nn.Module):
    def __init__(self, slm_size: int, output_size: int):
        super().__init__()
        if output_size < slm_size:
            raise ValueError(f"output_size ({output_size}) must be >= slm_size ({slm_size}).")
        if (output_size - slm_size) % 2 != 0:
            raise ValueError(f"output_size - slm_size must be even for symmetric padding, got {output_size - slm_size}.")
        self.slm_size = slm_size
        self.output_size = output_size
        self.scale = 1.0 / output_size
        self.pad = (output_size - slm_size) // 2

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        field = self.scale * amplitude * torch.exp(1j * phase)
        padded = F.pad(field, (self.pad, self.pad, self.pad, self.pad))
        output = torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(padded)))
        intensity = output.abs().square()
        phase = torch.angle(output)
        amplitude = torch.sqrt(intensity.clamp_min(1e-12))
        return amplitude, phase


def wrap_phase(phase: torch.Tensor) -> torch.Tensor:
    return torch.remainder(phase, 2.0 * math.pi)

