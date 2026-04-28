from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import Lin2025Config
from .model import Lin2025HologramNet, position_to_hologram
from .wgs import crop_center
from ai_holography.losses import target_core_weight


def _resolve_model_shape(cfg: Lin2025Config, checkpoint: Path | None) -> tuple[int, int, int, str]:
    input_channels = cfg.input_channels
    hidden_channels = cfg.hidden_channels
    num_blocks = cfg.num_residual_blocks
    phase_representation = cfg.phase_representation
    candidate_dirs: list[Path] = []
    if checkpoint is not None:
        candidate_dirs.append(checkpoint.parent)
    candidate_dirs.append(cfg.checkpoint_dir)
    for directory in candidate_dirs:
        history_path = Path(directory) / "lin2025_training_history.json"
        if not history_path.exists():
            continue
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            config_data = history.get("config", {})
            input_channels = int(config_data.get("input_channels", input_channels))
            hidden_channels = int(config_data.get("hidden_channels", hidden_channels))
            num_blocks = int(config_data.get("num_residual_blocks", num_blocks))
            phase_representation = str(config_data.get("phase_representation", phase_representation))
            return input_channels, hidden_channels, num_blocks, phase_representation
        except Exception:
            continue
    return input_channels, hidden_channels, num_blocks, phase_representation


def load_lin2025_model(cfg: Lin2025Config, checkpoint: Path | None = None) -> tuple[Lin2025HologramNet, Path | None]:
    device = torch.device(cfg.device)
    resolved_checkpoint = checkpoint
    if resolved_checkpoint is None:
        resolved_checkpoint = cfg.checkpoint_dir / "lin2025_best_hybrid_polish.pt"
    input_channels, hidden_channels, num_blocks, phase_representation = _resolve_model_shape(cfg, resolved_checkpoint)
    ckpt = resolved_checkpoint
    if not ckpt.exists():
        fallback = cfg.checkpoint_dir / "lin2025_best_quality.pt"
        if fallback.exists():
            ckpt = fallback
        else:
            legacy = cfg.checkpoint_dir / "lin2025_best_init_for_hybrid.pt"
            ckpt = legacy if legacy.exists() else (cfg.checkpoint_dir / "lin2025_best.pt")
    loaded = ckpt if ckpt.exists() else None
    model = Lin2025HologramNet(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        phase_representation=phase_representation,
    ).to(device)
    if loaded is not None:
        state = torch.load(loaded, map_location=device)
        try:
            model.load_state_dict(state)
        except RuntimeError:
            fallback_phase_representation = "scalar" if phase_representation == "phasor" else "phasor"
            model = Lin2025HologramNet(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                num_blocks=num_blocks,
                phase_representation=fallback_phase_representation,
            ).to(device)
            model.load_state_dict(state)
    model.eval()
    return model, loaded


def predict_hologram_phase(
    cfg: Lin2025Config,
    target_amp: torch.Tensor,
    target_phase: torch.Tensor,
    checkpoint: Path | None = None,
) -> tuple[torch.Tensor, Path | None]:
    device = torch.device(cfg.device)
    model, loaded = load_lin2025_model(cfg, checkpoint=checkpoint)
    size = cfg.slm_size
    if target_amp.ndim == 2:
        target_amp = target_amp.unsqueeze(0)
    if target_phase.ndim == 2:
        target_phase = target_phase.unsqueeze(0)
    amp_small = F.interpolate(target_amp.unsqueeze(1), size=(size, size), mode="bilinear", align_corners=False).squeeze(1).to(device)
    phase_small = F.interpolate(target_phase.unsqueeze(1), size=(size, size), mode="bilinear", align_corners=False).squeeze(1).to(device)
    roi_mask = (amp_small > 0.05 * amp_small.amax(dim=(-2, -1), keepdim=True)).to(amp_small.dtype)
    core_mask = target_core_weight(amp_small, roi_mask, cfg.flat_core_threshold)
    with torch.no_grad():
        pred_amp, pred_phase = model(
            amp_small,
            phase_small / torch.pi,
            core_mask=core_mask,
            roi_mask=roi_mask,
        )
    hologram = position_to_hologram(pred_amp, pred_phase)
    crop = crop_center(hologram.squeeze(0), size)
    return torch.angle(crop), loaded
