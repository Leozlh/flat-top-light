from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ai_holography.visualization import save_field_visualizations, save_linecuts

from .config import Lin2025Config
from .dataset import create_dataset
from .metrics import amplitude_l1, flat_top_metric_loss, phase_l2
from .model import Lin2025HologramNet, position_to_hologram
from .wgs import crop_center


def run_lin2025_demo(cfg: Lin2025Config, checkpoint: Path | None = None) -> dict[str, float]:
    device = torch.device(cfg.device)
    model = Lin2025HologramNet(hidden_channels=cfg.hidden_channels).to(device)
    ckpt = checkpoint or (cfg.checkpoint_dir / "lin2025_best.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    sample = create_dataset(cfg, num_samples=1, seed=7)[0]
    with torch.no_grad():
        pred_amp, pred_phase = model(sample["a_input"].unsqueeze(0), sample["phi_input"].unsqueeze(0))
    pred_amp = pred_amp.squeeze(0)
    pred_phase = pred_phase.squeeze(0)

    pred_holo = position_to_hologram(pred_amp, pred_phase)
    label_holo = position_to_hologram(sample["a_label"], sample["phi_label"])
    pred_crop = crop_center(pred_holo, cfg.slm_size)
    label_crop = crop_center(label_holo, cfg.slm_size)

    metrics = {
        "amp_l1": float(amplitude_l1(pred_amp, sample["a_label"]).cpu()),
        "phase_l2": float(phase_l2(pred_phase, sample["phi_label"]).cpu()),
        "hologram_mae": float(torch.mean(torch.abs(pred_crop - label_crop)).cpu()),
        "checkpoint": str(ckpt),
    }
    if cfg.target_mode == "flat_top":
        weight = (sample["a_input"] > cfg.weight_threshold).to(sample["a_input"].dtype)
        _, flat_metrics = flat_top_metric_loss(
            pred_amp=pred_amp.unsqueeze(0),
            pred_phase=pred_phase.unsqueeze(0),
            target_amp=sample["a_input"].unsqueeze(0),
            target_phase=sample["phi_input"].unsqueeze(0),
            weight=weight.unsqueeze(0),
            overlap_weight=cfg.flat_overlap_weight,
            uniformity_weight=cfg.flat_uniformity_weight,
            core_uniformity_weight=cfg.flat_core_uniformity_weight,
            efficiency_weight=cfg.flat_efficiency_weight,
            phase_weight=cfg.flat_phase_weight,
            core_phase_weight=cfg.flat_core_phase_weight,
            core_threshold=cfg.flat_core_threshold,
        )
        metrics.update(flat_metrics)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.output_dir / "predicted_hologram_phase.npy", torch.angle(pred_crop).cpu().numpy())
    (cfg.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_field_visualizations(
        cfg.output_dir,
        target_amp=sample["a_label"],
        target_phase=sample["phi_label"] * torch.pi,
        out_amp=pred_amp,
        out_phase=pred_phase * torch.pi,
        slm_phase=torch.angle(pred_crop),
    )
    save_linecuts(cfg.output_dir, sample["a_label"], pred_amp)
    return metrics
