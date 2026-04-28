from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import Lin2025Config
from .dataset import create_dataset
from .metrics import (
    amplitude_l1,
    flat_top_metric_loss,
    hologram_field_error,
    hybrid_init_score,
    phase_l2,
    soft_efficiency_constraint,
    weighted_wrapped_phase_l1,
)
from .model import Lin2025HologramNet, position_to_hologram


def _batchify(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = ("a_input", "phi_input", "a_label", "phi_label")
    batch = {key: torch.stack([sample[key] for sample in samples], dim=0) for key in keys}
    if "teacher_phase" in samples[0]:
        batch["teacher_phase"] = torch.stack([sample["teacher_phase"] for sample in samples], dim=0)
    if "core_mask" in samples[0]:
        batch["core_mask"] = torch.stack([sample["core_mask"] for sample in samples], dim=0)
    if "roi_mask" in samples[0]:
        batch["roi_mask"] = torch.stack([sample["roi_mask"] for sample in samples], dim=0)
    if "mode" in samples[0]:
        batch["mode"] = samples[0]["mode"]
    return batch


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _teacher_scale(cfg: Lin2025Config, epoch: int) -> float:
    if cfg.epochs <= 1:
        return 1.0
    progress = (epoch - 1) / max(1, cfg.epochs - 1)
    if progress <= cfg.teacher_anneal_start:
        return 1.0
    if progress >= cfg.teacher_anneal_end:
        return cfg.teacher_anneal_final_scale
    span = max(1e-6, cfg.teacher_anneal_end - cfg.teacher_anneal_start)
    alpha = (progress - cfg.teacher_anneal_start) / span
    return 1.0 + alpha * (cfg.teacher_anneal_final_scale - 1.0)


def _teacher_phase_distillation(
    cfg: Lin2025Config,
    pred_holo_phase: torch.Tensor,
    batch: dict[str, torch.Tensor],
    teacher_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    teacher_phase = batch["teacher_phase"].to(pred_holo_phase.device)
    core_weight = batch.get("core_mask")
    roi_weight = batch.get("roi_mask")
    if core_weight is not None:
        core_weight = core_weight.to(pred_holo_phase.device)
    if roi_weight is not None:
        roi_weight = roi_weight.to(pred_holo_phase.device)

    teacher_loss = torch.tensor(0.0, device=pred_holo_phase.device)
    metrics = {
        "teacher_phase_loss": 0.0,
        "core_teacher_phase_loss": 0.0,
    }
    if roi_weight is not None:
        teacher_phase_loss = weighted_wrapped_phase_l1(pred_holo_phase, teacher_phase, roi_weight)
        teacher_loss = teacher_loss + teacher_scale * cfg.hologram_roi_phase_imitation_weight * teacher_phase_loss
        metrics["teacher_phase_loss"] = float(teacher_phase_loss.detach().cpu())
    if core_weight is not None:
        core_teacher_phase_loss = weighted_wrapped_phase_l1(pred_holo_phase, teacher_phase, core_weight)
        teacher_loss = teacher_loss + teacher_scale * cfg.hologram_core_phase_imitation_weight * core_teacher_phase_loss
        metrics["core_teacher_phase_loss"] = float(core_teacher_phase_loss.detach().cpu())
    metrics["teacher_scale"] = float(teacher_scale)
    return teacher_loss, metrics


def train_lin2025_model(cfg: Lin2025Config) -> Path:
    device = torch.device(cfg.device)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = Lin2025HologramNet(
        input_channels=cfg.input_channels,
        hidden_channels=cfg.hidden_channels,
        num_blocks=cfg.num_residual_blocks,
        phase_representation=cfg.phase_representation,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        min_lr=cfg.scheduler_min_lr,
    )

    train_set = create_dataset(cfg, num_samples=cfg.train_samples, seed=42)
    val_set = create_dataset(cfg, num_samples=cfg.val_samples, seed=4242)
    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "collate_fn": _batchify,
        "num_workers": max(0, cfg.dataloader_num_workers),
        "pin_memory": bool(cfg.dataloader_pin_memory and device.type == "cuda"),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(1, cfg.dataloader_prefetch_factor)
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    best_val = float("inf")
    best_quality_score = float("inf")
    best_hybrid_polish_score = float("inf")
    best_path = cfg.checkpoint_dir / "lin2025_best_hybrid_polish.pt"
    best_supervised_path = cfg.checkpoint_dir / "lin2025_best_supervised.pt"
    best_quality_path = cfg.checkpoint_dir / "lin2025_best_quality.pt"
    best_hybrid_polish_path = cfg.checkpoint_dir / "lin2025_best_hybrid_polish.pt"
    latest_path = cfg.checkpoint_dir / "lin2025_latest.pt"
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        start = time.perf_counter()
        train_teacher_phase_sum = 0.0
        train_core_teacher_phase_sum = 0.0
        teacher_scale = _teacher_scale(cfg, epoch)
        for batch in train_loader:
            batch = _move_batch_to_device(batch, device)
            pred_amp, pred_phase = model(
                batch["a_input"],
                batch["phi_input"],
                core_mask=batch.get("core_mask"),
                roi_mask=batch.get("roi_mask"),
            )
            loss_amp = amplitude_l1(pred_amp, batch["a_label"])
            loss_phase = phase_l2(pred_phase, batch["phi_label"])
            loss_holo = hologram_field_error(pred_amp, pred_phase, batch["a_label"], batch["phi_label"])
            loss = loss_amp + loss_phase + 0.1 * loss_holo
            if "teacher_phase" in batch:
                pred_holo_phase = torch.angle(position_to_hologram(pred_amp, pred_phase))
                teacher_loss, teacher_metrics = _teacher_phase_distillation(cfg, pred_holo_phase, batch, teacher_scale)
                loss = loss + teacher_loss
                train_teacher_phase_sum += teacher_metrics["teacher_phase_loss"]
                train_core_teacher_phase_sum += teacher_metrics["core_teacher_phase_loss"]
            if cfg.target_mode == "flat_top":
                weight = (batch["a_input"] > cfg.weight_threshold).to(batch["a_input"].dtype)
                metric_loss, _ = flat_top_metric_loss(
                    pred_amp=pred_amp,
                    pred_phase=pred_phase,
                    target_amp=batch["a_input"],
                    target_phase=batch["phi_input"],
                    weight=weight,
                    overlap_weight=cfg.flat_overlap_weight,
                    uniformity_weight=cfg.flat_uniformity_weight,
                    core_uniformity_weight=cfg.flat_core_uniformity_weight,
                    efficiency_weight=cfg.flat_efficiency_weight,
                    phase_weight=cfg.flat_phase_weight,
                    core_phase_weight=cfg.flat_core_phase_weight,
                    intensity_tv_weight=cfg.flat_intensity_tv_weight,
                    core_threshold=cfg.flat_core_threshold,
                    efficiency_floor=cfg.efficiency_floor,
                )
                loss = loss + metric_loss
            if cfg.skip_nonfinite_batches and not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu())

        model.eval()
        val_loss_sum = 0.0
        val_metric_sums = {
            "uniformity_loss": 0.0,
            "core_uniformity_loss": 0.0,
            "efficiency": 0.0,
            "phase_flatness": 0.0,
            "core_phase_flatness": 0.0,
            "overlap": 0.0,
            "intensity_tv_loss": 0.0,
            "teacher_phase_loss": 0.0,
            "core_teacher_phase_loss": 0.0,
        }
        hybrid_polish_score_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch_to_device(batch, device)
                pred_amp, pred_phase = model(
                    batch["a_input"],
                    batch["phi_input"],
                    core_mask=batch.get("core_mask"),
                    roi_mask=batch.get("roi_mask"),
                )
                loss_amp = amplitude_l1(pred_amp, batch["a_label"])
                loss_phase = phase_l2(pred_phase, batch["phi_label"])
                loss_holo = hologram_field_error(pred_amp, pred_phase, batch["a_label"], batch["phi_label"])
                loss = loss_amp + loss_phase + 0.1 * loss_holo
                if "teacher_phase" in batch:
                    pred_holo_phase = torch.angle(position_to_hologram(pred_amp, pred_phase))
                    teacher_loss, teacher_metrics = _teacher_phase_distillation(cfg, pred_holo_phase, batch, teacher_scale)
                    loss = loss + teacher_loss
                    val_metric_sums["teacher_phase_loss"] += teacher_metrics["teacher_phase_loss"]
                    val_metric_sums["core_teacher_phase_loss"] += teacher_metrics["core_teacher_phase_loss"]
                metric_values = None
                if cfg.target_mode == "flat_top":
                    weight = (batch["a_input"] > cfg.weight_threshold).to(batch["a_input"].dtype)
                    metric_loss, metric_values = flat_top_metric_loss(
                        pred_amp=pred_amp,
                        pred_phase=pred_phase,
                        target_amp=batch["a_input"],
                        target_phase=batch["phi_input"],
                        weight=weight,
                        overlap_weight=cfg.flat_overlap_weight,
                        uniformity_weight=cfg.flat_uniformity_weight,
                        core_uniformity_weight=cfg.flat_core_uniformity_weight,
                        efficiency_weight=cfg.flat_efficiency_weight,
                        phase_weight=cfg.flat_phase_weight,
                        core_phase_weight=cfg.flat_core_phase_weight,
                        intensity_tv_weight=cfg.flat_intensity_tv_weight,
                        core_threshold=cfg.flat_core_threshold,
                        efficiency_floor=cfg.efficiency_floor,
                    )
                    loss = loss + metric_loss
                val_loss_sum += float(loss.detach().cpu())
                if metric_values is not None:
                    for key in (
                        "uniformity_loss",
                        "core_uniformity_loss",
                        "efficiency",
                        "phase_flatness",
                        "core_phase_flatness",
                        "overlap",
                        "intensity_tv_loss",
                    ):
                        val_metric_sums[key] += float(metric_values[key])
                    pred_holo = torch.angle(position_to_hologram(pred_amp, pred_phase))
                    init_score, _ = hybrid_init_score(
                        pred_hologram_phase=pred_holo,
                        target_amp_small=batch["a_input"],
                        target_phase_small=batch["phi_input"],
                        beam_sigma_px=cfg.beam_sigma_px,
                        overlap_weight=cfg.hybrid_init_overlap_weight,
                        uniformity_weight=cfg.hybrid_init_uniformity_weight,
                        core_uniformity_weight=cfg.hybrid_init_core_uniformity_weight,
                        efficiency_weight=cfg.hybrid_init_efficiency_weight,
                        core_phase_weight=cfg.hybrid_init_core_phase_weight,
                        core_threshold=cfg.flat_core_threshold,
                    )
                    hybrid_polish_score_sum += float(init_score)

        train_batches = max(1, len(train_loader))
        val_batches = max(1, len(val_loader))
        train_loss = train_loss_sum / train_batches
        val_loss = val_loss_sum / val_batches
        avg_metrics = {
            key: value / val_batches
            for key, value in val_metric_sums.items()
        }
        avg_hybrid_polish_score = hybrid_polish_score_sum / val_batches
        quality_score = (
            cfg.metric_score_core_uniformity_weight * avg_metrics["core_uniformity_loss"]
            + cfg.metric_score_core_phase_weight * avg_metrics["core_phase_flatness"]
            + cfg.metric_score_efficiency_weight * float(soft_efficiency_constraint(torch.tensor(avg_metrics["efficiency"]), cfg.efficiency_floor))
            + cfg.metric_score_uniformity_weight * avg_metrics["uniformity_loss"]
        ) if cfg.target_mode == "flat_top" else val_loss
        epoch_time = time.perf_counter() - start
        history.append({"epoch": float(epoch), "train_loss": train_loss, "supervised_loss": val_loss, "quality_score": quality_score, "hybrid_polish_score": avg_hybrid_polish_score, **avg_metrics, "epoch_sec": epoch_time})
        history[-1]["train_teacher_phase_loss"] = train_teacher_phase_sum / train_batches
        history[-1]["train_core_teacher_phase_loss"] = train_core_teacher_phase_sum / train_batches
        history[-1]["teacher_scale"] = teacher_scale
        history[-1]["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        print(
            f"epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"quality={quality_score:.6f} | polish={avg_hybrid_polish_score:.6f} | "
            f"core_teacher={avg_metrics['core_teacher_phase_loss']:.6f} | "
            f"teacher_scale={teacher_scale:.3f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | time={epoch_time:.2f}s"
        )
        scheduler.step(quality_score)
        torch.save(model.state_dict(), latest_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_supervised_path)
        if quality_score < best_quality_score:
            best_quality_score = quality_score
            torch.save(model.state_dict(), best_quality_path)
        if avg_hybrid_polish_score < best_hybrid_polish_score:
            best_hybrid_polish_score = avg_hybrid_polish_score
            torch.save(model.state_dict(), best_hybrid_polish_path)
            torch.save(model.state_dict(), best_path)

    metadata = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "best_val_loss": best_val,
        "best_quality_score": best_quality_score,
        "best_hybrid_polish_score": best_hybrid_polish_score,
        "best_checkpoint": str(best_path),
        "best_supervised_checkpoint": str(best_supervised_path),
        "best_quality_checkpoint": str(best_quality_path),
        "best_hybrid_polish_checkpoint": str(best_hybrid_polish_path),
        "history": history,
    }
    (cfg.checkpoint_dir / "lin2025_training_history.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if best_hybrid_polish_path.exists():
        legacy_best = cfg.checkpoint_dir / "lin2025_best.pt"
        legacy_init = cfg.checkpoint_dir / "lin2025_best_init_for_hybrid.pt"
        torch.save(torch.load(best_hybrid_polish_path, map_location="cpu"), legacy_best)
        torch.save(torch.load(best_hybrid_polish_path, map_location="cpu"), legacy_init)
    return best_path
