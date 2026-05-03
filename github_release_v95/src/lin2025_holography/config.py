from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Lin2025Config:
    target_mode: str = "trap_array"
    slm_size: int = 128
    oversample_factor: int = 4
    num_traps: int = 64
    min_spacing_px: float = 6.0
    trap_sigma_px: float = 1.8
    beam_sigma_px: float = 28.0
    reference_dir: Path = Path("ftl_gen")
    reference_output_size: int = 6876
    target_sigma_px: float = 18.0
    roi_softness_px: float = 2.0
    weight_threshold: float = 1e-4
    wgs_iterations: int = 20
    train_samples: int = 192
    val_samples: int = 24
    epochs: int = 8
    batch_size: int = 4
    learning_rate: float = 3e-4
    grad_clip_norm: float = 1.0
    skip_nonfinite_batches: bool = True
    scheduler_patience: int = 1
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-5
    device: str = "cpu"
    input_channels: int = 4
    hidden_channels: int = 48
    num_residual_blocks: int = 4
    flat_overlap_weight: float = 0.1
    flat_uniformity_weight: float = 0.25
    flat_core_uniformity_weight: float = 1.15
    flat_efficiency_weight: float = 0.35
    flat_core_phase_weight: float = 1.8
    flat_phase_weight: float = 0.1
    flat_intensity_tv_weight: float = 0.02
    flat_core_threshold: float = 0.78
    efficiency_floor: float = 0.05
    phase_representation: str = "phasor"
    teacher_sources: tuple[str, ...] = ("795", "hybrid")
    teacher_weight_795: float = 0.1
    teacher_weight_hybrid: float = 0.9
    teacher_weight_795_round: float = 0.4
    teacher_weight_hybrid_round: float = 0.6
    teacher_weight_795_flat: float = 0.0
    teacher_weight_hybrid_flat: float = 1.0
    hybrid_teacher_phase_gate: float = 0.0018
    hybrid_teacher_phase_gate_round: float = 0.0025
    hybrid_teacher_phase_gate_flat: float = 0.0022
    hybrid_teacher_top_k: int = 6
    hybrid_teacher_top_k_round: int = 6
    hybrid_teacher_top_k_flat: int = 6
    hybrid_teacher_score_uniformity_weight: float = 0.15
    hybrid_teacher_score_core_uniformity_weight: float = 1.2
    hybrid_teacher_score_efficiency_weight: float = 0.4
    hybrid_teacher_score_core_phase_weight: float = 1.3
    hybrid_teacher_score_uniformity_weight_round: float = 0.15
    hybrid_teacher_score_core_uniformity_weight_round: float = 1.2
    hybrid_teacher_score_efficiency_weight_round: float = 0.4
    hybrid_teacher_score_core_phase_weight_round: float = 1.3
    hybrid_teacher_score_uniformity_weight_flat: float = 0.05
    hybrid_teacher_score_core_uniformity_weight_flat: float = 0.7
    hybrid_teacher_score_efficiency_weight_flat: float = 0.15
    hybrid_teacher_score_core_phase_weight_flat: float = 2.2
    hybrid_teacher_round_path: Path | None = None
    hybrid_teacher_flat_path: Path | None = None
    hybrid_teacher_search_root: Path = Path("ai_holography") / "runs"
    hybrid_init_overlap_weight: float = 0.1
    hybrid_init_uniformity_weight: float = 0.2
    hybrid_init_core_uniformity_weight: float = 1.0
    hybrid_init_efficiency_weight: float = 0.5
    hybrid_init_core_phase_weight: float = 1.2
    hologram_phase_imitation_weight: float = 0.2
    hologram_core_phase_imitation_weight: float = 0.7
    hologram_roi_phase_imitation_weight: float = 0.2
    teacher_anneal_start: float = 0.35
    teacher_anneal_end: float = 0.85
    teacher_anneal_final_scale: float = 0.2
    metric_score_uniformity_weight: float = 0.15
    metric_score_core_uniformity_weight: float = 1.2
    metric_score_efficiency_weight: float = 0.4
    metric_score_core_phase_weight: float = 1.3
    run_name: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir: Path = field(default_factory=lambda: Path("lin2025_holography") / "runs")
    checkpoint_dir: Path = Path("lin2025_holography") / "checkpoints"
    output_dir: Path | None = None
    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir) / self.run_name
        if self.output_dir is None:
            self.output_dir = self.run_dir / "outputs"
        else:
            self.output_dir = Path(self.output_dir)

    @property
    def oversampled_size(self) -> int:
        return self.slm_size * self.oversample_factor
