from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class HolographyConfig:
    preset: str = "bowman_lg"
    slm_size: int = 256
    output_size: int = 512
    wavelength_m: float = 795e-9
    focal_length_m: float = 0.2
    slm_pixel_pitch_m: float = 12.5e-6
    magnification: float = 1.0
    target_shift_x_m: float = 0.0
    target_shift_y_m: float = 0.0
    beam_sigma_x: float = 40.0
    beam_sigma_y: float = 40.0
    target_sigma: float = 7.0
    vortex_charge: int = 1
    target_type: str = "target_lg"
    phase_type: str = "phase_spinning_continuous"
    weight_type: str = "gaussian_top_round"
    training_target_types: tuple[str, ...] | None = None
    target_center_x: float | None = None
    target_center_y: float | None = None
    slm_phase_correction_path: Path | None = None
    reference_phase_path: Path | None = None
    reference_phase_mix: float = 0.75
    init_candidate_sources: tuple[str, ...] = ("neural", "reference", "lin2025", "warm_start")
    lin2025_checkpoint_dir: Path = Path("lin2025_holography") / "checkpoints"
    lin2025_checkpoint: Path | None = None
    lin2025_mix: float = 0.6
    lin2025_primary: bool = False
    lin2025_reference_weak_mix: float = 0.35
    warm_start_external_mix_floor: float = 0.25
    init_score_core_uniformity_weight: float = 1.0
    init_score_core_phase_weight: float = 1.0
    init_score_efficiency_weight: float = 0.5
    init_score_uniformity_weight: float = 0.2
    reference_output_size: int = 6876
    roi_diameter: float = 32.0
    roi_softness: float = 2.0
    flat_top_order: float = 8.0
    flat_top_edge_softness: float = 0.12
    weight_threshold: float = 1e-4
    init_tilt: float = -1.5707963267948966
    init_aspect: float = 0.5
    init_curvature: float = 0.003
    init_angle: float = 0.7853981633974483
    init_cone: float = 0.0
    phase_residual_scale: float = 1.5707963267948966
    learning_rate: float = 3e-2
    iterations: int = 600
    multiscale_levels: tuple[int, ...] = (64, 128, 256)
    stage_iterations: tuple[int, ...] = (120, 90, 60)
    early_stop_patience: int = 20
    early_stop_min_delta: float = 1e-5
    target_overlap: float = 0.9993
    refine_with_lbfgs: bool = True
    lbfgs_steps: int = 40
    skip_lbfgs_if_target_met: bool = True
    hybrid_cg_maxiter: int = 60
    polish_stage1_maxiter: int = 20
    polish_stage2_maxiter: int = 15
    polish_stage3_maxiter: int = 8
    polish_enable_phase_priority: bool = True
    polish_phase_priority_core_uniformity_weight: float = 0.55
    polish_phase_priority_core_phase_weight: float = 0.8
    polish_phase_priority_uniformity_weight: float = 0.08
    polish_phase_priority_efficiency_weight: float = 0.03
    polish_phase_priority_phase_weight: float = 0.35
    polish_overlap_weight: float = 1.0
    polish_intensity_weight: float = 0.2
    polish_intensity_tv_weight: float = 0.02
    polish_phase_weight: float = 0.15
    polish_uniformity_weight: float = 0.25
    polish_efficiency_weight: float = 0.15
    polish_smoothness_weight: float = 1e-4
    efficiency_floor: float = 0.05
    phase_post_correction_enabled: bool = True
    phase_post_correction_steps: int = 12
    phase_post_correction_lr: float = 8e-3
    phase_post_correction_max_delta: float = 0.2
    phase_post_correction_core_uniformity_weight: float = 0.08
    phase_post_correction_core_phase_weight: float = 0.9
    phase_post_correction_efficiency_weight: float = 0.02
    phase_post_correction_uniformity_weight: float = 0.0
    warm_start_similarity_threshold: float = 0.995
    warm_start_min_mix: float = 0.15
    warm_start_max_mix: float = 0.9
    warm_start_candidate_threshold: float = 0.7
    warm_start_allow_direct: bool = True
    fast_init_overlap_threshold: float = 0.998
    adaptive_weighting: bool = True
    adaptive_weighting_alpha: float = 0.25
    adaptive_weight_clip_min: float = 0.5
    adaptive_weight_clip_max: float = 2.0
    overlap_weight: float = 1.0
    intensity_weight: float = 0.25
    phase_weight: float = 0.15
    uniformity_weight: float = 0.2
    efficiency_weight: float = 0.1
    core_uniformity_weight: float = 0.0
    core_phase_weight: float = 0.0
    core_region_threshold: float = 0.7
    smoothness_weight: float = 1e-4
    schedule_ramp_start: float = 0.35
    schedule_ramp_end: float = 0.85
    run_name: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir: Path = field(default_factory=lambda: Path("ai_holography") / "runs")
    output_dir: Path | None = None
    comparison_dir: Path | None = None
    benchmark_dir: Path | None = None
    checkpoint: Path | None = None
    auto_load_best_checkpoint: bool = True
    checkpoint_dir: Path = Path("ai_holography") / "checkpoints"
    reference_dir: Path = Path("ftl_gen")
    enable_reference_pretrain: bool = True
    reference_pretrain_epochs: int = 2
    reference_phase_loss_weight: float = 1.0
    reference_tv_loss_weight: float = 1e-4
    reference_physics_loss_weight: float = 0.2
    reference_pretrain_lowres_size: int = 128
    reference_augmented_samples: int = 64
    reference_shift_px: float = 8.0
    reference_scale_jitter: float = 0.12
    reference_beam_sigma_jitter: float = 0.08
    training_uniformity_metric_weight: float = 0.3
    training_core_uniformity_metric_weight: float = 1.0
    training_core_phase_metric_weight: float = 1.0
    training_efficiency_metric_weight: float = 0.6
    training_round_uniformity_metric_weight: float = 0.2
    training_round_core_uniformity_metric_weight: float = 0.6
    training_round_core_phase_metric_weight: float = 0.6
    training_round_efficiency_metric_weight: float = 1.0
    training_flat_uniformity_metric_weight: float = 0.1
    training_flat_core_uniformity_metric_weight: float = 1.6
    training_flat_core_phase_metric_weight: float = 1.6
    training_flat_efficiency_metric_weight: float = 0.5
    training_metric_patience: int = 2
    training_metric_min_delta: float = 1e-4
    device: str = "cpu"
    feedback_target_gain: float = 0.15

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir) / self.run_name
        if self.output_dir is None:
            self.output_dir = self.run_dir / "outputs"
        else:
            self.output_dir = Path(self.output_dir)
        if self.comparison_dir is None:
            self.comparison_dir = self.run_dir / "comparison"
        else:
            self.comparison_dir = Path(self.comparison_dir)
        if self.benchmark_dir is None:
            self.benchmark_dir = self.run_dir / "benchmarks"
        else:
            self.benchmark_dir = Path(self.benchmark_dir)
