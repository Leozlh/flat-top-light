from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import torch

from ai_holography.losses import target_core_weight
from ai_holography.references import apply_795_reference_metadata, find_795_reference_files, load_reference_phase, parse_795_reference_name
from ai_holography.targets import build_phase, build_target, build_weight, threshold_weight

from .config import Lin2025Config
from .encoding import encode_inputs
from .wgs import hologram_to_position_labels, weighted_gs_hologram


class OnTheFlyWGSDataset:
    def __init__(self, cfg: Lin2025Config, num_samples: int, seed: int = 42):
        self.cfg = cfg
        self.num_samples = num_samples
        self.rng = random.Random(seed)
        self.sample_device = torch.device("cpu")

    def __len__(self) -> int:
        return self.num_samples

    def _sample_coords(self) -> torch.Tensor:
        coords: list[tuple[float, float]] = []
        attempts = 0
        margin = self.cfg.slm_size * 0.12
        while len(coords) < self.cfg.num_traps and attempts < self.cfg.num_traps * 200:
            attempts += 1
            x = self.rng.uniform(margin, self.cfg.slm_size - margin)
            y = self.rng.uniform(margin, self.cfg.slm_size - margin)
            if all((x - px) ** 2 + (y - py) ** 2 >= self.cfg.min_spacing_px**2 for px, py in coords):
                coords.append((x, y))
        if len(coords) < self.cfg.num_traps:
            raise RuntimeError("Failed to sample enough non-colliding trap coordinates.")
        return torch.tensor(coords, dtype=torch.float32, device=self.sample_device)

    def _sample_phases(self) -> torch.Tensor:
        return (2.0 * torch.pi * torch.rand((self.cfg.num_traps,), device=self.sample_device) - torch.pi).float()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        del idx
        coords = self._sample_coords()
        phases = self._sample_phases()
        a_input, phi_input = encode_inputs(coords, phases, size=self.cfg.slm_size, device=str(self.sample_device))
        hologram = weighted_gs_hologram(
            coords=coords,
            phases=phases,
            slm_size=self.cfg.slm_size,
            oversampled_size=self.cfg.oversampled_size,
            trap_sigma=self.cfg.trap_sigma_px,
            beam_sigma=self.cfg.beam_sigma_px,
            iterations=self.cfg.wgs_iterations,
            device=str(self.sample_device),
        )
        a_label, phi_label = hologram_to_position_labels(hologram, label_size=self.cfg.slm_size)
        return {
            "a_input": a_input,
            "phi_input": phi_input,
            "a_label": a_label,
            "phi_label": phi_label,
            "coords": coords,
            "trap_phases": phases,
            "mode": "trap_array",
        }


class FlatTopReferenceDataset:
    """
    Lin-style supervised dataset adapted to flat-top holography:
    - input is the target amplitude/phase image in the position domain
    - label supervision comes from 795-style phase-only holograms
    """

    def __init__(self, cfg: Lin2025Config, num_samples: int, seed: int = 42):
        self.cfg = cfg
        self.num_samples = num_samples
        self.rng = random.Random(seed)
        self.sample_device = torch.device("cpu")
        refs = find_795_reference_files(Path(cfg.reference_dir))
        self.refs = [ref for ref in refs if parse_795_reference_name(ref).get("style") in {"round_top", "flat_top"}]
        if not self.refs:
            raise FileNotFoundError(f"No 795 round_top/flat_top references found in {cfg.reference_dir}")
        self.hybrid_teachers = self._discover_hybrid_teachers()

    def _discover_hybrid_teachers(self) -> dict[str, list[Path]]:
        teachers: dict[str, list[Path]] = {}
        explicit = {
            "round_top": self.cfg.hybrid_teacher_round_path,
            "flat_top": self.cfg.hybrid_teacher_flat_path,
        }
        for style, path in explicit.items():
            if path is not None and Path(path).exists():
                teachers[style] = [Path(path)]

        search_root = Path(self.cfg.hybrid_teacher_search_root)
        if search_root.exists():
            if "round_top" not in teachers:
                selected = self._select_best_hybrid_teachers(search_root, "round_top")
                if selected:
                    teachers["round_top"] = selected
            if "flat_top" not in teachers:
                selected = self._select_best_hybrid_teachers(search_root, "flat_top")
                if selected:
                    teachers["flat_top"] = selected
        return teachers

    def _teacher_score_weights(self, style: str) -> tuple[float, float, float, float]:
        if style == "flat_top":
            return (
                self.cfg.hybrid_teacher_score_uniformity_weight_flat,
                self.cfg.hybrid_teacher_score_core_uniformity_weight_flat,
                self.cfg.hybrid_teacher_score_efficiency_weight_flat,
                self.cfg.hybrid_teacher_score_core_phase_weight_flat,
            )
        return (
            self.cfg.hybrid_teacher_score_uniformity_weight_round,
            self.cfg.hybrid_teacher_score_core_uniformity_weight_round,
            self.cfg.hybrid_teacher_score_efficiency_weight_round,
            self.cfg.hybrid_teacher_score_core_phase_weight_round,
        )

    def _teacher_phase_gate(self, style: str) -> float:
        if style == "flat_top":
            return self.cfg.hybrid_teacher_phase_gate_flat
        return self.cfg.hybrid_teacher_phase_gate_round

    def _teacher_top_k(self, style: str) -> int:
        if style == "flat_top":
            return max(1, self.cfg.hybrid_teacher_top_k_flat)
        return max(1, self.cfg.hybrid_teacher_top_k_round)

    def _teacher_score(self, style: str, metrics: dict[str, float]) -> float:
        uniformity_weight, core_uniformity_weight, efficiency_weight, core_phase_weight = self._teacher_score_weights(style)
        return (
            core_uniformity_weight * float(metrics.get("core_uniformity_loss", float("inf")))
            + core_phase_weight * float(metrics.get("core_phase_flatness", float("inf")))
            + efficiency_weight * (1.0 - float(metrics.get("efficiency", 0.0)))
            + uniformity_weight * float(metrics.get("uniformity_loss", float("inf")))
        )

    def _teacher_patterns(self, style: str) -> list[str]:
        if style == "flat_top":
            return [
                "**/outputs_flat_top/slm_phase.npy",
                "**/outputs/flat_*/slm_phase.npy",
            ]
        return [
            "**/outputs_round_top/slm_phase.npy",
            "**/outputs/round_*/slm_phase.npy",
        ]

    def _select_best_hybrid_teachers(self, search_root: Path, style: str) -> list[Path]:
        candidates: list[tuple[float, float, Path]] = []
        gated: list[tuple[float, float, Path]] = []
        seen: set[Path] = set()
        for pattern in self._teacher_patterns(style):
            for phase_path in search_root.glob(pattern):
                phase_path = phase_path.resolve()
                if phase_path in seen:
                    continue
                seen.add(phase_path)
                metrics_path = phase_path.with_name("metrics.json")
                if not metrics_path.exists():
                    continue
                try:
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                core_phase_flatness = float(metrics.get("core_phase_flatness", float("inf")))
                core_uniformity_loss = float(metrics.get("core_uniformity_loss", float("inf")))
                efficiency = float(metrics.get("efficiency", 0.0))
                score = self._teacher_score(style, metrics)
                entry = (score, core_phase_flatness, phase_path)
                candidates.append(entry)
                if core_phase_flatness <= self._teacher_phase_gate(style):
                    gated.append(entry)
        pool = gated if gated else candidates
        pool.sort(key=lambda item: (item[0], item[1]))
        selected = [path for _, _, path in pool[: self._teacher_top_k(style)]]
        return selected

    def __len__(self) -> int:
        return self.num_samples

    def _source_weights_for_style(self, style: str) -> dict[str, float]:
        if style == "flat_top":
            return {
                "795": self.cfg.teacher_weight_795_flat,
                "hybrid": self.cfg.teacher_weight_hybrid_flat,
            }
        return {
            "795": self.cfg.teacher_weight_795_round,
            "hybrid": self.cfg.teacher_weight_hybrid_round,
        }

    def _choose_teacher_source(self, style: str) -> str:
        weights = self._source_weights_for_style(style)
        hybrid_available = bool(self.hybrid_teachers.get(style))
        pool: list[str] = []
        if "795" in self.cfg.teacher_sources and weights["795"] > 0:
            pool.extend(["795"] * max(1, int(round(weights["795"] * 100))))
        if "hybrid" in self.cfg.teacher_sources and hybrid_available and weights["hybrid"] > 0:
            pool.extend(["hybrid"] * max(1, int(round(weights["hybrid"] * 100))))
        if not pool:
            return "795"
        return self.rng.choice(pool)

    def _choose_hybrid_teacher_path(self, style: str) -> Path | None:
        candidates = self.hybrid_teachers.get(style, [])
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        ranks = list(range(1, len(candidates) + 1))
        weights = [1.0 / rank for rank in ranks]
        total = sum(weights)
        norm = [w / total for w in weights]
        index = self.rng.choices(range(len(candidates)), weights=norm, k=1)[0]
        return candidates[index]

    def _sample_aug(self) -> tuple[float, float, float]:
        shift_x = self.rng.uniform(-0.08, 0.08)
        shift_y = self.rng.uniform(-0.08, 0.08)
        scale = 1.0 + self.rng.uniform(-0.12, 0.12)
        return shift_x, shift_y, scale

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ref = self.refs[idx % len(self.refs)]
        info = parse_795_reference_name(ref)
        style = str(info.get("style"))
        shift_x, shift_y, scale = self._sample_aug()

        class DummyCfg:
            pass

        local = DummyCfg()
        local.output_size = self.cfg.slm_size
        local.reference_output_size = self.cfg.reference_output_size
        local.target_center_x = None
        local.target_center_y = None
        local.target_shift_x_m = 0.0
        local.target_shift_y_m = 0.0
        local.roi_diameter = self.cfg.target_sigma_px
        local.target_sigma = self.cfg.target_sigma_px
        local.target_type = "target_flat_top"
        local.weight_type = "gaussian_top_round"
        local.phase_type = "phase_flat"
        local.uniformity_weight = 0.0
        local.efficiency_weight = 0.0
        local = apply_795_reference_metadata(local, info)

        cx = (local.target_center_x if local.target_center_x is not None else self.cfg.slm_size / 2.0) + shift_x * self.cfg.slm_size
        cy = (local.target_center_y if local.target_center_y is not None else self.cfg.slm_size / 2.0) + shift_y * self.cfg.slm_size
        sigma = max(4.0, float(local.target_sigma) * scale)
        roi_d = max(4.0, float(local.roi_diameter) * scale)
        center = (cy, cx)

        target_amp = build_target(
            target_type=local.target_type,
            size=self.cfg.slm_size,
            center=center,
            sigma=sigma,
            charge=0,
            device=str(self.sample_device),
        )
        target_phase = build_phase(
            phase_type=local.phase_type,
            size=self.cfg.slm_size,
            center=center,
            device=str(self.sample_device),
        )
        roi = build_weight(
            weight_type=local.weight_type,
            size=self.cfg.slm_size,
            center=center,
            diameter=roi_d,
            softness=self.cfg.roi_softness_px,
            device=str(self.sample_device),
        )
        weight = threshold_weight(roi, fraction=self.cfg.weight_threshold)
        a_input = target_amp * weight
        if a_input.max() > 0:
            a_input = a_input / a_input.max()
        phi_input = target_phase / torch.pi
        core_mask = target_core_weight(a_input, weight, self.cfg.flat_core_threshold)
        core_mask = (core_mask > 0).to(a_input.dtype)
        roi_mask = (weight > 0).to(a_input.dtype)

        teacher_source = self._choose_teacher_source(style)
        hybrid_teacher = self._choose_hybrid_teacher_path(style)
        teacher_path = hybrid_teacher if teacher_source == "hybrid" and hybrid_teacher is not None else ref
        reference_phase = load_reference_phase(
            teacher_path,
            size=(self.cfg.slm_size, self.cfg.slm_size),
            device=str(self.sample_device),
        )
        teacher_hologram = torch.exp(1j * reference_phase)
        a_label, phi_label = hologram_to_position_labels(teacher_hologram, label_size=self.cfg.slm_size)
        return {
            "a_input": a_input,
            "phi_input": phi_input.float(),
            "a_label": a_label,
            "phi_label": phi_label,
            "teacher_phase": reference_phase.float(),
            "core_mask": core_mask.float(),
            "roi_mask": roi_mask.float(),
            "mode": "flat_top",
            "reference_path": str(teacher_path),
            "teacher_source": teacher_source if teacher_source != "hybrid" or hybrid_teacher is not None else "795",
            "teacher_style": style,
        }


def create_dataset(cfg: Lin2025Config, num_samples: int, seed: int) -> object:
    if cfg.target_mode == "flat_top":
        return FlatTopReferenceDataset(cfg, num_samples=num_samples, seed=seed)
    return OnTheFlyWGSDataset(cfg, num_samples=num_samples, seed=seed)
