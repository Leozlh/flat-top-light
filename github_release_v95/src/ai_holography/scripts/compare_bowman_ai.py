from __future__ import annotations

import sys
import json
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ai_holography.config import HolographyConfig
from ai_holography.hybrid import bowman_cg_refine
from ai_holography.pipeline import AIHolographyPipeline
from ai_holography.targets import quadratic_phase_guess
from ai_holography.visualization import save_field_visualizations, save_linecuts


def run_bowman_style_baseline(cfg: HolographyConfig, problem: dict[str, torch.Tensor]) -> dict[str, object]:
    init_phase = quadratic_phase_guess(
        cfg.slm_size,
        tilt=cfg.init_tilt,
        aspect=cfg.init_aspect,
        curvature=cfg.init_curvature,
        angle=cfg.init_angle,
        cone=cfg.init_cone,
        device=cfg.device,
    )
    return bowman_cg_refine(cfg, problem, init_phase, maxiter=120)


def save_comparison(
    root: Path,
    problem: dict[str, torch.Tensor],
    bowman: dict[str, object],
    ai: dict[str, object],
    hybrid: dict[str, object],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    cfg_dict = asdict(HolographyConfig())
    cfg_dict["output_dir"] = str(cfg_dict["output_dir"])
    cfg_dict["checkpoint_dir"] = str(cfg_dict["checkpoint_dir"])
    if cfg_dict["checkpoint"] is not None:
        cfg_dict["checkpoint"] = str(cfg_dict["checkpoint"])
    (root / "comparison_metrics.json").write_text(
        json.dumps(
            {
                "config": cfg_dict,
                "bowman": bowman["metrics"],
                "ai": ai["metrics"],
                "hybrid": hybrid["metrics"],
                "delta": {
                    "runtime_sec": ai["metrics"]["runtime_sec"] - bowman["metrics"]["runtime_sec"],
                    "overlap": ai["metrics"]["overlap"] - bowman["metrics"]["overlap"],
                    "phase_loss": ai["metrics"]["phase_loss"] - bowman["metrics"]["phase_loss"],
                    "intensity_loss": ai["metrics"]["intensity_loss"] - bowman["metrics"]["intensity_loss"],
                },
                "hybrid_delta_vs_bowman": {
                    "runtime_sec": hybrid["metrics"]["runtime_sec"] - bowman["metrics"]["runtime_sec"],
                    "overlap": hybrid["metrics"]["overlap"] - bowman["metrics"]["overlap"],
                    "phase_loss": hybrid["metrics"]["phase_loss"] - bowman["metrics"]["phase_loss"],
                    "intensity_loss": hybrid["metrics"]["intensity_loss"] - bowman["metrics"]["intensity_loss"],
                },
                "hybrid_delta_vs_ai": {
                    "runtime_sec": hybrid["metrics"]["runtime_sec"] - ai["metrics"]["runtime_sec"],
                    "overlap": hybrid["metrics"]["overlap"] - ai["metrics"]["overlap"],
                    "phase_loss": hybrid["metrics"]["phase_loss"] - ai["metrics"]["phase_loss"],
                    "intensity_loss": hybrid["metrics"]["intensity_loss"] - ai["metrics"]["intensity_loss"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    save_field_visualizations(
        root / "bowman",
        problem["target_amp"],
        problem["target_phase"],
        bowman["out_amp"],
        bowman["out_phase"],
        bowman["phase"],
    )
    save_linecuts(root / "bowman", problem["target_amp"], bowman["out_amp"])

    save_field_visualizations(
        root / "ai",
        problem["target_amp"],
        problem["target_phase"],
        ai["out_amp"],
        ai["out_phase"],
        ai["phase"],
    )
    save_linecuts(root / "ai", problem["target_amp"], ai["out_amp"])

    save_field_visualizations(
        root / "hybrid",
        problem["target_amp"],
        problem["target_phase"],
        hybrid["out_amp"],
        hybrid["out_phase"],
        hybrid["phase"],
    )
    save_linecuts(root / "hybrid", problem["target_amp"], hybrid["out_amp"])


def main() -> None:
    cfg = HolographyConfig()
    pipeline = AIHolographyPipeline(cfg)
    problem = pipeline.build_problem()

    bowman = run_bowman_style_baseline(cfg, problem)

    start = time.perf_counter()
    init_phase = pipeline.predict_initial_phase(problem["target_amp"], problem["target_phase"], problem["weight"])
    ai_phase, ai_metrics, ai_amp, ai_phase_out = pipeline.refine(
        problem["beam"],
        problem["target_amp"],
        problem["target_phase"],
        problem["weight"],
        init_phase,
    )
    ai_metrics["runtime_sec"] = time.perf_counter() - start
    ai = {
        "phase": ai_phase.cpu(),
        "out_amp": ai_amp.cpu(),
        "out_phase": ai_phase_out.cpu(),
        "metrics": ai_metrics,
    }
    ai["metrics"]["checkpoint"] = str(pipeline.loaded_checkpoint) if pipeline.loaded_checkpoint is not None else None
    hybrid = bowman_cg_refine(
        cfg,
        problem,
        ai_phase,
        maxiter=cfg.hybrid_cg_maxiter,
        target_overlap=cfg.target_overlap,
    )
    hybrid["metrics"]["checkpoint"] = ai["metrics"]["checkpoint"]
    hybrid["metrics"]["init"] = "trained_ai"

    save_comparison(cfg.comparison_dir, problem, bowman, ai, hybrid)
    print(json.dumps({"bowman": bowman["metrics"], "ai": ai["metrics"], "hybrid": hybrid["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
