import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_holography.config import HolographyConfig
from ai_holography.profiles import apply_profile
from ai_holography.references import apply_795_reference_metadata, find_795_reference_files, parse_795_reference_name
from ai_holography.runner import HybridHolographyRunner


def main() -> None:
    ref_dir = Path("D:/Trae products/flat_top light/ftl_gen")
    refs = find_795_reference_files(ref_dir)
    if not refs:
        raise FileNotFoundError(f"No 795*.npy files found in {ref_dir}")

    ref = refs[0]
    cfg = apply_profile(HolographyConfig(), "experiment")
    cfg.reference_phase_path = ref

    info = parse_795_reference_name(ref)
    cfg = apply_795_reference_metadata(cfg, info)
    cfg.output_dir = cfg.run_dir / f"outputs_795_reference_{info.get('style')}"
    print("Using reference:", ref)
    print("Reference info:", info)

    runner = HybridHolographyRunner(cfg)
    _, metrics = runner.run_once(use_warm_start=False, save_subdir=None)
    print(metrics)


if __name__ == "__main__":
    main()
