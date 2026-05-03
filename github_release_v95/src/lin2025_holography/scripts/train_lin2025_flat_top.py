import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lin2025_holography.config import Lin2025Config
from lin2025_holography.training import train_lin2025_model


def main() -> None:
    cfg = Lin2025Config(
        target_mode="flat_top",
        slm_size=128,
        train_samples=96,
        val_samples=12,
        epochs=4,
        batch_size=4,
    )
    best = train_lin2025_model(cfg)
    print("Best checkpoint:", best)


if __name__ == "__main__":
    main()

