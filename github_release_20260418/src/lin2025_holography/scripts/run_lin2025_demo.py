import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lin2025_holography.config import Lin2025Config
from lin2025_holography.inference import run_lin2025_demo


def main() -> None:
    cfg = Lin2025Config(run_name="lin2025_demo")
    metrics = run_lin2025_demo(cfg)
    print(metrics)


if __name__ == "__main__":
    main()
