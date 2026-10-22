"""Script to evaluate super-resolution models from a checkpoint or pre-trained config."""
import argparse
from pathlib import Path
import sys

# Insert repo root into sys.path
repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.utils.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate super-resolution models.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML experiment config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--set", action="append", default=[], help="Key=value overrides (repeatable)")
    parser.add_argument("--device", type=str, default=None, help="Device override (auto|cpu|cuda)")

    args = parser.parse_args()

    run_experiment(
        cfg_or_path=args.config,
        overrides=args.set,
        overrides_already_applied=False,
        checkpoint=args.checkpoint,
        eval_only=True,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
