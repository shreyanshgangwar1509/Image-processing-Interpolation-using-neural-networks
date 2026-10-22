"""Script to train or evaluate super-resolution models specified in a YAML config."""
import argparse
from pathlib import Path
import sys

# Insert repo root into sys.path
repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.utils.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate super-resolution models.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML experiment config")
    parser.add_argument("--set", action="append", default=[], help="Key=value overrides (repeatable)")
    parser.add_argument("--device", type=str, default=None, help="Device override (auto|cpu|cuda)")
    parser.add_argument("--force", action="store_true", help="Force re-running even if skip_if_done is true")

    args = parser.parse_args()

    run_experiment(
        cfg_or_path=args.config,
        overrides=args.set,
        overrides_already_applied=False,
        eval_only=False,
        device_override=args.device,
        force=args.force,
    )


if __name__ == "__main__":
    main()
