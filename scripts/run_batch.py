"""Script to run a batch of YAML experiment configs sequentially across platforms."""
import argparse
import glob
from pathlib import Path
import sys
import time
import traceback

# Insert repo root into sys.path
repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.utils.experiment import run_experiment


def expand_globs(patterns: list[str]) -> list[Path]:
    """Expands list of glob patterns into sorted unique Path objects."""
    found_paths = set()
    for pattern in patterns:
        # If glob pattern, expand with glob
        matched = glob.glob(pattern, recursive=True)
        if matched:
            for m in matched:
                p = Path(m)
                if p.is_file() and p.suffix.lower() in (".yaml", ".yml"):
                    if not p.name.startswith("_base_"):
                        found_paths.add(p.resolve())
        else:
            p = Path(pattern)
            if p.is_file() and not p.name.startswith("_base_"):
                found_paths.add(p.resolve())

    return sorted(list(found_paths))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch of super-resolution experiment configs.")
    parser.add_argument("configs", nargs="+", type=str, help="YAML config file paths or glob patterns")
    parser.add_argument("--device", type=str, default=None, help="Device override (auto|cpu|cuda)")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue running remaining configs if one fails (default True)",
    )
    parser.add_argument("--force", action="store_true", help="Force re-running experiments")

    args = parser.parse_args()

    config_paths = expand_globs(args.configs)
    if not config_paths:
        print("No valid non-base config files found matching the provided arguments.")
        sys.exit(1)

    print(f"Found {len(config_paths)} experiment config(s) to execute.")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    batch_log_path = results_dir / "batch_log.txt"

    log_lines = []
    log_lines.append(f"=== Batch Run Log [{time.strftime('%Y-%m-%d %H:%M:%S')}] ===")

    summary_table = []

    for idx, cfg_path in enumerate(config_paths, 1):
        rel_cfg = cfg_path.relative_to(repo_root) if cfg_path.is_relative_to(repo_root) else cfg_path
        print(f"\n[{idx}/{len(config_paths)}] Processing config: {rel_cfg}")

        t0 = time.time()
        status = "SUCCESS"
        err_msg = ""

        try:
            run_experiment(
                cfg_or_path=cfg_path,
                overrides_already_applied=False,
                device_override=args.device,
                force=args.force,
            )
        except Exception as e:
            status = "FAILED"
            err_msg = str(e)
            print(f"[ERROR] Config '{rel_cfg}' failed: {e}")
            if not args.continue_on_error:
                traceback.print_exc()

        elapsed = time.time() - t0
        summary_table.append((str(rel_cfg), status, elapsed, err_msg))

        entry_line = f"{idx:02d}. {rel_cfg} -> {status} ({elapsed:.2f}s)" + (
            f" [Error: {err_msg}]" if err_msg else ""
        )
        log_lines.append(entry_line)

        if status == "FAILED" and not args.continue_on_error:
            print("Stopping batch run due to error (--continue-on-error is false).")
            break

    # Print summary table
    print("\n" + "=" * 60)
    print("BATCH RUN SUMMARY RESULTS")
    print("=" * 60)
    print(f"{'Config File':<40} | {'Status':<8} | {'Time (s)':<8}")
    print("-" * 60)
    for cfg_file, status, elapsed, _ in summary_table:
        print(f"{cfg_file:<40} | {status:<8} | {elapsed:<8.2f}")
    print("=" * 60)

    # Append to batch_log.txt
    with open(batch_log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n\n")

    print(f"Batch log saved to: {batch_log_path}")


if __name__ == "__main__":
    main()
