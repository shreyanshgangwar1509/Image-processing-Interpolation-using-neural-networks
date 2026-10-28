#!/usr/bin/env python3
"""Helper script to configure and push Kaggle training jobs using .env settings."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
KAGGLE_DIR = ROOT_DIR / "kaggle"
ENV_FILE = ROOT_DIR / ".env"
TEMPLATE_METADATA = KAGGLE_DIR / "kernel-metadata.json.template"
TARGET_METADATA = KAGGLE_DIR / "kernel-metadata.json"
TARGET_RUNNER = KAGGLE_DIR / "run_on_kaggle.py"


def load_env(env_path: Path) -> dict:
    """Load simple key=value pairs from a .env file without external dependencies."""
    env = {}
    if not env_path.exists():
        return env
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("\"'")
            env[key] = val
    return env


def generate_metadata(username: str) -> None:
    """Generate kaggle/kernel-metadata.json for the specified username."""
    if TEMPLATE_METADATA.exists():
        content = TEMPLATE_METADATA.read_text(encoding="utf-8")
        content = content.replace("${KAGGLE_USERNAME}", username).replace("{KAGGLE_USERNAME}", username)
        data = json.loads(content)
    else:
        data = {
            "id": f"{username}/sr-run",
            "title": "sr-run",
            "code_file": "run_on_kaggle.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": ["suresh79061/sr-data"],
            "competition_sources": [],
            "kernel_sources": [],
        }
    TARGET_METADATA.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {TARGET_METADATA.relative_to(ROOT_DIR)} for user '{username}'.")


def update_runner_config(repo_url: str, branch: str, configs: list[str]) -> None:
    """Updates REPO_URL, BRANCH, and CONFIGS in kaggle/run_on_kaggle.py."""
    if not TARGET_RUNNER.exists():
        print(f"Warning: {TARGET_RUNNER} not found; skipping runner patch.")
        return

    text = TARGET_RUNNER.read_text(encoding="utf-8")
    text = re.sub(r'REPO_URL\s*=\s*".*?"', f'REPO_URL = "{repo_url}"', text)
    text = re.sub(r'BRANCH\s*=\s*".*?"', f'BRANCH = "{branch}"', text)

    configs_repr = "[\n" + "".join(f'    "{c}",\n' for c in configs) + "]"
    text = re.sub(r"CONFIGS\s*=\s*\[.*?\]", f"CONFIGS = {configs_repr}", text, flags=re.DOTALL)

    TARGET_RUNNER.write_text(text, encoding="utf-8")
    print(f"Updated {TARGET_RUNNER.relative_to(ROOT_DIR)} (branch: '{branch}', {len(configs)} config(s)).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle experiment manager using .env")
    parser.add_argument("--username", help="Kaggle username (overrides .env KAGGLE_USERNAME)")
    parser.add_argument("--key", help="Kaggle API key (overrides .env KAGGLE_KEY)")
    parser.add_argument("--branch", help="Git branch to clone on Kaggle (overrides .env KAGGLE_BRANCH)")
    parser.add_argument("--repo-url", help="Repo URL to clone on Kaggle (overrides .env KAGGLE_REPO_URL)")
    parser.add_argument("--configs", nargs="+", help="Config YAML paths to run (overrides .env KAGGLE_CONFIGS)")
    parser.add_argument("--prepare-only", action="store_true", help="Generate metadata and runner config without pushing")
    parser.add_argument("--status", action="store_true", help="Check status of remote Kaggle kernel")
    parser.add_argument("--output", action="store_true", help="Download output of remote Kaggle kernel to kaggle_out/")

    args = parser.parse_args()

    env = load_env(ENV_FILE)

    username = args.username or env.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_USERNAME")
    key = args.key or env.get("KAGGLE_KEY") or os.environ.get("KAGGLE_KEY")
    branch = args.branch or env.get("KAGGLE_BRANCH", "exp-v6")
    repo_url = args.repo_url or env.get("KAGGLE_REPO_URL", "https://github.com/sureshkumarhere/Image-processing-Interpolation-using-neural-networks.git")

    if args.configs:
        configs = args.configs
    elif env.get("KAGGLE_CONFIGS"):
        configs = [c.strip() for c in env["KAGGLE_CONFIGS"].split(",") if c.strip()]
    else:
        configs = ["configs/our_nn_x4_noise15_a1_skip.yaml"]

    if not username:
        sys.exit(
            "Error: Kaggle username is required. Set KAGGLE_USERNAME in .env, "
            "environment variable, or pass --username <user>."
        )

    # Set environment variables for Kaggle CLI if present in .env
    if key and "KAGGLE_KEY" not in os.environ:
        os.environ["KAGGLE_KEY"] = key
    if username and "KAGGLE_USERNAME" not in os.environ:
        os.environ["KAGGLE_USERNAME"] = username

    kernel_id = f"{username}/sr-run"

    if args.status:
        cmd = ["kaggle", "kernels", "status", kernel_id]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        return

    if args.output:
        out_dir = ROOT_DIR / "kaggle_out"
        out_dir.mkdir(exist_ok=True)
        cmd = ["kaggle", "kernels", "output", kernel_id, "-p", str(out_dir)]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        return

    generate_metadata(username)
    update_runner_config(repo_url, branch, configs)

    if args.prepare_only:
        print("\nPreparation complete. You can inspect kaggle/ and run:")
        print("  kaggle kernels push -p kaggle")
        return

    print("\nPushing kernel to Kaggle...")
    try:
        subprocess.run(["kaggle", "kernels", "push", "-p", str(KAGGLE_DIR)], check=True)
        print(f"\nKernel pushed successfully!")
        print(f"To monitor status: python scripts/push_kaggle.py --status")
        print(f"To download output: python scripts/push_kaggle.py --output")
    except FileNotFoundError:
        print(
            "\nNote: 'kaggle' CLI not found in PATH. You can install it with `pip install kaggle`.\n"
            f"Your kaggle/ folder is fully prepared. You can manually push it via:\n"
            f"  kaggle kernels push -p kaggle"
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error pushing kernel: {e}")


if __name__ == "__main__":
    main()
