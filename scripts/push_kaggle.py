#!/usr/bin/env python3
"""Cross-platform helper script to configure and push Kaggle training jobs.

Credentials and settings are inferred automatically with the following priority:
  1. CLI arguments (--username, --key, --branch, etc.)
  2. Local .env file (KAGGLE_USERNAME, KAGGLE_KEY, etc.)
  3. Environment variables (KAGGLE_USERNAME, KAGGLE_KEY)
  4. Local Kaggle config files (kaggle.json or credentials.json in .kaggle directory)
"""

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
DEFAULT_REPO_URL = "https://github.com/sureshkumarhere/Image-processing-Interpolation-using-neural-networks.git"
DEFAULT_BRANCH = "exp-v6"


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


def get_kaggle_config_dir() -> Path:
    """Return the cross-platform path to the Kaggle configuration directory."""
    env_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".kaggle"


def infer_kaggle_credentials() -> tuple[str | None, str | None]:
    """Infers username and key/token from kaggle.json or credentials.json cross-platform."""
    cfg_dir = get_kaggle_config_dir()
    username, key = None, None

    for fname in ["kaggle.json", "credentials.json"]:
        p = cfg_dir / fname
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if not username and "username" in data:
                    username = data["username"]
                if not key:
                    key = data.get("key") or data.get("access_token")
            except Exception:
                pass
    return username, key


def infer_git_repo_url() -> str:
    """Infers HTTPS git remote origin URL suitable for remote cloning by Kaggle."""
    try:
        res = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        url = res.stdout.strip()
        if url:
            if url.startswith("git@github.com:"):
                url = "https://github.com/" + url[len("git@github.com:"):]
            return url
    except Exception:
        pass
    return DEFAULT_REPO_URL


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
    parser = argparse.ArgumentParser(description="Cross-platform Kaggle experiment manager using .env or .kaggle")
    parser.add_argument("--username", help="Kaggle username (overrides .env / .kaggle)")
    parser.add_argument("--key", help="Kaggle API key (overrides .env / .kaggle)")
    parser.add_argument("--branch", help="Git branch to clone on Kaggle (overrides .env KAGGLE_BRANCH)")
    parser.add_argument("--repo-url", help="Repo URL to clone on Kaggle (overrides .env KAGGLE_REPO_URL)")
    parser.add_argument("--configs", nargs="+", help="Config YAML paths to run (overrides .env KAGGLE_CONFIGS)")
    parser.add_argument("--prepare-only", action="store_true", help="Generate metadata and runner config without pushing")
    parser.add_argument("--status", action="store_true", help="Check status of remote Kaggle kernel")
    parser.add_argument("--output", action="store_true", help="Download output of remote Kaggle kernel to kaggle_out/")

    args = parser.parse_args()

    env = load_env(ENV_FILE)
    inferred_user, inferred_key = infer_kaggle_credentials()

    username = (
        args.username
        or env.get("KAGGLE_USERNAME")
        or os.environ.get("KAGGLE_USERNAME")
        or inferred_user
    )
    key = (
        args.key
        or env.get("KAGGLE_KEY")
        or os.environ.get("KAGGLE_KEY")
        or inferred_key
    )
    branch = (
        args.branch
        or env.get("KAGGLE_BRANCH")
        or DEFAULT_BRANCH
    )
    repo_url = (
        args.repo_url
        or env.get("KAGGLE_REPO_URL")
        or infer_git_repo_url()
    )

    if args.configs:
        configs = args.configs
    elif env.get("KAGGLE_CONFIGS"):
        configs = [c.strip() for c in env["KAGGLE_CONFIGS"].split(",") if c.strip()]
    else:
        configs = ["configs/our_nn_x4_noise15_a1_skip.yaml"]

    if not username:
        sys.exit(
            "Error: Kaggle username could not be determined.\n"
            "Provide it in .env (KAGGLE_USERNAME), through ~/.kaggle/ (or %USERPROFILE%\\.kaggle\\),\n"
            "via environment variable KAGGLE_USERNAME, or pass --username <user>."
        )

    # Set environment variables for Kaggle CLI if present
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

    print(f"\nPushing kernel for user '{username}' to Kaggle...")
    try:
        subprocess.run(["kaggle", "kernels", "push", "-p", str(KAGGLE_DIR)], check=True)
        print("\nKernel pushed successfully!")
        print(f"To monitor status: python scripts/push_kaggle.py --status")
        print(f"To download output: python scripts/push_kaggle.py --output")
    except FileNotFoundError:
        print(
            "\nNote: 'kaggle' CLI not found in PATH. You can install it with `pip install kaggle`.\n"
            "Your kaggle/ folder is fully prepared. You can manually push it via:\n"
            "  kaggle kernels push -p kaggle"
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error pushing kernel: {e}")


if __name__ == "__main__":
    main()
