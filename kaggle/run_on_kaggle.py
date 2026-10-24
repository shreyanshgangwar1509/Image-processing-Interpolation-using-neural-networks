"""Run sr-project experiments on Kaggle (pushed as a "script" kernel).

What it does, in order:
  1. gets the project code (from a Kaggle dataset that contains scripts/train.py, or by git clone),
  2. links the image folders from the attached Kaggle dataset into <repo>/data/,
  3. runs each config in CONFIGS one after another with scripts/train.py,
  4. copies results/ to /kaggle/working/results after EVERY config, so partial work is
     kept even if the session is killed. Download it with `kaggle kernels output`.

Edit the block below, then `kaggle kernels push -p kaggle`.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ============================ EDIT THESE ============================
REPO_URL = "https://github.com/sureshkumarhere/Image-processing-Interpolation-using-neural-networks.git"   # must be public, or use a code dataset
BRANCH = "main"

# Configs to run, in this order. Keep one stage per push (see the steps in the chat).
# CONFIGS = [
#     "configs/our_nn_x4_clean.yaml",
# ]

CONFIGS = ["configs/our_nn_x4_clean.yaml"]
EXTRA_SET = ["training.epochs=2", "data.train.max_images=50"]
# Extra overrides for every config, e.g. a quick trial:
#   ["training.epochs=2", "data.train.max_images=50"]
EXTRA_SET: list = []

# Do not START a new config after this many hours (Kaggle sessions end at about 9 h).
MAX_HOURS = 8.0
# ====================================================================

INPUT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
WORK = Path(os.environ.get("KAGGLE_WORK_DIR", "/kaggle/working"))
REPO = Path(os.environ.get("REPO_DIR", "/tmp/sr-project"))   # not under /working, so it is not saved as output
DATA_FOLDERS = ["DIV2K_train_HR", "Set5", "Set14", "BSD100", "Urban100"]


def sh(cmd, **kw):
    print("$", " ".join(map(str, cmd)), flush=True)
    try:
        return subprocess.run(list(map(str, cmd)), **kw)
    except FileNotFoundError:                  # e.g. nvidia-smi when no GPU is attached
        print(f"(command not found: {cmd[0]})")
        return subprocess.CompletedProcess(cmd, 127)


def find_folder(name: str):
    """Find a directory called `name` anywhere under the Kaggle input folder."""
    for root, dirs, _ in os.walk(INPUT):
        if name in dirs:
            return Path(root) / name
    return None


def find_code():
    """A folder under /kaggle/input that looks like the project (has scripts/train.py)."""
    for root, dirs, files in os.walk(INPUT):
        if (Path(root) / "scripts" / "train.py").exists():
            return Path(root)
        if root.count(os.sep) - str(INPUT).count(os.sep) >= 3:
            dirs[:] = []                       # do not descend forever
    return None


def get_code() -> None:
    if REPO.exists():
        shutil.rmtree(REPO)
    code = find_code()
    if code is not None:
        print(f"Using project code from dataset folder {code}")
        shutil.copytree(code, REPO)
        return
    r = sh(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, REPO])
    if r.returncode != 0:
        sys.exit("git clone failed. Check REPO_URL / BRANCH, that the repo is public and that "
                 "Internet is enabled for this kernel (enable_internet in kernel-metadata.json).")


def link_data() -> list:
    (REPO / "data").mkdir(exist_ok=True)
    missing = []
    for name in DATA_FOLDERS:
        src = find_folder(name)
        dst = REPO / "data" / name
        if src is None:
            missing.append(name)
            print(f"WARNING: folder '{name}' not found under {INPUT}")
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink() if dst.is_symlink() else shutil.rmtree(dst)
        os.symlink(src, dst, target_is_directory=True)
        print(f"linked data/{name} -> {src}")
    return missing


def save_results() -> None:
    src = REPO / "results"
    if src.exists():
        shutil.copytree(src, WORK / "results", dirs_exist_ok=True)


def main() -> None:
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    sh(["nvidia-smi"])
    get_code()
    missing = link_data()
    if missing:
        print(f"Missing data folders: {missing}. Configs that need them will fail.")
    sh([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "scikit-image", "tqdm", "pandas", "pillow"])

    rows = []
    for cfg in CONFIGS:
        hours = (time.time() - t0) / 3600
        if hours > MAX_HOURS:
            rows.append((cfg, "SKIPPED (time budget)", 0.0))
            continue
        t1 = time.time()
        cmd = [sys.executable, "scripts/train.py", "--config", cfg]
        for s in EXTRA_SET:
            cmd += ["--set", s]
        r = sh(cmd, cwd=REPO)
        rows.append((cfg, "ok" if r.returncode == 0 else f"FAILED (exit {r.returncode})", time.time() - t1))
        save_results()                          # keep partial work after every config

    if (REPO / "scripts" / "aggregate_results.py").exists():
        sh([sys.executable, "scripts/aggregate_results.py"], cwd=REPO)
    save_results()

    lines = ["config | status | minutes"] + [f"{c} | {s} | {t/60:.1f}" for c, s, t in rows]
    lines.append(f"total hours: {(time.time() - t0) / 3600:.2f}")
    (WORK / "run_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()