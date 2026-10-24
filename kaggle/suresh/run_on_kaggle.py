"""Run sr-project experiments on Kaggle (pushed as a "script" kernel).

What it does, in order:
  1. gets the project code (from a Kaggle dataset that contains scripts/train.py, or by git clone),
  2. links the image folders from the attached Kaggle dataset into <repo>/data/,
  3. runs each config in CONFIGS one after another with scripts/train.py,
  4. copies the result folders created by THIS run (not the ones that came with the repo) to
     /kaggle/working/results every few minutes and after every config, so partial work is kept
     even if the session is killed. Download it with `kaggle kernels output`.
     Run scripts/aggregate_results.py on your own computer afterwards.

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
CONFIGS = [
    "configs/our_nn_x4_clean.yaml",
]

CONFIGS = ["configs/our_nn_x4_clean.yaml"]
# EXTRA_SET = ["data.loader.num_workers=4", "data.train.repeat=3"]
EXTRA_SET = ["data.loader.num_workers=4", "data.train.repeat=3", "evaluation.save_samples=1"]

# Do not START a new config after this many hours (Kaggle sessions end at about 9 h).
MAX_HOURS = 8.0
# ====================================================================

INPUT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
WORK = Path(os.environ.get("KAGGLE_WORK_DIR", "/kaggle/working"))
REPO = Path(os.environ.get("REPO_DIR", "/tmp/sr-project"))   # not under /working, so it is not saved as output
DATA_FOLDERS = ["DIV2K_train_HR", "Set5", "Set14", "BSD100", "Urban100"]
# SYNC_EVERY = float(os.environ.get("SYNC_EVERY", 300))   # seconds between result copies during training
# RUN_START = 0.0   # set in main() after the repo is cloned; only results newer than this are copied out


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


def save_results() -> int:
    """Copy to /kaggle/working/results only the experiment folders that this run created or
    changed (any file newer than RUN_START). Results that came with the repo are left out."""
    src = REPO / "results"
    n = 0
    if not src.exists():
        return n
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        newest = max((f.stat().st_mtime for f in d.rglob("*") if f.is_file()), default=0.0)
        if newest >= RUN_START:
            shutil.copytree(d, WORK / "results" / d.name, dirs_exist_ok=True)
            n += 1
    return n


def run_and_sync(cmd, cwd, every: float) -> int:
    """Run a command; while it runs, copy results/ to /kaggle/working every `every` seconds,
    so logs and checkpoints survive even if the session is killed in the middle of training."""
    print("$", " ".join(map(str, cmd)), flush=True)
    p = subprocess.Popen(list(map(str, cmd)), cwd=cwd)
    last = time.time()
    while p.poll() is None:
        time.sleep(2)
        if time.time() - last >= every:
            save_results()
            last = time.time()
    return p.returncode


def main() -> None:
    global RUN_START
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    print(f"CONFIGS   = {CONFIGS}\nEXTRA_SET = {EXTRA_SET}\nMAX_HOURS = {MAX_HOURS}", flush=True)
    sh(["nvidia-smi"])
    get_code()
    missing = link_data()
    RUN_START = time.time()        # everything in results/ older than this came with the repo
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
        rc = run_and_sync(cmd, REPO, SYNC_EVERY)
        rows.append((cfg, "ok" if rc == 0 else f"FAILED (exit {rc})", time.time() - t1))
        save_results()                          # keep partial work after every config

    n_saved = save_results()
    if n_saved == 0:
        print("WARNING: no new result folders were produced (did every config fail, or was it skipped "
              "because skip_if_done is true and the folder already exists in the repo?)")

    lines = ["config | status | minutes"] + [f"{c} | {s} | {t/60:.1f}" for c, s, t in rows]
    lines.append(f"total hours: {(time.time() - t0) / 3600:.2f}")
    (WORK / "run_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()