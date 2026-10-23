"""Download the DIV2K HR images from the official ETH Zurich server.

Result: data/DIV2K_train_HR/0001.png ... 0800.png  (and optionally data/DIV2K_valid_HR/)
Only the HR images are kept; no LR images are downloaded (the project makes LR itself).

Usage:
    python download_div2k.py                       # train HR only -> ./data/DIV2K_train_HR
    python download_div2k.py --out data --splits train valid
    python download_div2k.py --keep-zip            # do not delete the zip afterwards

DIV2K is for academic research use only. Cite the NTIRE 2017 DIV2K paper (Agustsson & Timofte).
"""
import argparse
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

BASE = "data.vision.ee.ethz.ch/cvl/DIV2K"
SPLITS = {
    # name: (zip file, folder name, expected image count)
    "train": ("DIV2K_train_HR.zip", "DIV2K_train_HR", 800),
    "valid": ("DIV2K_valid_HR.zip", "DIV2K_valid_HR", 100),
}


def download(zip_name: str, dest: Path) -> None:
    """Download to dest (resumes a partial '.part' file). Tries https, then http."""
    part = dest.with_suffix(dest.suffix + ".part")
    last_err = None
    for scheme in ("https", "http"):
        url = f"{scheme}://{BASE}/{zip_name}"
        try:
            start = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url)
            if start:
                req.add_header("Range", f"bytes={start}-")
            with urllib.request.urlopen(req, timeout=60) as resp:
                if start and resp.status != 206:  # server ignored Range -> restart
                    start = 0
                total = start + int(resp.headers.get("Content-Length", 0))
                mode = "ab" if start else "wb"
                done, t0 = start, time.time()
                print(f"Downloading {url}")
                with open(part, mode) as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        speed = (done - start) / max(time.time() - t0, 1e-6) / 1e6
                        if total:
                            pct = 100 * done / total
                            bar = "#" * int(pct / 2.5) + "-" * (40 - int(pct / 2.5))
                            eta = (total - done) / 1e6 / max(speed, 1e-6)
                            print(f"\r  [{bar}] {pct:5.1f}%  {done/1e6:7.1f}/{total/1e6:.1f} MB  "
                                  f"{speed:5.1f} MB/s  ETA {int(eta // 60)}m{int(eta % 60):02d}s ",
                                  end="", flush=True)
                        else:  # server did not report a size
                            print(f"\r  {done/1e6:8.1f} MB  {speed:5.1f} MB/s", end="", flush=True)
                print()
            part.replace(dest)
            return
        except Exception as e:  # try the next scheme
            last_err = e
            print(f"\n  {scheme} failed: {e}")
    sys.exit(f"Could not download {zip_name}: {last_err}\n"
             f"Download it manually from https://{BASE}/ and unzip into the data folder.")


def extract_hr(zip_path: Path, out_dir: Path) -> int:
    """Write every PNG in the zip flat into out_dir. Returns the number of images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for m in zf.infolist():
            if m.is_dir() or not m.filename.lower().endswith(".png"):
                continue
            with zf.open(m) as src, open(out_dir / Path(m.filename).name, "wb") as dst:
                dst.write(src.read())
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data", help="parent folder (default: data)")
    ap.add_argument("--splits", nargs="+", choices=list(SPLITS), default=["train"])
    ap.add_argument("--keep-zip", action="store_true", help="keep the downloaded zip")
    ap.add_argument("--force", action="store_true", help="redo even if the folder already looks complete")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        zip_name, folder, expected = SPLITS[split]
        target = out / folder
        have = len(list(target.glob("*.png"))) if target.exists() else 0
        if have >= expected and not args.force:
            print(f"[{split}] {target} already has {have} images, skipping (use --force to redo).")
            continue
        zip_path = out / zip_name
        if not zip_path.exists():
            download(zip_name, zip_path)
        print(f"[{split}] extracting to {target} ...")
        n = extract_hr(zip_path, target)
        print(f"[{split}] done: {n} images (expected {expected}).")
        if n != expected:
            print("  WARNING: image count differs from expected, the download may be incomplete.")
        elif not args.keep_zip:
            zip_path.unlink()


if __name__ == "__main__":
    main()