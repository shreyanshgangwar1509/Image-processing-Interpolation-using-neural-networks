"""This file checks the datasets - if you have the datasets ready then ignore it.

Flatten the SelfExSR benchmark folders so each holds only the HR images.

Before:  data/Set5/image_SRF_4/img_001_SRF_4_HR.png, img_001_SRF_4_LR.png, ...
After:   data/Set5/img_001_SRF_4_HR.png ...   (HR only, directly inside data/Set5)

HR images are identical across image_SRF_2/3/4, so one folder is enough.
By default the image_SRF_* subfolders are left untouched; use --clean to delete them afterwards.

Usage:
    python scripts/prepare_benchmarks.py
    python scripts/prepare_benchmarks.py --data data --datasets Set5 Set14 BSD100 Urban100 --clean
"""
import argparse
import shutil
from pathlib import Path


def prepare(ds_dir: Path, clean: bool) -> int:
    subs = sorted(p for p in ds_dir.glob("image_SRF_*") if p.is_dir())
    if not subs:
        flat = len(list(ds_dir.glob("*.png")))
        print(f"  {ds_dir.name}: no image_SRF_* folders; {flat} images directly inside (nothing to do).")
        return flat
    # prefer SRF_4, otherwise whichever folder exists
    src = next((p for p in subs if p.name.endswith("_4")), subs[0])
    files = sorted(src.glob("*_HR.*"))
    if not files:
        print(f"  {ds_dir.name}: no *_HR files found in {src}. Skipped.")
        return 0
    for f in files:
        shutil.copy2(f, ds_dir / f.name)
    print(f"  {ds_dir.name}: copied {len(files)} HR images from {src.name}.")
    if clean:
        for p in subs:
            shutil.rmtree(p)
        print(f"  {ds_dir.name}: removed {', '.join(p.name for p in subs)}.")
    return len(files)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--datasets", nargs="+", default=["Set5", "Set14", "BSD100", "Urban100"])
    ap.add_argument("--clean", action="store_true", help="delete image_SRF_* folders after copying")
    args = ap.parse_args()

    expected = {"Set5": 5, "Set14": 14, "BSD100": 100, "Urban100": 100}
    for name in args.datasets:
        d = Path(args.data) / name
        if not d.is_dir():
            print(f"  {name}: folder {d} not found, skipped.")
            continue
        n = prepare(d, args.clean)
        if name in expected and n != expected[name]:
            print(f"  WARNING: {name} has {n} images, expected {expected[name]}.")


if __name__ == "__main__":
    main()