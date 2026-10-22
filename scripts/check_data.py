"""Checks dataset image folders for existence, counts, min/median/max sizes, and small images."""
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image

# Insert repo root into sys.path
repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.utils.image_io import list_images


def check_folder(folder: Path) -> None:
    print(f"\nChecking dataset folder: {folder}")
    if not folder.exists() or not folder.is_dir():
        print(f"  [ERROR] Folder '{folder}' does not exist or is not a directory.")
        return

    images = list_images(folder)
    if not images:
        print(f"  [WARNING] No images (.png, .jpg, .jpeg, .bmp) found in '{folder}'.")
        return

    widths = []
    heights = []
    small_images = []

    for img_path in images:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                if w < 96 or h < 96:
                    small_images.append((img_path.name, w, h))
        except Exception as e:
            print(f"  [ERROR] Failed to open image {img_path.name}: {e}")

    if widths and heights:
        med_w, med_h = int(np.median(widths)), int(np.median(heights))
        min_w, min_h = min(widths), min(heights)
        max_w, max_h = max(widths), max(heights)

        print(f"  Total images found: {len(widths)}")
        print(f"  Min size   : {min_w} x {min_h} px")
        print(f"  Median size: {med_w} x {med_h} px")
        print(f"  Max size   : {max_w} x {max_h} px")

        if small_images:
            print(f"  [WARNING] {len(small_images)} image(s) smaller than 96px on a side:")
            for name, w, h in small_images[:5]:
                print(f"    - {name}: {w}x{h} px")
            if len(small_images) > 5:
                print(f"    ... and {len(small_images) - 5} more.")
        else:
            print("  [OK] All images are >= 96px on both sides.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dataset image folders.")
    parser.add_argument(
        "folders", nargs="+", type=str, help="Folder path(s) to check"
    )
    args = parser.parse_args()

    for folder_str in args.folders:
        check_folder(Path(folder_str))


if __name__ == "__main__":
    main()
