"""Generates synthetic structured dummy images for smoke testing."""
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image

# Insert repo root into sys.path
repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def generate_structured_image(width: int, height: int, seed: int) -> Image.Image:
    """Generates a synthetic image with smooth gradients, sharp shapes, and subtle texture."""
    rng = np.random.RandomState(seed)

    # 1. Smooth gradient background
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)

    r_bg = 0.5 + 0.5 * np.sin(2 * np.pi * xx)
    g_bg = 0.5 + 0.5 * np.cos(2 * np.pi * yy)
    b_bg = 0.5 + 0.5 * np.sin(2 * np.pi * (xx + yy))

    img_arr = np.stack([r_bg, g_bg, b_bg], axis=-1)

    # 2. Random shapes with sharp edges
    num_shapes = rng.randint(3, 8)
    for _ in range(num_shapes):
        cx, cy = rng.randint(0, width), rng.randint(0, height)
        radius = rng.randint(10, min(width, height) // 3)
        color = rng.rand(3)

        dist = np.sqrt((xx * width - cx) ** 2 + (yy * height - cy) ** 2)
        mask = dist < radius
        img_arr[mask] = color

    # 3. Random sharp lines
    num_lines = rng.randint(2, 5)
    for _ in range(num_lines):
        color = rng.rand(3)
        if rng.rand() < 0.5:
            # Vertical line
            x_pos = rng.randint(0, width)
            img_arr[:, max(0, x_pos - 2) : min(width, x_pos + 2)] = color
        else:
            # Horizontal line
            y_pos = rng.randint(0, height)
            img_arr[max(0, y_pos - 2) : min(height, y_pos + 2), :] = color

    # 4. Low-amplitude texture noise
    texture = rng.normal(0, 0.02, img_arr.shape)
    img_arr = np.clip(img_arr + texture, 0.0, 1.0)

    arr_uint8 = (img_arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr_uint8, mode="RGB")


def make_dummy_dataset(out_dir: Path, seed: int) -> None:
    rng = np.random.RandomState(seed)

    splits = [
        ("train", 20),
        ("val", 5),
        ("test", 5),
    ]

    for split_name, count in splits:
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            w = rng.randint(128, 257)
            h = rng.randint(128, 257)
            img = generate_structured_image(w, h, seed=seed + i * 100)
            img_path = split_dir / f"img_{i+1:03d}.png"
            img.save(img_path)

    print(f"Dummy dataset successfully created at: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make dummy structured images for testing.")
    parser.add_argument("--out", type=str, default="data/dummy", help="Output folder")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out_path = Path(args.out)
    make_dummy_dataset(out_path, args.seed)


if __name__ == "__main__":
    main()
