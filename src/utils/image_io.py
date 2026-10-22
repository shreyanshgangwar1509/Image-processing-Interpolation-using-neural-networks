"""Image I/O and conversion utilities."""
from pathlib import Path
from typing import List, Union
import numpy as np
from PIL import Image
import torch


def load_image(path: Union[str, Path]) -> torch.Tensor:
    """Loads an image file into a float32 tensor of shape (3, H, W) in range [0, 1].

    Converts grayscale, RGBA, or palette images to RGB.
    """
    img_path = Path(path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found at: {img_path}")

    with Image.open(img_path) as img:
        img_rgb = img.convert("RGB")
        arr = np.array(img_rgb, dtype=np.float32) / 255.0  # (H, W, 3)

    # Convert to C, H, W
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


def save_image(tensor: torch.Tensor, path: Union[str, Path]) -> None:
    """Saves a float tensor (3, H, W) in range [0, 1] to disk as a PNG image."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    tensor_clamped = tensor.detach().cpu().clamp(0.0, 1.0)
    arr = (tensor_clamped.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    img.save(save_path, format="PNG")


def list_images(folder: Union[str, Path]) -> List[Path]:
    """Returns a sorted list of image file paths (.png, .jpg, .jpeg, .bmp) in folder."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        return []

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    images = [
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in valid_exts
    ]
    return sorted(images)


def rgb_to_y(tensor: torch.Tensor) -> torch.Tensor:
    """Converts RGB tensor in range [0, 1] to ITU-R BT.601 Y channel in range [16, 235].

    Supports input shape (3, H, W) or (B, 3, H, W).
    Returns tensor with 1 channel: (1, H, W) or (B, 1, H, W).
    """
    if tensor.ndim == 3:
        r = tensor[0:1, :, :]
        g = tensor[1:2, :, :]
        b = tensor[2:3, :, :]
    elif tensor.ndim == 4:
        r = tensor[:, 0:1, :, :]
        g = tensor[:, 1:2, :, :]
        b = tensor[:, 2:3, :, :]
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tensor.shape}")

    y = 16.0 + 65.481 * r + 128.553 * g + 24.966 * b
    return y
