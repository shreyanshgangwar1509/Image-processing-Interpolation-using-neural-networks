"""Image folder dataset supporting training and evaluation modes."""
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset

from src.data.degradation import DegradationPipeline
from src.data.registry import DATASETS
from src.utils.image_io import load_image, list_images


@DATASETS.register("image_folder")
class ImageFolderDataset(Dataset):
    """Dataset for HR image folders with online degradation pipeline."""

    def __init__(
        self,
        root: Union[str, Path],
        scale: int,
        degradation: List[Dict[str, Any]],
        mode: str = "eval",
        patch_size: Optional[int] = None,
        augment: bool = False,
        repeat: int = 1,
        max_images: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.scale = scale
        self.mode = mode
        self.patch_size = patch_size
        self.augment = augment
        self.repeat = repeat
        self.seed = seed

        if not self.root.exists() or not self.root.is_dir():
            raise FileNotFoundError(
                f"Data folder '{self.root}' does not exist or is not a directory. "
                "Expected images in this folder. See docs/SETUP.md for dataset setup instructions."
            )

        self.image_paths = list_images(self.root)
        if not self.image_paths:
            raise ValueError(
                f"No images found in data folder '{self.root}'. "
                "Supported extensions: .png, .jpg, .jpeg, .bmp. "
                "See docs/SETUP.md for dataset setup instructions."
            )

        if max_images is not None and max_images > 0:
            self.image_paths = self.image_paths[:max_images]

        if self.mode == "train":
            if self.patch_size is None:
                raise ValueError("patch_size must be specified for train mode.")
            if self.patch_size % self.scale != 0:
                raise ValueError(
                    f"patch_size ({self.patch_size}) must be divisible by scale ({self.scale})."
                )

        self.degradation_pipeline = DegradationPipeline(degradation, scale=self.scale)

    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.image_paths) * self.repeat
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        img_idx = idx % len(self.image_paths)
        img_path = self.image_paths[img_idx]
        hr_img = load_image(img_path)  # Tensor (3, H, W)
        c, h, w = hr_img.shape

        if self.mode == "train":
            p = self.patch_size
            if h < p or w < p:
                raise ValueError(
                    f"Image '{img_path.name}' size ({w}x{h}) is smaller than patch_size ({p}x{p})."
                )

            # Random crop
            top = random.randint(0, h - p)
            left = random.randint(0, w - p)
            hr_patch = hr_img[:, top : top + p, left : left + p]

            # Augmentation: flip & rotation
            if self.augment:
                if random.random() < 0.5:
                    hr_patch = torch.flip(hr_patch, dims=[2])  # Horizontal flip
                rot_k = random.randint(0, 3)
                if rot_k > 0:
                    hr_patch = torch.rot90(hr_patch, k=rot_k, dims=[1, 2])

            lr_patch = self.degradation_pipeline(hr_patch, generator=None)

            # Verify size ratio
            lr_c, lr_h, lr_w = lr_patch.shape
            if lr_h != p // self.scale or lr_w != p // self.scale:
                raise ValueError(
                    f"LR size ({lr_w}x{lr_h}) is not equal to HR size / scale ({p // self.scale}x{p // self.scale})."
                )

            return {
                "lr": lr_patch,
                "hr": hr_patch,
                "name": img_path.stem,
            }

        else:  # eval mode
            # Crop bottom/right so H and W are divisible by scale
            crop_h = (h // self.scale) * self.scale
            crop_w = (w // self.scale) * self.scale
            hr_cropped = hr_img[:, :crop_h, :crop_w]

            gen = torch.Generator().manual_seed(self.seed + img_idx)
            lr_img = self.degradation_pipeline(hr_cropped, generator=gen)

            lr_c, lr_h, lr_w = lr_img.shape
            if lr_h != crop_h // self.scale or lr_w != crop_w // self.scale:
                raise ValueError(
                    f"LR size ({lr_w}x{lr_h}) is not equal to HR size / scale ({crop_w // self.scale}x{crop_h // self.scale})."
                )

            return {
                "lr": lr_img,
                "hr": hr_cropped,
                "name": img_path.stem,
            }
