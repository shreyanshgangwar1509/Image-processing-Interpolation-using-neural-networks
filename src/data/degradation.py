"""Degradation steps and degradation pipeline implementation."""
import math
from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F

from src.data.registry import DEGRADATIONS


@DEGRADATIONS.register("bicubic_downscale")
class BicubicDownscale:
    """Bicubic downscaling degradation step."""

    def __init__(self, scale: int) -> None:
        self.scale = int(scale)

    def describe(self) -> str:
        return f"bicubic_down_x{self.scale}"

    def __call__(
        self, img: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        # img shape: (C, H, W)
        c, h, w = img.shape
        img_4d = img.unsqueeze(0)
        out_4d = F.interpolate(
            img_4d,
            scale_factor=1.0 / self.scale,
            mode="bicubic",
            antialias=True,
            align_corners=False,
        )
        return out_4d.squeeze(0).clamp(0.0, 1.0)


@DEGRADATIONS.register("gaussian_noise")
class GaussianNoise:
    """Gaussian noise addition degradation step."""

    def __init__(self, sigma: float) -> None:
        self.sigma = float(sigma)

    def describe(self) -> str:
        s_str = int(self.sigma) if self.sigma == int(self.sigma) else self.sigma
        return f"gaussian_noise_s{s_str}"

    def __call__(
        self, img: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        noise_std = self.sigma / 255.0
        if generator is not None:
            noise = torch.randn(img.shape, generator=generator, dtype=img.dtype, device=img.device)
        else:
            noise = torch.randn_like(img)
        out = img + noise * noise_std
        return out.clamp(0.0, 1.0)


@DEGRADATIONS.register("gaussian_blur")
class GaussianBlur:
    """Separable Gaussian blur degradation step."""

    def __init__(self, sigma: float, kernel_size: Optional[int] = None) -> None:
        self.sigma = float(sigma)
        if kernel_size is None:
            ks = 2 * math.ceil(3 * self.sigma) + 1
            if ks % 2 == 0:
                ks += 1
            self.kernel_size = ks
        else:
            self.kernel_size = int(kernel_size)

    def describe(self) -> str:
        s_str = int(self.sigma) if self.sigma == int(self.sigma) else self.sigma
        return f"gaussian_blur_s{s_str}"

    def __call__(
        self, img: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        if self.sigma <= 0 or self.kernel_size <= 1:
            return img.clone()

        radius = self.kernel_size // 2
        x = torch.arange(-radius, radius + 1, dtype=img.dtype, device=img.device)
        kernel_1d = torch.exp(-0.5 * (x / self.sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Separable convolution: horizontal then vertical
        c, h, w = img.shape
        img_4d = img.unsqueeze(0)  # (1, C, H, W)

        # Horizontal kernel (1, 1, 1, K)
        k_h = kernel_1d.view(1, 1, 1, self.kernel_size).repeat(c, 1, 1, 1)
        # Vertical kernel (1, 1, K, 1)
        k_v = kernel_1d.view(1, 1, self.kernel_size, 1).repeat(c, 1, 1, 1)

        padded = F.pad(img_4d, (radius, radius, radius, radius), mode="reflect")
        # Apply horizontal conv then vertical conv using depthwise conv (groups=c)
        out = F.conv2d(padded, k_h, groups=c)
        out = F.conv2d(out, k_v, groups=c)

        return out.squeeze(0).clamp(0.0, 1.0)


@DEGRADATIONS.register("salt_pepper")
class SaltPepper:
    """Salt and pepper noise degradation step."""

    def __init__(self, amount: float) -> None:
        self.amount = float(amount)

    def describe(self) -> str:
        return f"salt_pepper_a{self.amount}"

    def __call__(
        self, img: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        if self.amount <= 0:
            return img.clone()

        c, h, w = img.shape
        if generator is not None:
            rand_grid = torch.rand((h, w), generator=generator, dtype=img.dtype, device=img.device)
        else:
            rand_grid = torch.rand((h, w), dtype=img.dtype, device=img.device)

        out = img.clone()
        pepper_mask = rand_grid < (self.amount / 2.0)
        salt_mask = (rand_grid >= (self.amount / 2.0)) & (rand_grid < self.amount)

        out[:, pepper_mask] = 0.0
        out[:, salt_mask] = 1.0

        return out.clamp(0.0, 1.0)


class DegradationPipeline:
    """Pipeline executing a sequence of degradation steps."""

    def __init__(self, steps: List[Dict[str, Any]], scale: int) -> None:
        self.scale = scale
        self.pipeline_steps = []

        for step_cfg in steps:
            params = dict(step_cfg)
            name = params.pop("name")
            # If step requires scale (e.g., bicubic_downscale) and scale is not provided, inject it
            if name == "bicubic_downscale" and "scale" not in params:
                params["scale"] = scale

            step_obj = DEGRADATIONS.build(name, **params)
            self.pipeline_steps.append(step_obj)

    @property
    def tag(self) -> str:
        if not self.pipeline_steps:
            return "none"
        return "+".join(step.describe() for step in self.pipeline_steps)

    def __call__(
        self, img: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        curr = img
        for step in self.pipeline_steps:
            curr = step(curr, generator=generator)
        return curr
