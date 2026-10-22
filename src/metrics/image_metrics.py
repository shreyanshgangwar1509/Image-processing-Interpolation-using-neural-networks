"""Image metrics: PSNR, SSIM, LPIPS."""
import math
from typing import Optional
import numpy as np
import torch
from skimage.metrics import structural_similarity

from src.metrics.registry import METRICS
from src.utils.image_io import rgb_to_y


def _crop_border_tensor(tensor: torch.Tensor, border: int) -> torch.Tensor:
    """Crops border pixels from height and width of tensor (1, C, H, W)."""
    if border <= 0:
        return tensor
    return tensor[:, :, border:-border, border:-border]


@METRICS.register("psnr")
class PSNRMetric:
    """Peak Signal-to-Noise Ratio metric."""

    def __init__(self, y_channel: bool = True, crop_border: Optional[int] = None) -> None:
        self.y_channel = y_channel
        self.crop_border = crop_border

    def __call__(self, sr: torch.Tensor, hr: torch.Tensor, scale: int) -> float:
        # Quantize SR to 8-bit
        sr_quant = (sr.clamp(0.0, 1.0) * 255.0).round() / 255.0
        hr_quant = hr.clamp(0.0, 1.0)

        border = self.crop_border if self.crop_border is not None else scale
        sr_cropped = _crop_border_tensor(sr_quant, border)
        hr_cropped = _crop_border_tensor(hr_quant, border)

        if self.y_channel:
            sr_proc = rgb_to_y(sr_cropped)  # Range [16, 235]
            hr_proc = rgb_to_y(hr_cropped)
        else:
            sr_proc = sr_cropped * 255.0
            hr_proc = hr_cropped * 255.0

        mse = torch.mean((sr_proc - hr_proc) ** 2).item()
        if mse == 0.0:
            return 100.0

        psnr_val = 10.0 * math.log10((255.0 ** 2) / mse)
        return float(psnr_val)


@METRICS.register("ssim")
class SSIMMetric:
    """Structural Similarity Index Metric."""

    def __init__(self, y_channel: bool = True, crop_border: Optional[int] = None) -> None:
        self.y_channel = y_channel
        self.crop_border = crop_border

    def __call__(self, sr: torch.Tensor, hr: torch.Tensor, scale: int) -> float:
        sr_quant = (sr.clamp(0.0, 1.0) * 255.0).round() / 255.0
        hr_quant = hr.clamp(0.0, 1.0)

        border = self.crop_border if self.crop_border is not None else scale
        sr_cropped = _crop_border_tensor(sr_quant, border)
        hr_cropped = _crop_border_tensor(hr_quant, border)

        if self.y_channel:
            sr_y = rgb_to_y(sr_cropped).squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
            hr_y = rgb_to_y(hr_cropped).squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
            ssim_val = structural_similarity(
                sr_y,
                hr_y,
                data_range=255.0,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
            )
        else:
            # RGB array shape: (H, W, 3) in range 0-255
            sr_rgb = (sr_cropped.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0)
            hr_rgb = (hr_cropped.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0)
            ssim_val = structural_similarity(
                sr_rgb,
                hr_rgb,
                data_range=255.0,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
                channel_axis=2,
            )

        return float(ssim_val)


@METRICS.register("lpips")
class LPIPSMetric:
    """Learned Perceptual Image Patch Similarity metric with lazy lpips import."""

    def __init__(self, y_channel: bool = False, crop_border: Optional[int] = None) -> None:
        try:
            import lpips
        except ImportError:
            raise ImportError(
                "lpips package is not installed. Please run `pip install lpips` to use the LPIPS metric."
            )
        self.loss_fn = lpips.LPIPS(net="alex", verbose=False)
        self.crop_border = crop_border

    def __call__(self, sr: torch.Tensor, hr: torch.Tensor, scale: int) -> float:
        sr_quant = (sr.clamp(0.0, 1.0) * 255.0).round() / 255.0
        hr_quant = hr.clamp(0.0, 1.0)

        border = self.crop_border if self.crop_border is not None else scale
        sr_cropped = _crop_border_tensor(sr_quant, border)
        hr_cropped = _crop_border_tensor(hr_quant, border)

        # LPIPS expects input in range [-1, 1]
        sr_input = sr_cropped * 2.0 - 1.0
        hr_input = hr_cropped * 2.0 - 1.0

        with torch.no_grad():
            dist = self.loss_fn(sr_input.cpu(), hr_input.cpu())
        return float(dist.item())
