"""Classical upscaling interpolation methods (Nearest, Bilinear, Bicubic, Lanczos)."""
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.registry import MODELS


class _BaseClassicalModel(nn.Module):
    trainable: bool = False

    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale = scale


@MODELS.register("nearest")
class NearestModel(_BaseClassicalModel):
    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        return F.interpolate(lr, scale_factor=self.scale, mode="nearest")


@MODELS.register("bilinear")
class BilinearModel(_BaseClassicalModel):
    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        return F.interpolate(lr, scale_factor=self.scale, mode="bilinear", align_corners=False)


@MODELS.register("bicubic")
class BicubicModel(_BaseClassicalModel):
    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        return F.interpolate(lr, scale_factor=self.scale, mode="bicubic", align_corners=False)


@MODELS.register("lanczos")
class LanczosModel(_BaseClassicalModel):
    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        # Per-image Lanczos upscaling via PIL
        b, c, h, w = lr.shape
        target_h, target_w = h * self.scale, w * self.scale

        output_tensors = []
        for i in range(b):
            img_tensor = lr[i].detach().cpu().clamp(0.0, 1.0)
            arr = (img_tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            pil_img = Image.fromarray(arr, mode="RGB")
            resized_pil = pil_img.resize((target_w, target_h), resample=Image.LANCZOS)
            arr_res = np.array(resized_pil, dtype=np.float32) / 255.0  # (H, W, 3)
            out_tensor = torch.from_numpy(arr_res).permute(2, 0, 1)  # (3, H, W)
            output_tensors.append(out_tensor.to(lr.device))

        return torch.stack(output_tensors, dim=0)
