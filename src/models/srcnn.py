"""SRCNN model architecture."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.registry import MODELS


@MODELS.register("srcnn")
class SRCNN(nn.Module):
    """Super-Resolution Convolutional Neural Network (SRCNN)."""

    trainable: bool = True

    def __init__(self, scale: int, n1: int = 64, n2: int = 32) -> None:
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(3, n1, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(n2, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        # Pre-upscale LR input with bicubic interpolation to target resolution
        x = F.interpolate(lr, scale_factor=self.scale, mode="bicubic", align_corners=False)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
