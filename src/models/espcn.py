"""ESPCN model architecture."""
import torch
import torch.nn as nn

from src.models.registry import MODELS


@MODELS.register("espcn")
class ESPCN(nn.Module):
    """Efficient Sub-Pixel Convolutional Neural Network (ESPCN)."""

    trainable: bool = True

    def __init__(
        self, scale: int, n1: int = 64, n2: int = 32, activation: str = "tanh"
    ) -> None:
        super().__init__()
        self.scale = scale

        if activation.lower() == "tanh":
            act_layer = nn.Tanh
        elif activation.lower() == "relu":
            act_layer = nn.ReLU
        else:
            raise ValueError(f"Unsupported activation '{activation}'. Expected 'tanh' or 'relu'.")

        self.conv1 = nn.Conv2d(3, n1, kernel_size=5, padding=2)
        self.act1 = act_layer()
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=3, padding=1)
        self.act2 = act_layer()
        self.conv3 = nn.Conv2d(n2, 3 * (scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.conv1(lr))
        x = self.act2(self.conv2(x))
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x
