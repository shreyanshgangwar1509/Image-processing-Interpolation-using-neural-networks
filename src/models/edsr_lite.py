"""EDSR-lite model architecture."""
import torch
import torch.nn as nn

from src.models.registry import MODELS


class ResBlock(nn.Module):
    def __init__(self, n_feats: int, res_scale: float = 0.1) -> None:
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x) * self.res_scale
        return x + res


@MODELS.register("edsr_lite")
class EDSRLite(nn.Module):
    """Enhanced Deep Super-Resolution network lite version (EDSR-lite)."""

    trainable: bool = True

    def __init__(
        self, scale: int, n_feats: int = 32, n_resblocks: int = 8, res_scale: float = 0.1
    ) -> None:
        super().__init__()
        self.scale = scale

        # Head
        self.head = nn.Conv2d(3, n_feats, kernel_size=3, padding=1)

        # Body
        resblocks = [ResBlock(n_feats, res_scale=res_scale) for _ in range(n_resblocks)]
        self.body = nn.Sequential(*resblocks)
        self.body_conv = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)

        # Upsampler
        upsample_layers = []
        if scale in (2, 3):
            upsample_layers.append(
                nn.Conv2d(n_feats, n_feats * (scale ** 2), kernel_size=3, padding=1)
            )
            upsample_layers.append(nn.PixelShuffle(scale))
        elif scale == 4:
            for _ in range(2):
                upsample_layers.append(
                    nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1)
                )
                upsample_layers.append(nn.PixelShuffle(2))
        else:
            raise ValueError(f"EDSR-lite only supports scales 2, 3, 4. Got scale={scale}.")

        self.upsample = nn.Sequential(*upsample_layers)

        # Tail
        self.tail = nn.Conv2d(n_feats, 3, kernel_size=3, padding=1)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        head_feat = self.head(lr)
        body_feat = self.body_conv(self.body(head_feat))
        feat = head_feat + body_feat
        up_feat = self.upsample(feat)
        out = self.tail(up_feat)
        return out
