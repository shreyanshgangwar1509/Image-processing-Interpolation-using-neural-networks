"""OurNN super-resolution model architecture with optional edge branch, attention, and bicubic skip.

Model Overview:
`our_nn` is a lightweight residual neural network for image super-resolution supporting
multi-scale upscaling (x2, x3, x4) and configurable structural components.

Component Flags:
- `bicubic_skip`: If True, adds bicubic upscaled input directly to the network output.
- `attention`: If True, applies Channel Attention (Squeeze-and-Excitation) in residual blocks.
- `edge_branch`: If True, extracts luminance and fixed Sobel/Laplacian edge features, fusioning them with head features.
- `zero_init_tail`: If True (and `bicubic_skip` is True), initializes tail conv weights and bias to zero.

Related Ideas:
- Residual learning on top of a bicubic upscale as in VDSR (Kim et al.).
- Residual blocks without batch normalization as in EDSR (Lim et al.).
- Channel attention mechanism as in Squeeze-and-Excitation / RCAN (Zhang et al.).
- Sub-pixel convolution upsampling via PixelShuffle as in ESPCN (Shi et al.).

Contribution:
The specific modular combination of these techniques alongside a fixed edge-prior fusion branch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.registry import MODELS


class ChannelAttention(nn.Module):
    """Channel attention module (Squeeze-and-Excitation)."""

    def __init__(self, n_feats: int, reduction: int = 4) -> None:
        super().__init__()
        mid_feats = max(1, n_feats // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(n_feats, mid_feats, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_feats, n_feats, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = self.conv_du(self.avg_pool(x))
        return x * att


class OurResBlock(nn.Module):
    """Residual block with optional channel attention."""

    def __init__(
        self,
        n_feats: int,
        res_scale: float = 0.1,
        reduction: int = 4,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.res_scale = res_scale
        body_layers = [
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
        ]
        if use_attention:
            body_layers.append(ChannelAttention(n_feats, reduction=reduction))

        self.body = nn.Sequential(*body_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x) * self.res_scale
        return x + res


@MODELS.register("our_nn")
class OurNN(nn.Module):
    """Modular Super-Resolution Network architecture."""

    trainable: bool = True

    def __init__(
        self,
        scale: int,
        n_feats: int = 32,
        n_blocks: int = 6,
        res_scale: float = 0.1,
        reduction: int = 4,
        bicubic_skip: bool = True,
        attention: bool = True,
        edge_branch: bool = True,
        zero_init_tail: bool = True,
    ) -> None:
        super().__init__()
        if scale not in (2, 3, 4):
            raise ValueError(f"OurNN supports scales 2, 3, 4. Got scale={scale}.")

        self.scale = scale
        self.bicubic_skip = bicubic_skip
        self.edge_branch_flag = edge_branch

        # Head
        self.head = nn.Conv2d(3, n_feats, kernel_size=3, padding=1)

        # Edge Branch
        if self.edge_branch_flag:
            sobel_x = torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
            ).view(1, 1, 3, 3)
            sobel_y = torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
            ).view(1, 1, 3, 3)
            laplacian = torch.tensor(
                [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
            ).view(1, 1, 3, 3)

            self.register_buffer("sobel_x", sobel_x)
            self.register_buffer("sobel_y", sobel_y)
            self.register_buffer("laplacian", laplacian)

            self.edge_conv = nn.Sequential(
                nn.Conv2d(3, n_feats, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.fuse_conv = nn.Conv2d(2 * n_feats, n_feats, kernel_size=1)

        # Body
        resblocks = [
            OurResBlock(
                n_feats,
                res_scale=res_scale,
                reduction=reduction,
                use_attention=attention,
            )
            for _ in range(n_blocks)
        ]
        self.body = nn.Sequential(*resblocks)
        self.body_conv = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)

        # Upsampler
        upsample_layers = []
        if scale in (2, 3):
            upsample_layers.extend(
                [
                    nn.Conv2d(n_feats, n_feats * (scale ** 2), kernel_size=3, padding=1),
                    nn.PixelShuffle(scale),
                ]
            )
        elif scale == 4:
            for _ in range(2):
                upsample_layers.extend(
                    [
                        nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1),
                        nn.PixelShuffle(2),
                    ]
                )

        self.upsample = nn.Sequential(*upsample_layers)

        # Tail
        self.tail = nn.Conv2d(n_feats, 3, kernel_size=3, padding=1)

        if zero_init_tail and bicubic_skip:
            nn.init.zeros_(self.tail.weight)
            if self.tail.bias is not None:
                nn.init.zeros_(self.tail.bias)

    def _extract_edge_features(self, lr: torch.Tensor) -> torch.Tensor:
        # Luminance: 0.299R + 0.587G + 0.114B
        gray = 0.299 * lr[:, 0:1, :, :] + 0.587 * lr[:, 1:2, :, :] + 0.114 * lr[:, 2:3, :, :]
        padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")

        resp_x = F.conv2d(padded, self.sobel_x)
        resp_y = F.conv2d(padded, self.sobel_y)
        resp_lap = F.conv2d(padded, self.laplacian)

        concat_resp = torch.cat([resp_x, resp_y, resp_lap], dim=1)  # (B, 3, H, W)
        return self.edge_conv(concat_resp)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        head_feat = self.head(lr)

        if self.edge_branch_flag:
            edge_feat = self._extract_edge_features(lr)
            fused_feat = torch.cat([head_feat, edge_feat], dim=1)
            body_in = self.fuse_conv(fused_feat)
        else:
            body_in = head_feat

        body_feat = self.body_conv(self.body(body_in))
        feat = body_in + body_feat
        up_feat = self.upsample(feat)
        tail_out = self.tail(up_feat)

        if self.bicubic_skip:
            x_up = F.interpolate(
                lr, scale_factor=self.scale, mode="bicubic", align_corners=False
            )
            return tail_out + x_up
        else:
            return tail_out
