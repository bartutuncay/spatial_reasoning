import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class UpBlock(nn.Module):
    """
    Upsample by 2 then apply a ConvBlock-like residual refinement.
    """
    def __init__(self, in_ch: int, out_ch: int, norm: str = "group", groups: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)

        if norm == "batch":
            self.n1 = nn.BatchNorm2d(out_ch)
            self.n2 = nn.BatchNorm2d(out_ch)
        elif norm == "group":
            g = min(groups, out_ch)
            self.n1 = nn.GroupNorm(g, out_ch)
            self.n2 = nn.GroupNorm(g, out_ch)
        else:
            raise ValueError("norm must be 'group' or 'batch'.")

        self.act = nn.SiLU(inplace=True)

        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsample spatially
        x_up = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)

        identity = x_up
        if self.proj is not None:
            identity = self.proj(identity)

        h = self.conv1(x_up)
        h = self.n1(h)
        h = self.act(h)

        h = self.conv2(h)
        h = self.n2(h)

        return self.act(h + identity)


class ImageDecoder(nn.Module):
    """
    Decode latent z -> RGB image (B,3,H,W).

    You must know the target output resolution (H,W) at init time, or pass it and
    rebuild layers (less convenient). This version fixes (H,W) in __init__.
    """
    def __init__(
        self,
        latent_dim: int = 128,
        base_dim: int = 32,
        out_hw: Tuple[int, int] = (384, 512),
        norm: str = "group",
        groups: int = 16,
        out_act: str = "tanh",  # "tanh", "sigmoid", or "none"
    ):
        super().__init__()
        H, W = out_hw
        if H % 16 != 0 or W % 16 != 0:
            raise ValueError("out_hw must be divisible by 16 to mirror the encoder downsampling.")

        self.H = H
        self.W = W
        self.out_act = out_act

        # Mirror encoder channel ladder:
        # encoder: 32 → 64 → 128 → 256 at /16
        # decoder starts at /16 with C4 then goes 256→128→64→32 and finally 3
        C1 = base_dim
        C2 = base_dim * 2
        C3 = base_dim * 4
        C4 = base_dim * 8  # typically 256 if base_dim=32

        Hf, Wf = H // 16, W // 16
        self.Hf, self.Wf = Hf, Wf
        self.C4 = C4

        # z -> initial feature grid
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, C4 * Hf * Wf),
            nn.SiLU(inplace=True),
        )

        # refinement at /16 (optional but helps)
        self.refine = nn.Sequential(
            nn.Conv2d(C4, C4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(groups, C4), C4) if norm == "group" else nn.BatchNorm2d(C4),
            nn.SiLU(inplace=True),
        )

        # Upsampling stages: /16→/8→/4→/2→/1
        self.up1 = UpBlock(C4, C3, norm=norm, groups=groups)  # 256→128
        self.up2 = UpBlock(C3, C2, norm=norm, groups=groups)  # 128→64
        self.up3 = UpBlock(C2, C1, norm=norm, groups=groups)  # 64→32
        self.up4 = UpBlock(C1, C1, norm=norm, groups=groups)  # 32→32

        # final RGB head
        self.to_rgb = nn.Conv2d(C1, 3, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, latent_dim)
        returns: (B, 3, H, W)
        """
        B = z.shape[0]
        x = self.fc(z).view(B, self.C4, self.Hf, self.Wf)  # (B,C4,H/16,W/16)
        x = self.refine(x)

        x = self.up1(x)  # /8
        x = self.up2(x)  # /4
        x = self.up3(x)  # /2
        x = self.up4(x)  # /1

        x = self.to_rgb(x)

        if self.out_act == "tanh":
            return torch.tanh(x)          # if you normalize images to [-1, 1]
        if self.out_act == "sigmoid":
            return torch.sigmoid(x)       # if you normalize images to [0, 1]
        if self.out_act == "none":
            return x
        raise ValueError("out_act must be 'tanh', 'sigmoid', or 'none'.")
