import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

## Updated image encoder pipeline:
# check if ResNet-based
# otherwise unchanged

class SinCos2DPositionalEncoding(nn.Module):
    """
    Fixed 2D sinusoidal positional encoding for a (Hf, Wf) token grid.
    Adds a (C) vector to each token.

    Note: This is lightweight and avoids learning a giant embedding table.
    """
    def __init__(self, dim: int, temperature: float = 10000.0):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("positional encoding dim must be divisible by 4")
        self.dim = dim
        self.temperature = temperature

    def forward(self, Hf: int, Wf: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Returns: (Hf*Wf, dim) positional encoding.
        """
        y = torch.linspace(0, 1, steps=Hf, device=device, dtype=dtype)
        x = torch.linspace(0, 1, steps=Wf, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")  # (Hf, Wf)
        yy = yy.reshape(-1)  # (T,)
        xx = xx.reshape(-1)  # (T,)

        dim_quarter = self.dim // 4
        omega = torch.arange(dim_quarter, device=device, dtype=dtype)
        omega = 1.0 / (self.temperature ** (omega / dim_quarter))  # (dim_quarter,)

        # (T, dim_quarter)
        out_y = yy[:, None] * omega[None, :]
        out_x = xx[:, None] * omega[None, :]

        pe = torch.cat([torch.sin(out_x), torch.cos(out_x), torch.sin(out_y), torch.cos(out_y)], dim=-1)
        return pe  # (T, dim)

# Building blocks

class ConvBlock(nn.Module):
    """
    Lightweight residual-ish block:
    - Conv(3x3), norm, act
    - Conv(3x3), norm, act
    - optional downsample via stride in first conv
    - optional FiLM after each norm
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        norm: str = "group",
        groups: int = 16,
    ):
        super().__init__()
        self.stride = stride
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        if norm == "batch":
            self.n1 = nn.BatchNorm2d(out_ch)
            self.n2 = nn.BatchNorm2d(out_ch)
        elif norm == "group":
            g1 = min(groups, out_ch)
            self.n1 = nn.GroupNorm(num_groups=g1, num_channels=out_ch)
            self.n2 = nn.GroupNorm(num_groups=g1, num_channels=out_ch)
        else:
            raise ValueError("norm must be 'group' or 'batch'.")

        self.act = nn.SiLU(inplace=True)

        # projection for residual if shape changes
        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        identity = x

        h = self.conv1(x)
        h = self.n1(h)

        h = self.act(h)

        h = self.conv2(h)
        h = self.n2(h)

        if self.proj is not None:
            identity = self.proj(identity)

        h = self.act(h + identity)
        return h

class AttentionPool(nn.Module):
    """
    Attention pooling over tokens:
    tokens: (B, T, C) -> pooled: (B, C)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.q, std=0.02)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T, C = tokens.shape
        q = self.q[None, None, :].expand(B, 1, C)  # (B,1,C)
        #attn_logits = (q * self.k(tokens)).sum(dim=-1) / math.sqrt(C)  # (B,T)
        #attn = attn_logits.softmax(dim=-1)  # (B,T)
        #pooled = (attn[:, :, None] * self.v(tokens)).sum(dim=1)  # (B,C)

        q = tokens.mean(dim=1, keepdim=True)  # (B,1,C)
        attn_logits = (q * self.k(tokens)).sum(dim=-1) / math.sqrt(C)
        pooled = tokens.mean(dim=1)
        return pooled

# ImageEncoder

class ImageEncoder(nn.Module):
    """
    Lightweight CNN token encoder + attention pooling into a shared latent z.

    Inputs
    ------
    img: (B, 3, H, W) float
    R:   (B, 3, 3) camera rotation (recommend: camera-to-world or world-to-camera, but be consistent)
    t:   (B, 3)    camera translation (same convention as R)
    K:   (B, 3, 3) intrinsics matrix

    Optional
    --------
    crop: dict with keys such as:
        - "x0", "y0", "w", "h" in pixel units of the original image, if you crop before resizing
      These can be embedded to help the model disambiguate cropping. If unused, pass None.

    Output
    ------
    z: (B, latent_dim)
    extras: dict with intermediate tensors (optional; can be removed if you prefer)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        base_dim: int = 32,
        token_dim: int = 256,
        norm: str = "group",
        groups: int = 16,
        add_posenc: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.token_dim = token_dim
        self.add_posenc = add_posenc

        in_cond = 6 + 3 + 4
        in_cond_crop = 4  # (x0/W, y0/H, w/W, h/H)
        self.cond_in_dim = in_cond + in_cond_crop

        # CNN stem: 4 stages, downsample x16
        ch1 = base_dim
        ch2 = base_dim * 2
        ch3 = base_dim * 4
        ch4 = base_dim * 8  # typically 256 if base_dim=32

        self.stem = nn.Sequential(
            nn.Conv2d(3, ch1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(num_groups=min(groups, ch1), num_channels=ch1) if norm == "group" else nn.BatchNorm2d(ch1),
            nn.SiLU(inplace=True),
        )

        self.b1 = ConvBlock(ch1, ch2, stride=2, norm=norm, groups=groups)  # /4
        self.b2 = ConvBlock(ch2, ch3, stride=2, norm=norm, groups=groups)  # /8
        self.b3 = ConvBlock(ch3, ch4, stride=2, norm=norm, groups=groups)  # /16

        # Project grid features to token_dim (keeps token size constant)
        self.to_tokens = nn.Conv2d(ch4, token_dim, kernel_size=1, stride=1, padding=0, bias=False)

        # Optional small "process" MLP on tokens (very light)
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(token_dim * 2, token_dim),
        )

        self.posenc = SinCos2DPositionalEncoding(token_dim) if add_posenc else None

        # Pooling and final projection to shared latent
        self.pool = AttentionPool(token_dim)
        self.out = nn.Sequential(
            nn.LayerNorm(token_dim),  # include cond embedding
            nn.Linear(token_dim, latent_dim),
        )

    def forward(
        self,
        img: torch.Tensor,
        crop: Optional[Dict[str, torch.Tensor]] = None,
        return_extras: bool = False,
    ):
        """
        img: (B,3,H,W)
        R:   (B,3,3)
        t:   (B,3)
        K:   (B,3,3)
        """
        if img.dim() != 4 or img.size(1) != 3:
            raise ValueError("img must be (B,3,H,W).")

        B, _, H, W = img.shape

        x = self.stem(img)  # /2

        x = self.b1(x) # /4
        x = self.b2(x) # /8
        x = self.b3(x) # /16

        # Grid -> tokens
        feat = self.to_tokens(x)  # (B, token_dim, Hf, Wf)
        _, C, Hf, Wf = feat.shape
        tokens = feat.flatten(2).transpose(1, 2).contiguous()  # (B, T=Hf*Wf, C)

        # Add fixed 2D positional encoding
        if self.posenc is not None:
            pe = self.posenc(Hf, Wf, device=tokens.device, dtype=tokens.dtype)  # (T,C)
            tokens = tokens + pe[None, :, :]

        # Very light token processing
        tokens = tokens + self.token_mlp(tokens)

        pooled = self.pool(tokens)  # (B, C)
        z = self.out(pooled)  # (B, latent_dim)

        extras = {
            "feat_grid": feat,     # (B, C, Hf, Wf)
            "tokens": tokens,      # (B, T, C)
            "pooled": pooled,      # (B, C)
        }

        return z