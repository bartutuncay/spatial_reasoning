import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

def convert_rotmat_quaternion(R:torch.Tensor) -> torch.Tensor:
    # R is a tensor of quaternion (qw,qx,qy,qz): W=0,X=1,Y=2,Z=3
    # quaternion: qw,qx,qy,qz: qx,qy,qz = rotation axis unit vector*sin(angle/2); qw = cos(angle/2)
    #    |       2     2                                |
    #    | 1 - 2Y  - 2Z    2XY - 2ZW      2XZ + 2YW     |
    #    |                                              |
    #    |                       2     2                |
    #M = | 2XY + 2ZW       1 - 2X  - 2Z   2YZ - 2XW     |
    #    |                                              |
    #    |                                      2     2 |
    #    | 2XZ - 2YW       2YZ + 2XW      1 - 2X  - 2Y  |
    #    |                                              |
    M = torch.empty(3,3,dtype=R.dtype)
    M[0,0] = 1-2*R[2]**2-2*R[3]**2
    M[0,1] = 2*R[2]*R[1]-2*R[3]*R[0]
    M[0,2] = 2*R[2]*R[1]-2*R[3]*R[0]
    M[1,0] = 2*R[2]*R[1]+2*R[3]*R[0]
    M[1,1] = 1-2*R[1]**2-2*R[3]**2
    M[1,2] = 2*R[2]*R[3]-2*R[1]*R[0]
    M[2,0] = 2*R[1]*R[3]-2*R[2]*R[0]
    M[2,1] = 2*R[2]*R[3]+2*R[1]*R[0]
    M[2,2] = 1-2*R[1]**2-2*R[2]**2
    return M

def rotmat_to_6d(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to a continuous 6D representation (Zhou et al.).
    R: (..., 3, 3) -> (..., 6) using first two columns.
    """
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)

def safe_intrinsics_vec() -> torch.Tensor:
    #(K: torch.Tensor, img_hw: Tuple[int, int])
    """
    Create a compact, normalized intrinsics vector from K.
    K: (B, 3, 3) (or (3,3) broadcastable) with fx, fy, cx, cy in pixels.
    Returns: (B, 4) = [fx/W, fy/H, cx/W, cy/H]
    """
    #H, W = img_hw
    #fx = K[..., 0, 0]
    #fy = K[..., 1, 1]
    #cx = K[..., 0, 2]
    #cy = K[..., 1, 2]
    #v = torch.stack([fx / W, fy / H, cx / W, cy / H], dim=-1)

    #camera_intrinsics = {'W':6208,'H':4135,'fx':3408.59,'fy':3408.87,'cx':3117.24,'cy':2064.07}
    v = torch.Tensor([3408.59 / 6208, 3408.87 / 4135, 3117.24 / 6208, 2064.07 / 4135])
    return v

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

class FiLM(nn.Module):
    """
    FiLM modulation: h -> gamma * h + beta, with (gamma,beta) from conditioning vector.
    """
    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.to_gb = nn.Linear(cond_dim, 2 * feat_dim)
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        h: (B, C, H, W) or (B, T, C)
        cond: (B, cond_dim)
        """
        gb = self.to_gb(cond)  # (B, 2C)
        gamma, beta = gb.chunk(2, dim=-1)  # (B, C), (B, C)

        if h.dim() == 4:
            gamma = gamma[:, :, None, None]
            beta  = beta[:, :, None, None]
        elif h.dim() == 3:
            gamma = gamma[:, None, :]
            beta  = beta[:, None, :]
        else:
            raise ValueError("Unsupported tensor rank for FiLM.")

        return h * (1.0 + gamma) + beta

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
        cond_dim: Optional[int] = None,
        norm: str = "group",
        groups: int = 16,
    ):
        super().__init__()
        self.stride = stride
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.cond_dim = cond_dim

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

        self.film1 = FiLM(cond_dim, out_ch) if cond_dim is not None else None
        self.film2 = FiLM(cond_dim, out_ch) if cond_dim is not None else None

        # projection for residual if shape changes
        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        identity = x

        h = self.conv1(x)
        h = self.n1(h)
        if self.film1 is not None:
            if cond is None:
                raise ValueError("cond must be provided when FiLM is enabled.")
            h = self.film1(h, cond)
        h = self.act(h)

        h = self.conv2(h)
        h = self.n2(h)
        if self.film2 is not None:
            if cond is None:
                raise ValueError("cond must be provided when FiLM is enabled.")
            h = self.film2(h, cond)

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
        attn_logits = (q * self.k(tokens)).sum(dim=-1) / math.sqrt(C)  # (B,T)
        attn = attn_logits.softmax(dim=-1)  # (B,T)
        pooled = (attn[:, :, None] * self.v(tokens)).sum(dim=1)  # (B,C)
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
        pose_dim: int = 64,
        use_film: bool = True,
        norm: str = "group",
        groups: int = 16,
        add_posenc: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.token_dim = token_dim
        self.pose_dim = pose_dim
        self.use_film = use_film
        self.add_posenc = add_posenc

        # Pose/intrinsics/crop embedding
        # pose: rot6d (6) + t (3) + intrinsics (4) + optional crop (4) = 13..17 dims
        in_cond = 6 + 3 + 4
        self.use_crop = True
        in_cond_crop = 4  # (x0/W, y0/H, w/W, h/H)
        self.cond_in_dim = in_cond + in_cond_crop

        self.cond_mlp = nn.Sequential(
            nn.Linear(self.cond_in_dim, pose_dim),
            nn.SiLU(inplace=True),
            nn.Linear(pose_dim, pose_dim),
            nn.SiLU(inplace=True),
        )

        cond_dim = pose_dim if use_film else None

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

        self.b1 = ConvBlock(ch1, ch2, stride=2, cond_dim=cond_dim, norm=norm, groups=groups)  # /4
        self.b2 = ConvBlock(ch2, ch3, stride=2, cond_dim=cond_dim, norm=norm, groups=groups)  # /8
        self.b3 = ConvBlock(ch3, ch4, stride=2, cond_dim=cond_dim, norm=norm, groups=groups)  # /16

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
            nn.LayerNorm(token_dim + pose_dim),  # include cond embedding
            nn.Linear(token_dim + pose_dim, latent_dim),
        )

    def _encode_condition(
        self,
        R: torch.Tensor,
        t: torch.Tensor,
        #K: torch.Tensor,
        img_hw: Tuple[int, int],
        crop: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B = R.shape[0]
        rot6d = rotmat_to_6d(R)  # (B,6)
        intr4 = safe_intrinsics_vec()#.unsqueeze(0) #(K, img_hw) # (B,4)

        # Crop embedding: normalized (x0/W, y0/H, w/W, h/H)
        # If you do not crop, pass zeros.
        if crop is None:
            crop4 = torch.zeros((B, 4), device=R.device, dtype=R.dtype)
        else:
            # Expect tensors broadcastable to (B,)
            H, W = img_hw
            x0 = crop["x0"].to(device=R.device, dtype=R.dtype)
            y0 = crop["y0"].to(device=R.device, dtype=R.dtype)
            w  = crop["w"].to(device=R.device, dtype=R.dtype)
            h  = crop["h"].to(device=R.device, dtype=R.dtype)
            crop4 = torch.stack([x0 / W, y0 / H, w / W, h / H], dim=-1)
        print(rot6d.shape,t.shape,intr4.shape,crop4.shape)
        cond_in = torch.cat([rot6d, t, intr4, crop4], dim=-1)  # (B, cond_in_dim)
        cond = self.cond_mlp(cond_in)  # (B, pose_dim)
        return cond

    def forward(
        self,
        img: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        #K: torch.Tensor,
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
        cond = self._encode_condition(R, t, (H, W), crop=crop)  # (B, pose_dim)
        #cond = self._encode_condition(R, t, K, (H, W), crop=crop)  # (B, pose_dim)

        x = self.stem(img)  # /2

        if self.use_film:
            x = self.b1(x, cond=cond)  # /4
            x = self.b2(x, cond=cond)  # /8
            x = self.b3(x, cond=cond)  # /16
        else:
            x = self.b1(x, cond=None)
            x = self.b2(x, cond=None)
            x = self.b3(x, cond=None)

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
        z = self.out(torch.cat([pooled, cond], dim=-1))  # (B, latent_dim)

        if not return_extras:
            return z

        extras = {
            "cond": cond,          # (B, pose_dim)
            "feat_grid": feat,     # (B, C, Hf, Wf)
            "tokens": tokens,      # (B, T, C)
            "pooled": pooled,      # (B, C)
        }
        return z, extras

# Example usage
if False:
    B = 2
    img = torch.randn(B, 3, 384, 512)
    R = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
    t = torch.zeros(B, 3)
    K = torch.tensor([[[800., 0., 256.],
                       [0., 800., 192.],
                       [0.,   0.,   1.]]]).repeat(B, 1, 1)

    enc = ImageEncoder(latent_dim=128, base_dim=32, token_dim=256, pose_dim=64, use_film=True)
    z, extras = enc(img, R, t, K, return_extras=True)
    print(z.shape, extras["tokens"].shape)
