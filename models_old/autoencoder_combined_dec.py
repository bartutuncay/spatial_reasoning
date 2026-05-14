import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from torch_geometric.data import Data
import torch.nn.functional as F
import torch_geometric.nn.functional as F_geom
from torch_geometric.nn import GENConv, MLP, Linear, pool
from torch.nn import ModuleList
import math

## Combined Decoder Pipeline:
# start with single latent
# reconstruct image
# initialize image pixels as pcd points
# compare image losses
# concatenate latent space with GENConv from camera
# run final GNN pass on k-NN edges
# compare pcd losses

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


# Data format:
# Input:
# latent    [latent_dim]
# Output:
# pcd       [x,y,z,R,G,B] (relative positions)
# edges     [dist]

class PCDDecoder(nn.Module):
    def __init__(self,nodes_dim:int,latent_dim:int,nodes_k:int,layers_points:int,layers_camera:int):
        super().__init__()
        self.latent_dim = latent_dim
        self.nodes_k = nodes_k # number of neighbors for each node
        self.layers_points = layers_points
        self.layers_camera = layers_camera
        self.nodes_dim = nodes_dim

        # node generator latent --> N nodes
        self.node_mlp = nn.Sequential( 
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, self.nodes_init * self.nodes_dim))

        # Edge scorer: (xi, xj, |xi-xj|, xi*xj) -> logit
        # N nodes --> latent
        in_edge = 4 * nodes_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_edge, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1))

        self.conv_layers_points = ModuleList([
            GENConv(in_channels=latent_dim*2,out_channels=latent_dim)
            for _ in range(layers_points)])

        self.conv_layers_camera = ModuleList([
            GENConv(in_channels=latent_dim,out_channels=latent_dim)
            for _ in range(layers_camera)])
    
    def init_points(
        image: torch.Tensor,
        fov_deg: float,
        fov_axis: str = "horizontal",
        normalize_rgb: bool = True) -> torch.Tensor:
        if image.ndim != 3:
            raise ValueError(f"Expected image with 3 dims, got shape {tuple(image.shape)}")

        # Convert CHW -> HWC if needed
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = image.permute(1, 2, 0)

        H, W, _ = image.shape
        device = image.device

        if image.dtype == torch.uint8:
            rgb = image.float()
            if normalize_rgb:
                rgb = rgb / 255.0
        else:
            rgb = image.float()

        # Focal length from FOV
        fov_rad = math.radians(fov_deg)
        if fov_axis == "horizontal":
            fx = (W / 2.0) / math.tan(fov_rad / 2.0)
            fy = fx
        elif fov_axis == "vertical":
            fy = (H / 2.0) / math.tan(fov_rad / 2.0)
            fx = fy
        else:
            raise ValueError("fov_axis must be 'horizontal' or 'vertical'")

        # Principal point at image center
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0

        # Pixel grid using pixel centers
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )

        # Camera-space ray directions:
        # x right, y down, z forward
        x = (xs - cx) / fx
        y = (ys - cy) / fy
        z = torch.ones_like(x)

        dirs = torch.stack([x, y, z], dim=-1)  # [H, W, 3]
        dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True).clamp_min(1e-8)
        dists = torch.ones(dirs.shape(0))

        # Flatten to [H*W, 3]
        coords = dirs.reshape(-1, 3)
        colors = rgb.reshape(-1, 3)

        return torch.cat([dists, coords, colors], dim=-1)

    def make_graph(points):
        origin_sources = torch.full(points.shape[0], 0)
        targets = torch.arange(1, points.shape[0]+1)
        origin_to_nodes = torch.stack([origin_sources, targets], axis=0)
        nodes_to_origin = torch.stack([targets, origin_sources], axis=0)
        ei_camera = torch.cat([origin_to_nodes, nodes_to_origin], axis=1)
        src, dst = ei_camera
        edge_dists = points[dst]-points[src]
        edge_norm = torch.linalg.norm(edge_dists,dim=1)
        ea_camera = torch.cat([edge_norm,edge_dists/edge_norm],dim=1)
        return ei_camera, ea_camera

    def forward(self, z: torch.Tensor, img: torch.Tensor):
        # get points from image
        pcd_pts = self.init_points(img)
        pcd_pts = pcd_pts.vstack([torch.zeros((1,7)),pcd_pts],axis=0)
        # only visible nodes
        pcd_mask = pcd_pts[1:,4:] > torch.ones(1,3)*0.01 
        pcd_pts = pcd_pts[pcd_mask]
        # build edges - initialize with (1,u,v,w)
        # knn graph between located points
        ei_camera, ea_camera = self.make_graph(pcd_pts)
        pcd_z = self.node_mlp(pcd_pts)

        for conv in self.conv_layers_camera:
            m = conv(pcd_z,ei_camera,edge_attr=ea_camera)
            pcd_z = pcd_z+m
        
            
        
        x_expanded = pool.global_max_pool #pool the strongest signals in fine pass
        


        # losses guided by chamfer loss + encoded graph latent

        

        return graphs if B > 1 else graphs[0]