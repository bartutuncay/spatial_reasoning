import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn import GENConv, MLP, knn_graph, pool


class PCDDecoder(nn.Module):
    def __init__(self, nodes_dim: int, layers_mlp: int, latent_dim: int, nodes_k: int, layers_points: int, layers_camera: int):
        super().__init__()
        self.nodes_k = nodes_k

        self.pcd_proj = MLP(in_channels=7,hidden_channels=latent_dim,
            out_channels=latent_dim,num_layers=layers_mlp,act="relu",norm="layer")
        
        self.depth_head = nn.Sequential(nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),nn.Linear(latent_dim, 1))
        
        self.point_convs = ModuleList(GENConv(latent_dim * 2, latent_dim, edge_dim=1, norm="layer")
            for _ in range(layers_points))
        
        self.camera_convs = ModuleList(GENConv(latent_dim, latent_dim, edge_dim=4, norm="layer")
            for _ in range(layers_camera))

    def nodes_to_xyz(self, nodes: torch.Tensor) -> torch.Tensor:
        return nodes[:, 1:4] * nodes[:, :1]

    def _empty_graph(self, device: torch.device, dtype: torch.dtype):
        return (
            torch.empty((2, 0), device=device, dtype=torch.long),
            torch.empty((0, 1), device=device, dtype=dtype),
        )

    def _edge_attr(self, xyz: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        edge_vec = xyz[dst] - xyz[src]
        edge_len = torch.linalg.norm(edge_vec, dim=1, keepdim=True).clamp_min(1e-8)
        return torch.cat([edge_len, edge_vec / edge_len], dim=1)

    def _image_hw(self, image: torch.Tensor):
        if image.shape[0] == 3 and image.shape[-1] != 3:
            return image.shape[-2], image.shape[-1]
        return image.shape[0], image.shape[1]

    def init_points(self, image: torch.Tensor, normalize_rgb: bool = True) -> torch.Tensor:
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = image.permute(1, 2, 0)

        h, w, _ = image.shape
        rgb = image.float()
        if image.dtype == torch.uint8 and normalize_rgb:
            rgb = rgb / 255.0

        ys, xs = torch.meshgrid(
            torch.arange(h, device=image.device, dtype=torch.float32),
            torch.arange(w, device=image.device, dtype=torch.float32),
            indexing="ij",
        )
        x_den = max(w - 1, 1)
        y_den = max(h - 1, 1)
        tanx = math.tan(0.5 * 1.4773256165109574)
        tany = math.tan(0.5 * 1.0903791454597398)
        xn = ((xs / x_den) * 2.0 - 1.0) * tanx
        yn = (1.0 - (ys / y_den) * 2.0) * tany
        dirs = torch.stack([torch.ones_like(xn), -xn, yn], dim=-1)
        dirs = F.normalize(dirs, dim=-1, eps=1e-8)
        depth = torch.ones((h * w, 1), device=image.device, dtype=rgb.dtype)
        return torch.cat([depth, dirs.reshape(-1, 3), rgb.reshape(-1, 3)], dim=-1)

    def make_camera_graph(self, points: torch.Tensor):
        num_points = points.size(0) - 1
        targets = torch.arange(1, num_points + 1, device=points.device, dtype=torch.long)
        sources = torch.zeros(num_points, device=points.device, dtype=torch.long)
        edge_index = torch.cat(
            [torch.stack([sources, targets], dim=0), torch.stack([targets, sources], dim=0)],
            dim=1,
        )
        return edge_index, self._edge_attr(self.nodes_to_xyz(points), edge_index)

    def build_point_graph(self, points: torch.Tensor, batch_points: torch.Tensor):
        if points.size(0) == 0:
            return self._empty_graph(points.device, points.dtype)

        dirs = F.normalize(points[:, 1:4], dim=-1, eps=1e-8)
        xyz = self.nodes_to_xyz(points)
        edge_index_parts = []
        edge_attr_parts = []

        num_graphs = int(batch_points.max().item()) + 1 if batch_points.numel() > 0 else 0
        for graph_id in range(num_graphs):
            graph_ids = torch.nonzero(batch_points == graph_id, as_tuple=False).flatten()
            if graph_ids.numel() <= 1:
                continue

            k = min(self.nodes_k, graph_ids.numel() - 1)
            local_edges = knn_graph(dirs[graph_ids], k=k, loop=False)
            edge_index = graph_ids[local_edges]
            src, dst = edge_index
            edge_attr = torch.linalg.norm(xyz[dst] - xyz[src], dim=1, keepdim=True)
            edge_index_parts.append(edge_index)
            edge_attr_parts.append(edge_attr)

        if not edge_index_parts:
            return self._empty_graph(points.device, points.dtype)
        return torch.cat(edge_index_parts, dim=1), torch.cat(edge_attr_parts, dim=0)

    def _assemble_graphs(self, images: torch.Tensor, device: torch.device, dtype: torch.dtype, depth_maps: Optional[torch.Tensor] = None):
        points_all = []
        batch_all = []
        point_batch_all = []
        camera_edges = []
        camera_attrs = []
        camera_idx = []
        offset = 0

        for graph_id in range(images.size(0)):
            points = self.init_points(images[graph_id]).to(device=device, dtype=dtype)
            if depth_maps is not None:
                points = points.clone()
                points[:, 0] = depth_maps[graph_id].reshape(-1)

            graph = torch.cat([points.new_zeros((1, points.size(1))), points], dim=0)
            edge_index, edge_attr = self.make_camera_graph(graph)
            points_all.append(graph)
            batch_all.append(torch.full((graph.size(0),), graph_id, device=device, dtype=torch.long))
            point_batch_all.append(torch.full((points.size(0),), graph_id, device=device, dtype=torch.long))
            camera_edges.append(edge_index + offset)
            camera_attrs.append(edge_attr)
            camera_idx.append(offset)
            offset += graph.size(0)

        return (
            torch.cat(points_all, dim=0),
            torch.cat(batch_all, dim=0),
            torch.cat(point_batch_all, dim=0),
            torch.tensor(camera_idx, device=device, dtype=torch.long),
            torch.cat(camera_edges, dim=1),
            torch.cat(camera_attrs, dim=0),
        )

    def _finalize_output(self, pcd_pred: torch.Tensor, batch: torch.Tensor, batch_points: torch.Tensor, camera_idx: torch.Tensor, ei_camera: torch.Tensor):
        point_mask = torch.ones(batch.size(0), device=batch.device, dtype=torch.bool)
        point_mask[camera_idx] = False
        point_ids = torch.nonzero(point_mask, as_tuple=False).flatten()
        ei_points, ew_points = self.build_point_graph(pcd_pred[point_mask], batch_points)
        return (
            pcd_pred,
            batch,
            point_ids[ei_points],
            ew_points.squeeze(-1),
            ei_camera,
            self._edge_attr(self.nodes_to_xyz(pcd_pred), ei_camera),
        )

    def forward(self, z: torch.Tensor, img: torch.Tensor, img_full: Optional[torch.Tensor] = None):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        if img_full is not None and img_full.dim() == 3:
            img_full = img_full.unsqueeze(0)
        if img_full is not None and img_full.size(0) != z.size(0):
            raise ValueError("img_full batch size must match z batch size.")

        device = z.device
        dtype = img.dtype if img.is_floating_point() else torch.float32
        x_all, batch, batch_points, camera_idx, ei_camera, _ = self._assemble_graphs(img, device, dtype)

        pcd_z = self.pcd_proj(x_all)
        point_mask = torch.ones(x_all.size(0), device=x_all.device, dtype=torch.bool)
        point_mask[camera_idx] = False
        h, w = self._image_hw(img[0])
        img_coarse = img if img.dim() == 4 and img.size(1) == 3 else img.permute(0, 3, 1, 2)
        img_coarse = F.adaptive_avg_pool2d(img_coarse, (6, 8))
        x_coarse, _, _, camera_idx_coarse, ei_camera_coarse, ea_camera_coarse = self._assemble_graphs(img_coarse, device, dtype)
        coarse_z = self.pcd_proj(x_coarse)
        coarse_z = coarse_z.index_copy(0, camera_idx_coarse, z)
        coarse_point_mask = torch.ones(x_coarse.size(0), device=device, dtype=torch.bool)
        coarse_point_mask[camera_idx_coarse] = False
        coarse_base = coarse_z[coarse_point_mask].clone()
        for conv in self.camera_convs:
            coarse_z = coarse_z + conv(coarse_z, ei_camera_coarse, edge_attr=ea_camera_coarse)
        delta = coarse_z[coarse_point_mask] - coarse_base
        delta = F.interpolate(delta.view(z.size(0), 6, 8, -1).permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False)
        point_z = pcd_z[point_mask] + delta.permute(0, 2, 3, 1).reshape(-1, delta.size(1))
        if point_z.size(0) == 0:
            pcd_depths = x_all.new_empty((0, 1))
        else:
            ei_points, ew_points = self.build_point_graph(x_all[point_mask], batch_points)
            if ei_points.numel() > 0:
                point_global = pool.global_mean_pool(point_z, batch_points)
                for conv in self.point_convs:
                    point_z = point_z + conv(
                        torch.cat([point_z, point_global[batch_points]], dim=-1),
                        ei_points,
                        edge_attr=ew_points,
                    )
            pcd_depths = x_all[point_mask, :1] * torch.exp(self.depth_head(point_z).clamp(-4.0, 4.0))

        pcd_pred = x_all.clone()
        pcd_pred[point_mask, :1] = pcd_depths
        if img_full is None:
            return self._finalize_output(pcd_pred, batch, batch_points, camera_idx, ei_camera)

        coarse_h, coarse_w = self._image_hw(img[0])
        full_h, full_w = self._image_hw(img_full[0])
        depth_maps = pcd_depths[:, 0].view(z.size(0), 1, coarse_h, coarse_w)
        depth_maps = F.interpolate(depth_maps, size=(full_h, full_w), mode="bilinear", align_corners=False)
        pcd_full, batch_full, batch_points_full, camera_idx_full, ei_camera_full, _ = self._assemble_graphs(
            img_full,
            device,
            dtype,
            depth_maps[:, 0],
        )
        return self._finalize_output(pcd_full, batch_full, batch_points_full, camera_idx_full, ei_camera_full)
