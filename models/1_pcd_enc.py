import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn import GENConv, MLP, pool


class PCDEnc(nn.Module):
    def __init__(self, latent_dim, layers_points, layers_camera, layers_mlp, z_dim):
        super().__init__()
        self.pcd_proj = MLP(in_channels=7,hidden_channels=latent_dim,out_channels=latent_dim,
            num_layers=layers_mlp,act="relu",norm="layer")

        self.point_conv_init = GENConv(latent_dim * 2, latent_dim, edge_dim=1, norm="layer")

        self.point_convs = ModuleList(GENConv(latent_dim, latent_dim, edge_dim=1, norm="layer")
            for _ in range(layers_points))
        
        self.camera_convs = ModuleList(
            GENConv(latent_dim, latent_dim, edge_dim=4, norm="layer")
            for _ in range(layers_camera))
        
        self.latent_proj = MLP(in_channels=latent_dim,hidden_channels=latent_dim,
            out_channels=z_dim,num_layers=layers_mlp,act="relu",norm="layer")
        
        self.mu_head = nn.Linear(z_dim, z_dim)
        self.logvar_head = nn.Linear(z_dim, z_dim)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _normalize_batch(self, pcd, batch):
        if batch is None:
            return torch.zeros(pcd.size(0), device=pcd.device, dtype=torch.long)
        if not torch.is_tensor(batch):
            batch = torch.as_tensor(batch, device=pcd.device)
        batch = batch.to(device=pcd.device, dtype=torch.long).reshape(-1)
        if batch.numel() == 1:
            return batch.expand(pcd.size(0))
        if batch.numel() != pcd.size(0):
            raise ValueError(f"batch must have {pcd.size(0)} entries, got {batch.numel()}.")
        return batch

    def _origin_mask(self, pcd):
        return pcd[:, :4].abs().amax(dim=1) < 1e-6

    def _build_point_edges(self, pcd, batch):
        point_mask = ~self._origin_mask(pcd)
        edge_index_parts = []
        edge_attr_parts = []

        for graph_id in range(int(batch.max().item()) + 1):
            graph_nodes = torch.nonzero((batch == graph_id) & point_mask, as_tuple=False).flatten()
            num_points = graph_nodes.numel()
            if num_points <= 1:
                continue

            k = min(3, num_points - 1)
            dirs = F.normalize(pcd[graph_nodes, 1:4], dim=-1, eps=1e-8)
            dist = torch.cdist(dirs, dirs)
            dist.fill_diagonal_(float("inf"))
            nn_idx = torch.topk(dist, k=k, largest=False).indices

            src = graph_nodes.repeat_interleave(k)
            dst = graph_nodes[nn_idx.reshape(-1)]
            edge_index_parts.append(torch.stack([src, dst], dim=0))

            src_dirs = dirs.repeat_interleave(k, dim=0)
            dst_dirs = dirs[nn_idx.reshape(-1)]
            edge_attr_parts.append(torch.linalg.norm(dst_dirs - src_dirs, dim=-1, keepdim=True))

        if not edge_index_parts:
            return None, None
        return torch.cat(edge_index_parts, dim=1), torch.cat(edge_attr_parts, dim=0)

    def _build_camera_edges(self, pcd, batch):
        edge_index_parts = []
        edge_attr_parts = []
        origin_mask = self._origin_mask(pcd)

        for graph_id in range(int(batch.max().item()) + 1):
            graph_nodes = torch.nonzero(batch == graph_id, as_tuple=False).flatten()
            graph_origins = graph_nodes[origin_mask[graph_nodes]]
            if graph_origins.numel() != 1:
                raise ValueError(f"Expected one origin node in graph {graph_id}, found {graph_origins.numel()}.")

            origin = graph_origins[0]
            points = graph_nodes[~origin_mask[graph_nodes]]
            if points.numel() == 0:
                continue

            camera = torch.full((points.numel(),), origin, device=pcd.device, dtype=torch.long)
            edge_index = torch.cat(
                [torch.stack([camera, points], dim=0), torch.stack([points, camera], dim=0)],
                dim=1,
            )
            src, dst = edge_index
            edge_vec = pcd[dst, :3] - pcd[src, :3]
            edge_norm = torch.linalg.norm(edge_vec, dim=1, keepdim=True).clamp_min(1e-8)

            edge_index_parts.append(edge_index)
            edge_attr_parts.append(torch.cat([edge_norm, edge_vec / edge_norm], dim=1))

        if not edge_index_parts:
            return None, None
        return torch.cat(edge_index_parts, dim=1), torch.cat(edge_attr_parts, dim=0)

    def _pool_view_dirs(self, pcd, x, batch):
        pcd_parts = []
        x_parts = []
        batch_parts = []
        origin_mask = self._origin_mask(pcd)

        for graph_id in range(int(batch.max().item()) + 1):
            graph_nodes = torch.nonzero(batch == graph_id, as_tuple=False).flatten()
            graph_origin = graph_nodes[origin_mask[graph_nodes]]
            graph_points = graph_nodes[~origin_mask[graph_nodes]]
            if graph_origin.numel() != 1:
                raise ValueError(f"Expected one origin node in graph {graph_id}, found {graph_origin.numel()}.")
            if graph_points.numel() == 0:
                pcd_parts.append(pcd[graph_origin])
                x_parts.append(x[graph_origin])
                batch_parts.append(torch.full((1,), graph_id, device=batch.device, dtype=batch.dtype))
                continue

            dirs = F.normalize(pcd[graph_points, 1:4], dim=-1, eps=1e-8)
            u = (-dirs[:, 1] / dirs[:, 0].clamp_min(1e-8)).clamp(-0.9315964579582214, 0.9315964579582214)
            v = (dirs[:, 2] / dirs[:, 0].clamp_min(1e-8)).clamp(-0.6073485612869263, 0.6073485612869263)
            bx = torch.clamp((((u / 0.9315964579582214) * 0.5 + 0.5) * 8).long(), min=0, max=7)
            by = torch.clamp(((0.5 - (v / 0.6073485612869263) * 0.5) * 6).long(), min=0, max=5)
            bins, inv = torch.unique(by * 8 + bx, sorted=True, return_inverse=True)

            point_pcd = pcd[graph_points]
            point_x = x[graph_points]
            counts = torch.bincount(inv, minlength=bins.numel()).to(point_pcd.dtype).unsqueeze(-1)
            pooled_pcd = point_pcd.new_zeros((bins.numel(), point_pcd.size(1)))
            pooled_x = point_x.new_zeros((bins.numel(), point_x.size(1)))
            pooled_pcd.index_add_(0, inv, point_pcd)
            pooled_x.index_add_(0, inv, point_x)
            pooled_pcd = pooled_pcd / counts
            pooled_pcd = torch.cat(
                [pooled_pcd[:, :1], F.normalize(pooled_pcd[:, 1:4], dim=-1, eps=1e-8), pooled_pcd[:, 4:]],
                dim=-1,
            )
            pooled_x = pooled_x / counts

            pcd_parts.append(torch.cat([pcd[graph_origin], pooled_pcd], dim=0))
            x_parts.append(torch.cat([x[graph_origin], pooled_x], dim=0))
            batch_parts.append(torch.full((bins.numel() + 1,), graph_id, device=batch.device, dtype=batch.dtype))

        return torch.cat(pcd_parts, dim=0), torch.cat(x_parts, dim=0), torch.cat(batch_parts, dim=0)

    def forward(self, pcd, batch=None, ei_points=None, ew_points=None, ei_camera=None, ea_camera=None):
        if pcd.dim() == 1:
            pcd = pcd.unsqueeze(0)
        if pcd.dim() != 2 or pcd.size(-1) != 7:
            raise ValueError(f"pcd must have shape [N, 7], got {tuple(pcd.shape)}.")

        batch = self._normalize_batch(pcd, batch)
        origin_mask = self._origin_mask(pcd)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        if int(origin_mask.sum().item()) != num_graphs:
            raise ValueError(
                f"Expected exactly one origin node per graph, found {int(origin_mask.sum().item())} for {num_graphs} graph(s)."
            )

        if ei_points is None or ew_points is None or ei_points.numel() == 0:
            ei_points, ew_points = self._build_point_edges(pcd, batch)
        elif ew_points.dim() == 1:
            ew_points = ew_points.unsqueeze(-1)

        x = self.pcd_proj(pcd)

        if ei_points is not None and ew_points is not None:
            x_global = pool.global_mean_pool(x, batch)
            x = x + self.point_conv_init(torch.cat([x, x_global[batch]], dim=-1), ei_points, ew_points)
            for conv in self.point_convs:
                x = x + conv(x_global[batch], ei_points, ew_points)

        pcd, x, batch = self._pool_view_dirs(pcd, x, batch)
        origin_mask = self._origin_mask(pcd)
        ei_camera, ea_camera = self._build_camera_edges(pcd, batch)
        if ei_camera is not None and ea_camera is not None:
            for conv in self.camera_convs:
                x = x + conv(x, ei_camera, ea_camera)

        graph_features = pool.global_mean_pool(x[origin_mask], batch[origin_mask])
        hidden = self.latent_proj(graph_features)
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)
        return self.reparameterize(mu, logvar), mu, logvar
