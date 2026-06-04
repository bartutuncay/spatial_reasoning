from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import GENConv, MLP, knn_graph
from torch_geometric.utils import softmax, to_undirected


def _ensure_batch(batch: Optional[torch.Tensor], num_nodes: int, device: torch.device) -> torch.Tensor:
    if batch is None:
        return torch.zeros(num_nodes, device=device, dtype=torch.long)
    return batch.reshape(-1).to(device=device, dtype=torch.long)


def _scatter_sum(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = values.new_zeros((dim_size, values.size(-1)))
    out.index_add_(0, index, values)
    return out


def _scatter_mean(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    summed = _scatter_sum(values, index, dim_size)
    counts = values.new_zeros((dim_size, 1))
    counts.index_add_(0, index, values.new_ones((index.numel(), 1)))
    mean = summed / counts.clamp_min(1.0)
    return mean, counts


def _global_mean(values: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
    mean, _ = _scatter_mean(values, batch, num_graphs)
    return mean


class ConvBlock(nn.Module):
    # scale invariant coarsening conv block
    def __init__(self, in_dims: int, out_dims: int, stride: int = 4, outlier_ratio: float = 0.15, coarse_k: int = 8):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.stride = max(1, int(stride))
        self.outlier_ratio = float(max(0.0, min(0.5, outlier_ratio)))
        self.coarse_k = max(1, int(coarse_k))

        self.mu_head = nn.Linear(in_dims, in_dims)
        self.logvar_head = nn.Linear(in_dims, in_dims)

        score_dims = (3 * in_dims) + 3
        self.pool = nn.Sequential(
            nn.Linear(score_dims, in_dims),
            nn.LayerNorm(in_dims),
            nn.GELU(),
            nn.Linear(in_dims, 1),
        )
        self.project = nn.Sequential(
            nn.Linear(in_dims, out_dims),
            nn.LayerNorm(out_dims),
            nn.GELU(),
        )
        self.process = GENConv(out_dims * 2, out_dims, norm="layer", msg_norm=True, edge_dim=3)

    def _neighbor_stats(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        num_nodes = mu.size(0)
        src, dst = edge_index

        neigh_mu_mean, counts = _scatter_mean(mu[src], dst, num_nodes)
        neigh_mu_sq_mean, _ = _scatter_mean(mu[src] * mu[src], dst, num_nodes)
        neigh_logvar_mean, _ = _scatter_mean(logvar[src], dst, num_nodes)

        neigh_var = (neigh_mu_sq_mean - neigh_mu_mean.pow(2)).clamp_min(1e-6)
        neigh_std = neigh_var.sqrt()

        agreement_edges = F.cosine_similarity(mu[src], mu[dst], dim=-1).unsqueeze(-1)
        agreement, _ = _scatter_mean(agreement_edges, dst, num_nodes)

        uncertainty = 0.5 * (logvar.mean(dim=-1, keepdim=True) + neigh_logvar_mean.mean(dim=-1, keepdim=True))
        confidence = torch.exp(-uncertainty.clamp(min=-6.0, max=6.0))

        novelty = (((mu - neigh_mu_mean).pow(2)) / (neigh_var + 1e-3)).mean(dim=-1, keepdim=True).sqrt()
        novelty = torch.tanh(novelty)

        isolated = counts.eq(0)
        neigh_mu_mean = torch.where(isolated, mu, neigh_mu_mean)
        neigh_std = torch.where(isolated, torch.zeros_like(neigh_std), neigh_std)
        agreement = torch.where(isolated, torch.ones_like(agreement), agreement)
        novelty = torch.where(isolated, torch.ones_like(novelty), novelty)

        return {
            "neighbor_mu_mean": neigh_mu_mean,
            "neighbor_mu_std": neigh_std,
            "agreement": agreement,
            "confidence": confidence,
            "novelty": novelty,
        }

    def _select_nodes(
        self,
        core_score: torch.Tensor,
        outlier_score: torch.Tensor,
        batch: torch.Tensor,
        stride_override: Optional[int] = None,
    ) -> torch.Tensor:
        stride = max(1, int(self.stride if stride_override is None else stride_override))
        keep_parts = []
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        for graph_id in range(num_graphs):
            graph_ids = torch.nonzero(batch == graph_id, as_tuple=False).squeeze(-1)
            num_nodes = graph_ids.numel()
            if num_nodes <= 1:
                keep_parts.append(graph_ids)
                continue

            num_keep = min(num_nodes, max(1, int(math.ceil(num_nodes / stride))))
            num_outliers = min(num_keep - 1, int(round(num_keep * self.outlier_ratio)))
            num_core = max(1, num_keep - num_outliers)

            local_core_score = core_score[graph_ids]
            core_local = torch.topk(local_core_score, k=num_core, largest=True).indices
            selected = graph_ids[core_local]

            if num_outliers > 0 and num_core < num_nodes:
                remaining_mask = torch.ones(num_nodes, device=batch.device, dtype=torch.bool)
                remaining_mask[core_local] = False
                remaining_ids = graph_ids[remaining_mask]
                remaining_scores = outlier_score[remaining_ids]
                outlier_local = torch.topk(
                    remaining_scores,
                    k=min(num_outliers, remaining_ids.numel()),
                    largest=True,
                ).indices
                selected = torch.cat([selected, remaining_ids[outlier_local]], dim=0)

            keep_parts.append(selected.unique(sorted=True))

        return torch.cat(keep_parts, dim=0).unique(sorted=True)

    def _build_coarse_graph(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pos.size(0) <= 1:
            empty_index = torch.empty((2, 0), device=pos.device, dtype=torch.long)
            empty_attr = torch.empty((0, 3), device=pos.device, dtype=pos.dtype)
            return empty_index, empty_attr

        edge_parts = []
        attr_parts = []
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        for graph_id in range(num_graphs):
            graph_ids = torch.nonzero(batch == graph_id, as_tuple=False).squeeze(-1)
            if graph_ids.numel() <= 1:
                continue

            k = min(self.coarse_k, graph_ids.numel() - 1)
            local_pos = pos[graph_ids]
            local_edge_index = knn_graph(local_pos, k=k, loop=False)
            local_edge_index = graph_ids[local_edge_index]
            local_edge_index = to_undirected(local_edge_index, num_nodes=pos.size(0))

            src, dst = local_edge_index
            local_edge_attr = pos[src] - pos[dst]
            edge_parts.append(local_edge_index)
            attr_parts.append(local_edge_attr)

        if not edge_parts:
            empty_index = torch.empty((2, 0), device=pos.device, dtype=torch.long)
            empty_attr = torch.empty((0, 3), device=pos.device, dtype=pos.dtype)
            return empty_index, empty_attr

        return torch.cat(edge_parts, dim=1), torch.cat(attr_parts, dim=0)

    def _aggregate(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        keep_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        num_nodes = x.size(0)
        keep_mask = torch.zeros(num_nodes, device=x.device, dtype=torch.bool)
        keep_mask[keep_idx] = True

        src, dst = edge_index
        edge_similarity = 0.5 * (F.cosine_similarity(mu[src], mu[dst], dim=-1) + 1.0)
        edge_confidence = torch.exp(
            -0.25 * (logvar[src].mean(dim=-1) + logvar[dst].mean(dim=-1)).clamp(min=-6.0, max=6.0)
        )
        edge_logits = edge_similarity * edge_confidence

        candidate_mask = (~keep_mask[src]) & keep_mask[dst]
        candidate_src = src[candidate_mask]
        candidate_dst = dst[candidate_mask]
        candidate_logits = edge_logits[candidate_mask]

        covered = torch.zeros(num_nodes, device=x.device, dtype=torch.bool)
        if candidate_src.numel() > 0:
            covered[candidate_src.unique()] = True
        extra_keep = torch.nonzero((~keep_mask) & (~covered), as_tuple=False).squeeze(-1)
        if extra_keep.numel() > 0:
            keep_idx = torch.cat([keep_idx, extra_keep], dim=0).unique(sorted=True)
            keep_mask = torch.zeros(num_nodes, device=x.device, dtype=torch.bool)
            keep_mask[keep_idx] = True

            candidate_mask = (~keep_mask[src]) & keep_mask[dst]
            candidate_src = src[candidate_mask]
            candidate_dst = dst[candidate_mask]
            candidate_logits = edge_logits[candidate_mask]

        anchor_map = torch.full((num_nodes,), -1, device=x.device, dtype=torch.long)
        anchor_map[keep_idx] = torch.arange(keep_idx.numel(), device=x.device, dtype=torch.long)

        if candidate_src.numel() > 0:
            candidate_weights = softmax(candidate_logits, candidate_src, num_nodes=num_nodes).unsqueeze(-1)
            assign_src = torch.cat([keep_idx, candidate_src], dim=0)
            assign_anchor = torch.cat([keep_idx, candidate_dst], dim=0)
            self_weights = torch.exp(-logvar[keep_idx].mean(dim=-1, keepdim=True).clamp(min=-6.0, max=6.0))
            assign_weights = torch.cat([self_weights, candidate_weights], dim=0)
        else:
            assign_src = keep_idx
            assign_anchor = keep_idx
            assign_weights = torch.exp(-logvar[keep_idx].mean(dim=-1, keepdim=True).clamp(min=-6.0, max=6.0))

        coarse_index = anchor_map[assign_anchor]
        num_coarse = keep_idx.numel()

        weight_sum = x.new_zeros((num_coarse, 1))
        weight_sum.index_add_(0, coarse_index, assign_weights)
        weight_sum = weight_sum.clamp_min(1e-6)

        coarse_x = _scatter_sum(assign_weights * x[assign_src], coarse_index, num_coarse) / weight_sum
        coarse_pos = _scatter_sum(assign_weights * pos[assign_src], coarse_index, num_coarse) / weight_sum
        coarse_mu = _scatter_sum(assign_weights * mu[assign_src], coarse_index, num_coarse) / weight_sum

        src_var = logvar[assign_src].exp().clamp_min(1e-6)
        centered = mu[assign_src] - coarse_mu[coarse_index]
        coarse_var = _scatter_sum(assign_weights * (src_var + centered.pow(2)), coarse_index, num_coarse) / weight_sum
        coarse_logvar = coarse_var.clamp_min(1e-6).log()

        return {
            "x": coarse_x,
            "pos": coarse_pos,
            "mu": coarse_mu,
            "logvar": coarse_logvar,
            "batch": batch[keep_idx],
            "keep_idx": keep_idx,
        }

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        stride_override: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        mu = self.mu_head(x)
        logvar = self.logvar_head(x).clamp(min=-6.0, max=6.0)

        if edge_index.numel() == 0 or x.size(0) <= 1:
            coarse_x = self.project(x)
            coarse_edge_index, coarse_edge_attr = self._build_coarse_graph(pos, batch)
            if coarse_edge_index.numel() > 0:
                coarse_global = _global_mean(coarse_x, batch)[batch]
                coarse_x = coarse_x + self.process(
                    torch.cat([coarse_x, coarse_global], dim=-1),
                    coarse_edge_index,
                    coarse_edge_attr,
                )
            return {
                "x": coarse_x,
                "pos": pos,
                "edge_index": coarse_edge_index,
                "edge_attr": coarse_edge_attr,
                "batch": batch,
                "mu": mu,
                "logvar": logvar,
                "scores": torch.ones((x.size(0),), device=x.device, dtype=x.dtype),
            }

        stats = self._neighbor_stats(mu, logvar, edge_index)
        score_input = torch.cat(
            [
                mu,
                stats["neighbor_mu_mean"],
                stats["neighbor_mu_std"],
                stats["confidence"],
                stats["novelty"],
                stats["agreement"],
            ],
            dim=-1,
        )
        learned_score = torch.sigmoid(self.pool(score_input)).squeeze(-1)

        agreement_score = 0.5 * (stats["agreement"].squeeze(-1) + 1.0)
        confidence = stats["confidence"].squeeze(-1)
        novelty = stats["novelty"].squeeze(-1)

        core_score = 0.7 * (confidence * agreement_score) + 0.3 * learned_score
        outlier_score = 0.7 * (confidence * novelty) + 0.3 * learned_score

        keep_idx = self._select_nodes(core_score, outlier_score, batch, stride_override=stride_override)
        pooled = self._aggregate(x, pos, mu, logvar, edge_index, batch, keep_idx)

        coarse_x = self.project(pooled["x"])
        coarse_edge_index, coarse_edge_attr = self._build_coarse_graph(pooled["pos"], pooled["batch"])
        if coarse_edge_index.numel() > 0:
            coarse_global = _global_mean(coarse_x, pooled["batch"])[pooled["batch"]]
            coarse_x = coarse_x + self.process(
                torch.cat([coarse_x, coarse_global], dim=-1),
                coarse_edge_index,
                coarse_edge_attr,
            )

        return {
            "x": coarse_x,
            "pos": pooled["pos"],
            "edge_index": coarse_edge_index,
            "edge_attr": coarse_edge_attr,
            "batch": pooled["batch"],
            "mu": pooled["mu"],
            "logvar": pooled["logvar"],
            "scores": learned_score,
        }


class GraphEncoder(nn.Module):
    # input: graph in RGB, knn edges
    # output: graph pooled to latent_dims
    # graph encoder performs resnet-like coarsening
    # project graph to high dimensional space
    # use latent space distribution to pool

    def __init__(self, latent_dim: int, in_dims: int = 6, base_k: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_dims = in_dims
        self.layers_mlp = 4
        self.base_k = max(1, int(base_k))

        # 4-stage pooling: 32 -> 64 -> 128 -> 256
        dim_1 = int(latent_dim / 8)
        dim_2 = int(latent_dim / 4)
        dim_3 = int(latent_dim / 2)

        self.project_graph = MLP(
            in_channels=in_dims,
            hidden_channels=dim_1,
            out_channels=dim_1,
            num_layers=2,
            act="gelu",
            norm="layer",
        )
        self.project_edges = MLP(
            in_channels=3,
            hidden_channels=dim_1,
            out_channels=dim_1,
            num_layers=2,
            act="gelu",
            norm="layer",
        )
        self.fine_conv = ModuleList(
            [GENConv(dim_1 * 2, dim_1, norm="layer", msg_norm=True, edge_dim=dim_1) for _ in range(self.layers_mlp)]
        )
        self.conv_blocks = ModuleList(
            [
                ConvBlock(in_dims=dim_1, out_dims=dim_2, stride=4, outlier_ratio=0.10),
                ConvBlock(in_dims=dim_2, out_dims=dim_3, stride=4, outlier_ratio=0.12),
                ConvBlock(in_dims=dim_3, out_dims=latent_dim, stride=4, outlier_ratio=0.15),
            ]
        )
        self.node_mu_head = nn.Linear(latent_dim, latent_dim)
        self.node_logvar_head = nn.Linear(latent_dim, latent_dim)
        self.readout_gate = nn.Linear(latent_dim * 2, 1)
        self.graph_proj = MLP(
            in_channels=latent_dim * 2,
            hidden_channels=latent_dim,
            out_channels=latent_dim,
            num_layers=2,
            act="gelu",
            norm="layer",
        )
        self.mu_head = nn.Linear(latent_dim, latent_dim)
        self.logvar_head = nn.Linear(latent_dim, latent_dim)

    def _graph_inputs(self, graph: Data | Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(graph, dict):
            pos = graph.get("pos")
            rgb = graph.get("rgb")
            x = graph.get("x")
            edge_index = graph.get("edge_index")
            batch = graph.get("batch")
        else:
            pos = getattr(graph, "pos", None)
            rgb = getattr(graph, "rgb", None)
            x = getattr(graph, "x", None)
            edge_index = getattr(graph, "edge_index", None)
            batch = getattr(graph, "batch", None)

        if pos is None:
            if x is None or x.size(1) < 3:
                raise ValueError("GraphEncoder expects `pos` or an `x` tensor whose first 3 channels are XYZ.")
            pos = x[:, :3]

        if x is None:
            if rgb is None:
                raise ValueError("GraphEncoder expects either `x` or both `pos` and `rgb`.")
            x = torch.cat([pos, rgb], dim=-1)

        x = x.float()
        pos = pos.float()
        batch = _ensure_batch(batch, x.size(0), x.device)

        if edge_index is None:
            if x.size(0) <= 1:
                edge_index = torch.empty((2, 0), device=x.device, dtype=torch.long)
            else:
                k = min(self.base_k, x.size(0) - 1)
                edge_index = knn_graph(pos, k=k, batch=batch, loop=False)
        else:
            edge_index = edge_index.to(device=x.device, dtype=torch.long)

        if edge_index.numel() > 0:
            edge_index = to_undirected(edge_index, num_nodes=x.size(0))

        return x, pos, edge_index, batch

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        if (not self.training) or (not sample):
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        graph: Data | Dict[str, torch.Tensor],
        target_nodes: int,
        sample: bool = False,
    ) -> Dict[str, torch.Tensor]:
        x, pos, edge_index, batch = self._graph_inputs(graph)

        x = self.project_graph(x)
        if edge_index.numel() > 0:
            src, dst = edge_index
            edge_attr = self.project_edges(pos[src] - pos[dst])
            for conv in self.fine_conv:
                x_global = _global_mean(x, batch)[batch]
                x = x + conv(torch.cat([x, x_global], dim=-1), edge_index, edge_attr)

        block_summaries = []
        for block in self.conv_blocks:
            # Keep pooling behavior architectural rather than steering it
            # toward a requested node count.
            block_out = block(x, pos, edge_index, batch)
            x = block_out["x"]
            pos = block_out["pos"]
            edge_index = block_out["edge_index"]
            batch = block_out["batch"]
            block_summaries.append(block_out["scores"].mean().detach())

        node_mu = self.node_mu_head(x)
        node_logvar = self.node_logvar_head(x).clamp(min=-6.0, max=6.0)
        node_conf = torch.exp(-node_logvar.mean(dim=-1, keepdim=True).clamp(min=-6.0, max=6.0))

        readout_logits = self.readout_gate(torch.cat([x, node_mu], dim=-1)).squeeze(-1) + node_conf.squeeze(-1).log()
        readout_weights = softmax(readout_logits, batch)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        pooled_x = _scatter_sum(readout_weights.unsqueeze(-1) * x, batch, num_graphs)
        pooled_mu = _scatter_sum(readout_weights.unsqueeze(-1) * node_mu, batch, num_graphs)

        h = self.graph_proj(torch.cat([pooled_x, pooled_mu], dim=-1))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(min=-6.0, max=6.0)
        z = self.reparameterize(mu, logvar, sample=sample)

        return {
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "node_x": x,
            "node_mu": node_mu,
            "node_logvar": node_logvar,
            "node_pos": pos,
            "node_batch": batch,
            "edge_index": edge_index,
            "readout_weights": readout_weights,
            "pool_score_mean": x.new_tensor(0.0) if not block_summaries else torch.stack(block_summaries).mean(),
        }


class Locator(nn.Module):
    def __init__(
        self,
        nodes_dim: int,
        latent_dim: int,
        hidden_dim: int,
        scene_pool_nodes: int,
        view_pool_nodes: int,
    ):
        super().__init__()
        self.nodes_dim = nodes_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.scene_pool_nodes = max(1, int(scene_pool_nodes))
        self.view_pool_nodes = max(1, int(view_pool_nodes))

        self.encoder = GraphEncoder(latent_dim=latent_dim, in_dims=nodes_dim)
        self.match_mlp = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _batch_to_graph(self, pcd: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "x": pcd.float(),
            "pos": pcd[:, :3].float(),
            "edge_index": edge_index.to(dtype=torch.long, device=pcd.device) if edge_index is not None else None,
            "batch": _ensure_batch(batch, pcd.size(0), pcd.device),
        }

    def _kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)).mean()

    def _scene_match(
        self,
        scene_encoding: Dict[str, torch.Tensor],
        view_encoding: Dict[str, torch.Tensor],
        ) -> Dict[str, torch.Tensor]:
        scene_mu = scene_encoding["node_mu"].float()
        scene_logvar = scene_encoding["node_logvar"].float()
        scene_pos = scene_encoding["node_pos"].float()
        view_query = view_encoding["z"].float()
        view_logvar = view_encoding["logvar"].float()

        pair_scene = scene_mu.unsqueeze(0).expand(view_query.size(0), -1, -1)
        pair_view = view_query.unsqueeze(1).expand(-1, scene_mu.size(0), -1)
        pair_features = torch.cat([pair_scene, pair_view, torch.abs(pair_scene - pair_view)], dim=-1)

        learned_logits = self.match_mlp(pair_features).squeeze(-1)
        cosine = F.cosine_similarity(pair_scene, pair_view, dim=-1)

        scene_conf = torch.exp(-scene_logvar.mean(dim=-1).clamp(min=-6.0, max=6.0)).unsqueeze(0)
        view_conf = torch.exp(-view_logvar.mean(dim=-1, keepdim=True).clamp(min=-6.0, max=6.0))
        practical_logits = cosine * torch.sqrt(scene_conf * view_conf)

        logits = learned_logits + practical_logits
        weights = torch.softmax(logits, dim=-1)
        pred_loc = weights @ scene_pos

        return {
            "match_logits": logits,
            "match_weights": weights,
            "pred_loc": pred_loc,
        }

    def encode_scene(self, scene_graph: Data, sample: bool = False):
        return self.encoder(scene_graph, target_nodes=self.scene_pool_nodes, sample=sample)

    def encode_view(self, walk_batch, sample: bool = False):
        graph = self._batch_to_graph(walk_batch.pcd, walk_batch.edge_index, walk_batch.batch)
        return self.encoder(graph, target_nodes=self.view_pool_nodes, sample=sample)

    def forward_with_scene_encoding(self, scene_encoding, walk_batch, sample_view: bool = True):
        view_encoding = self.encode_view(walk_batch, sample=sample_view)
        match_outputs = self._scene_match(scene_encoding, view_encoding)

        outputs = {
            **match_outputs,
            "target_loc": walk_batch.loc.float() if walk_batch.loc is not None else None,
            "scene_mu": scene_encoding["mu"],
            "scene_logvar": scene_encoding["logvar"],
            "view_mu": view_encoding["mu"],
            "view_logvar": view_encoding["logvar"],
            "scene_pool_score_mean": scene_encoding["pool_score_mean"],
            "view_pool_score_mean": view_encoding["pool_score_mean"],
        }
        return outputs

    def forward(self, scene_graph, walk_batch, sample_scene: bool = False, sample_view: bool = True):
        scene_encoding = self.encode_scene(scene_graph, sample=sample_scene)
        return self.forward_with_scene_encoding(scene_encoding, walk_batch, sample_view=sample_view)

    def loss(self, outputs):
        if outputs["target_loc"] is None:
            raise ValueError("Locator.loss expects `walk_batch.loc` to be available.")

        loc_loss = F.mse_loss(outputs["pred_loc"], outputs["target_loc"])
        kl_loss = self._kl_divergence(outputs["view_mu"], outputs["view_logvar"])

        weights = outputs["match_weights"].clamp_min(1e-8)
        entropy = -(weights * weights.log()).sum(dim=-1).mean()

        loss = loc_loss + (1e-2 * kl_loss) + (1e-4 * entropy)
        metrics = {
            "loc_loss": float(loc_loss.detach().item()),
            "kl_loss": float(kl_loss.detach().item()),
            "match_entropy": float(entropy.detach().item()),
            "pool_score_mean": float(
                0.5 * (outputs["scene_pool_score_mean"].detach().item() + outputs["view_pool_score_mean"].detach().item())
            ),
        }
        return loss, metrics
