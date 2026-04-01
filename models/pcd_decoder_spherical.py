import torch
from torch_geometric.data import Data
import torch_geometric.nn.functional as F_geom
from torch_geometric.nn import GCNConv, MLP, Linear, pool, knn_graph
from torch.nn import ModuleList
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict
import torch.nn.functional as F

# Data format:
# Input:
# latent    [latent_dim]
# Output:
# pcd       [x,y,z,R,G,B] (relative positions, from [r,theta,phi])
# edges     [dist]

class PCDDecoder(nn.Module):
    def __init__(self,nodes_dim:int,latent_dim:int,nodes_init:int,nodes_k:int,layers:int):
        super().__init__()
        self.latent_dim = latent_dim
        self.nodes_init = nodes_init # number of blank nodes to generate
        self.nodes_k = nodes_k # number of neighbors for each node
        self.layers = layers
        self.nodes_dim = nodes_dim

        # node generator latent --> N nodes
        self.node_mlp = nn.Sequential( 
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, self.nodes_init * self.nodes_dim),
        )

        # Edge scorer: (xi, xj, |xi-xj|, xi*xj) -> logit
        # N nodes --> latent
        in_edge = 4 * nodes_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_edge, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )
    
    def decode_nodes(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, latent_dim] or [latent_dim]
        if z.dim() == 1:
            z = z.unsqueeze(0)
        B = z.size(0)
        x = self.node_mlp(z).view(B, self.nodes_init, self.nodes_dim)
        return x

    def decode_edges_from_candidates(self, x: torch.Tensor, tau: float = 0.5):
        """
        x: [N, F] for a single graph
        Returns edge_index [2, E] and edge_prob [E]
        """
        # Candidate edges via kNN in feature space (often positions)
        edge_index_cand = knn_graph(x[:,:3], k=self.nodes_k, loop=False)  # [2, E_cand] - positional kNN

        src, dst = edge_index_cand[0], edge_index_cand[1]
        xi, xj = x[src], x[dst]
        feat = torch.cat([xi, xj, (xi - xj).abs(), xi * xj], dim=-1)
        logits = self.edge_mlp(feat).squeeze(-1)
        prob = torch.sigmoid(logits)

        keep = prob > tau
        edge_index = edge_index_cand[:, keep]
        edge_prob = prob[keep]

        # Add reverse edges (simple symmetrization)
        rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_index = torch.cat([edge_index, rev], dim=1)
        edge_prob = torch.cat([edge_prob, edge_prob], dim=0)

        return edge_index, edge_prob

    def to_spherical(self,pcd):
        r = torch.linalg.norm(pcd,dim=1)
        theta = torch.acos((pcd[:,2]/(r+1e-8)).clamp(-1.0,1.0))
        phi = torch.atan2(pcd[:,1],pcd[:,0])
        phi = (phi + 2*torch.pi) % (2*torch.pi)
        return torch.stack([r,theta,phi],dim=1)

    def to_cartesian(self,pcd):
        r = pcd[:,0]
        x = r * torch.sin(pcd[:,1]) * torch.cos(pcd[:,2])
        y = r * torch.sin(pcd[:,1]) * torch.sin(pcd[:,2])
        z = r * torch.cos(pcd[:,1])
        return torch.stack((x, y, z), dim=1)

    def forward(self, z: torch.Tensor, tau: float = 0.5):
        """
        z can be batched. Returns a list of Data objects if batched.
        """
        x_b = self.decode_nodes(z)  # [B, N, F]
        B = x_b.size(0)

        graphs = []
        for b in range(B):
            x = x_b[b]
            edge_index, edge_prob = self.decode_edges_from_candidates(x, tau=tau)

            pos = x[:, :3]  # [N, 3]
            src, dst = edge_index[0], edge_index[1]

            # Euclidean distance per edge
            edge_weight = (pos[src] - pos[dst]).norm(dim=-1) + 1e-8  # [E]

            graphs.append(
                Data(
                    x=x,edge_index=edge_index,
                    edge_prob=edge_prob,edge_weight=edge_weight,   # for GCNConv-style APIs
                    # alternatively: edge_attr=edge_weight.view(-1, 1)
                ))

        return graphs if B > 1 else graphs[0]