import torch
from torch_geometric.data import Data
import torch_geometric.nn.functional as F_geom
from torch_geometric.transforms import KNNGraph
from torch_geometric.nn import GCNConv, MLP, Linear, pool, knn_graph
from torch.nn import ModuleList
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict
import torch.nn.functional as F

# Data format:
# Input:
# latent    [N,latent_dim]
# transform [tx,ty,tz]
# view dir  [u,v,w]
# reconstructed PCD slice
# Output:
# large pcd [x,y,z,R,G,B] (relative positions)

# TODO: implement registration-type network; using latent space to reconstruct points

## registration logic:
# use latent spaces to construct a graph of latents
# convert latent graph transforms into physical transforms (localization)
# subdivide selected graph and calculate distances of each cluster to neighbors

class ReconPCD(nn.Module):
    def __init__(self,in_channels,out_channels,latent_dim,layers,layers_mp):
        super().__init__()

        self.translation_projection = nn.Sequential(
            nn.Linear(in_channels,in_channels),
            nn.Tanh(),
            nn.Linear(in_channels,in_channels),
            nn.Tanh(),
            nn.Linear(in_channels,out_channels))
        
        self.rotation_projection = nn.Sequential(
            nn.Linear(in_channels,in_channels),
            nn.Tanh(),
            nn.Linear(in_channels,in_channels),
            nn.Tanh(),
            nn.Linear(in_channels,out_channels))

        self.conv_layers = ModuleList([
            GCNConv(in_channels=latent_dim*2,out_channels=latent_dim)
            for _ in range(layers)])
        
        self.conv_layers_mp = ModuleList([
            GCNConv(in_channels=latent_dim*2,out_channels=latent_dim)
            for _ in range(layers_mp)])
        
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
    
    def top4_by_cosine(q, X, coords):
        """
        q:      (d,)        query vector
        X:      (N, d)      candidate vectors
        coords: (N, 3)      coordinates associated with X
        returns:
            top_coords: (4, 3)
            top_scores: (4,)
            top_indices: (4,)
        """
        # Normalize (important for cosine similarity)
        q_norm = F.normalize(q, dim=0)
        X_norm = F.normalize(X, dim=1)
        # Cosine similarity
        sims = torch.matmul(X_norm, q_norm)   # (N,)
        # Top 4
        top_scores, top_indices = torch.topk(sims, k=4, largest=True)
        # Retrieve coordinates
        top_coords = coords[top_indices]
        
        return top_coords, top_scores, top_indices
    
    def trilaterate_3d_diff(anchor_coords, anchor_dists):
        """
        Differentiable linear trilateration.
        anchor_coords: (B, K, 3)
        anchor_dists:  (B, K)

        Returns:
            p: (B, 3)
        """
        a1 = anchor_coords[:, :1, :]          # (B, 1, 3)
        d1 = anchor_dists[:, :1]              # (B, 1)

        ai = anchor_coords[:, 1:, :]          # (B, K-1, 3)
        di = anchor_dists[:, 1:]              # (B, K-1)

        A = 2.0 * (ai - a1)                   # (B, K-1, 3)

        b = (
            d1**2 - di**2
            + (ai**2).sum(dim=2)
            - (a1**2).sum(dim=2)
        )                                     # (B, K-1)

        # torch.linalg.lstsq supports batches
        p = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)  # (B, 3)
        return p

    def forward(self,pcd_latent,pcd_partial,pcd_partial_latent,ei_conn,ei_latent,ew_conn,ew_latent,pcd_l_comb,batch):
        #partial: point cloud to be registered
        pcd_conn = KNNGraph(pcd_latent) # generate knn graph from partial pcds
        
        print(pcd_conn)

        for conv in self.conv_layers: # message passing between connected subgraphs
            x_global = pool.global_mean_pool(pcd_conn,batch)
            x_expanded = x_global[batch]
            m = conv(torch.cat([x, x_expanded], dim=1), ei_conn, edge_weight=ew_conn)
            x = x + m
        
        ##TODO: include physical transforms as input

        # message all nodes --> selected node
        for conv in self.conv_layers_mp: # message passing between connected subgraphs (edge weights come from latent distances)
            x_global = pool.global_mean_pool(x,batch)
            x_expanded = x_global[batch]
            m = conv(torch.cat([p, x_expanded], dim=1), ei_latent, edge_weight=ew_latent)
            p = p + m
        
        # decode rigid translation from edge connections --> R: rotation, t: translation
        # translation comes from edge weights only
        # tr denotes the center point of the point cloud after movement
        tr = self.translation_projection(p) # [Neighbors, 1 (weight)]
        # apply rigid transform to selected point cloud by trilateration (use top 4 matches)
        # get distances from top 4 matches to mean point
        top4_coords, _, top4_idx = self.top4_by_cosine(pcd_partial_latent,pcd_latent,pcd_partial)
        pcd_center = self.trilaterate_3d_diff(top4_coords,tr)
        pcd_partial = (pcd_partial.mean()-pcd_center) + pcd_partial

        # decode rotation from latent differences
        rt = self.rotation_projection(p[top4_idx]) # [Neighbors, 1 (weight)]
        pcd_r = pcd_partial @ rt.T

        return pcd_r