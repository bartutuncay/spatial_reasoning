import torch
from torch_geometric.data import Data
import torch_geometric.nn.functional as F_geom
from torch_geometric.nn import GCNConv, MLP, Linear
from torch_geometric.nn import pool
from torch.nn import ModuleList
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict
import torch.nn.functional as F

# Data format:
# pcd: [x,y,z,R,G,B] (relative positions)
# edges: [dist]

class PCDEncoder(nn.Module):
    def __init__(self,latent_dim,layers,layers_mlp,z_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.z_dim = z_dim
        self.pcd_proj = MLP(in_channels=6,hidden_channels=latent_dim,out_channels=latent_dim,num_layers=layers_mlp,act='relu',norm='layer')
                
        self.conv_layers = ModuleList([
            GCNConv(in_channels=latent_dim*2,out_channels=latent_dim)
            for _ in range(layers)])

        
        
        self.latent_proj = MLP(in_channels=latent_dim,hidden_channels=latent_dim,out_channels=z_dim,num_layers=layers_mlp,act='relu',norm='layer')

        return

    def forward(self,pcd,batch,edge_index,edge_weights):
        x = self.pcd_proj(pcd)
        for conv in self.conv_layers:
            x_global = pool.global_mean_pool(x,batch)
            x_expanded = x_global[batch]
            m = conv(torch.cat([x, x_expanded], dim=1), edge_index, edge_weight=edge_weights)
            x = x + m
        g = pool.global_mean_pool(x,batch)
        z = self.latent_proj(g)

        return z