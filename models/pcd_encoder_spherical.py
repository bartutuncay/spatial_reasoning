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
# pcd: [r,phi,theta,R,G,B] (relative positions)
# edges: [dist]
class SineActivation(nn.Module):
    def __init__(self,omega_0 = 30.0):
        super(SineActivation,self).__init__()
        self.omega_0 = omega_0

    def forward(self,x):
        return torch.sin(self.omega_0*x)

class MLP_sin(nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,num_layers,act='sigmoid',norm='layer'):
        super().__init__()
        layers = []

        layers.append(nn.Linear(in_channels,hidden_channels))
        if norm == 'layer':
            layers.append(nn.LayerNorm(hidden_channels))
        if act == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif act == 'sine':
            layers.append(SineActivation())
        
        for _ in range(num_layers-1):
            layers.append(nn.Linear(hidden_channels,hidden_channels))
            if norm == 'layer':
                layers.append(nn.LayerNorm(hidden_channels))
            if act == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif act == 'sine':
                layers.append(SineActivation())
            
        layers.append(nn.Linear(hidden_channels,out_channels))
        self.network = nn.Sequential(*layers)

    def forward(self,x):
        return self.network(x)

class PCDEncoder(nn.Module):
    def __init__(self,latent_dim,layers,layers_mlp,z_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.z_dim = z_dim
        self.pcd_proj = MLP_sin(in_channels=6,hidden_channels=latent_dim,out_channels=latent_dim,num_layers=layers_mlp,act='tanh',norm='layer')
                
        self.conv_layers = ModuleList([
            GCNConv(in_channels=latent_dim*2,out_channels=latent_dim)
            for _ in range(layers)])
        
        self.latent_proj = MLP_sin(in_channels=latent_dim,hidden_channels=latent_dim,out_channels=z_dim,num_layers=layers_mlp,act='tanh',norm='layer')

        return

    def to_spherical(self,pcd):
        r = torch.linalg.norm(pcd,dim=1)
        theta = torch.acos((pcd[:,2]/(r+1e-8)).clamp(-1.0,1.0))
        phi = torch.atan2(pcd[:,1],pcd[:,0])
        phi = (phi + 2*torch.pi) % (2*torch.pi)
        return torch.stack([r,theta,phi],dim=1)

    def forward(self,pcd,batch,edge_index,edge_weights):
        pcd = self.to_spherical(pcd)
        x = self.pcd_proj(pcd)
        for conv in self.conv_layers:
            x_global = pool.global_mean_pool(x,batch)
            x_expanded = x_global[batch]
            m = conv(torch.cat([x, x_expanded], dim=1), edge_index, edge_weight=edge_weights)
            x = x + m
        g = pool.global_mean_pool(x,batch)
        z = self.latent_proj(g)

        return z