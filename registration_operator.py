import torch
from torch_geometric.data import Data
import torch_geometric.nn.functional as F_geom
from torch_geometric.nn import SimpleConv, MLP, Linear
from torch.utils.data import Dataset
from torch_geometric.nn import pool, knn_graph, radius
from torch_geometric.loader import ClusterData, ClusterLoader, DataLoader
from torch.nn import ModuleList
import torch.nn as nn
import math
import pandas as pd
import os
from typing import Optional, Tuple, Dict
import glob
import torch.nn.functional as F
from models.image_encoder_no_cond import ImageEncoder
from models.pcd_encoder import PCDEncoder
from models.locator import Locator
from models.recon_largepcd import ReconPCD
from models.autoencoder_model import ImageGraphAE

from preprocessing_src.dataloader_partial import make_loader_partial
from preprocessing_src.dataloader_walk import make_loader_walk
from preprocessing_src.dataloader import make_loader


## registration training pipeline:
# load partial point cloud with known missing part

# construct data graphs in parallel from latents/kept point cloud
# add new latent <-> correlate latent transform and physical transform

## registration model logic:
# give entire knn subgraph with missing partition to model
# feed initial physical differences as input to the model
# compare transformation losses
# inverse function: find translations from given partial pcd to the neighbors

## TODO: navigation mixture of experts: translation, registration, location in pcd

device = torch.device('cuda')
torch.set_default_dtype(torch.float32)


class Registration(nn.Module):
    def __init__(self,in_channels,pcd_dims,latent_dim):
        super().__init__()
        self.in_channels = in_channels
        self.pcd_dims = pcd_dims
        self.latent_dim = latent_dim
        self.pcd_encoder = PCDEncoder(latent_dim=64,layers=8,layers_mlp=2,z_dim=latent_dim)
        self.recon = ReconPCD(in_channels=in_channels,out_channels=6,latent_dim=latent_dim,layers=4,layers_mp=2)

    def forward(self,pcd,pcd_n): #pcd: point cloud to transform, pcd_n: neighboring point clouds
        #relative transform
        #visible point cloud portion
        #point_location_pcd,node_translation,ei_n_t = self.locator(pcd,pcd_ei,pcd_ea,latents)

        # process point clouds to latent
        pcd_l = self.pcd_encoder(pcd.x.float(),pcd.batch,pcd.edge_index,pcd.edge_attr.float())
        pcd_n_l = self.pcd_encoder(pcd_n.x.float(),pcd_n.batch,pcd_n.edge_index,pcd_n.edge_attr.float())
        
        # generate knn graph with latent neighbors
        print(pcd_n_l.shape)
        ei_conn = knn_graph(pcd_n, k=2, loop=False) #edge indices between known nodes, physical
        ea_conn = (pcd_n[ei_conn[0]]-pcd_n[ei_conn[1]]).norm(dim=1)
        pcd_l_combined = torch.cat([pcd_l,pcd_n_l],dim=0)
        N = pcd_n_l.size(0)
        ei_latent = torch.stack([torch.zeros(N,dtype=torch.long),torch.arange(1,N+1,dtype=torch.long)],dim=0)
        ew_latent = torch.norm(pcd_n_l-pcd_l,dim=1) #latent distances to encoded node to encoded neighbor nodes

        #point cloud recon
        pcd_t = self.recon(pcd_n,pcd_n_l,pcd,pcd_l,ei_conn,ei_latent,ea_conn,ew_latent,pcd_l_combined,pcd.batch) #pcd_ei_l,pcd_ea,pcd_ea_l

        return pcd_t

# coarsen point cloud since graph is HUGE --> file saved

EPOCHS = 20000

alias = '0318'
model = Registration(in_channels=6,pcd_dims=6,latent_dim=128).to(device)
#loader = make_loader('../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/rw_latents/z_img', batch_size=1, shuffle=True, num_workers=4)
pcd_graph = torch.load('../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/scan_pcd_graph/combined_aligned.pt',weights_only=False)
opt = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,0,-1)

#dataset = ClusteredDataset("../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/scan_pcd_graph/clusters/", pattern="*.pt")
#cluster_loader = DataLoader(dataset, batch_size=32, shuffle=True)
#cluster_loader = make_loader_subgraph("../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/scan_pcd_graph/clusters/", pattern="*.pt", batch_size=4, shuffle=True, num_workers=4)
partial_loader = make_loader_partial("../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/scan_pcd_graph/clusters/", pattern="*.pt", batch_size=4, shuffle=True, num_workers=4)
walk_loader = make_loader_walk('../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/rw_latents/z_img', batch_size=16, shuffle=True, num_workers=4)

def random_transform(loader_dict, max_angle_deg=45, tr_factor=10,device = torch.device('cuda')):
    # transform the point cloud with random noise
    # transform the initial point cloud to face the camera directly
    pcd = loader_dict['graph'].to(device)
    #pcd_n = loader_dict['neighbor_graphs'].to(device)
    #device = pcd.device

    angles = (torch.rand(3, device=device) * 2 - 1) * max_angle_deg * torch.pi / 180
    # rotation from bounded Euler angles
    cx, cy, cz = torch.cos(angles)
    sx, sy, sz = torch.sin(angles)

    Rx = torch.tensor([[1, 0, 0],
                       [0, cx, -sx],
                       [0, sx, cx]], device=device)

    Ry = torch.tensor([[cy, 0, sy],
                       [0, 1, 0],
                       [-sy, 0, cy]], device=device)

    Rz = torch.tensor([[cz, -sz, 0],
                       [sz,  cz, 0],
                       [0,   0,  1]], device=device)

    R = Rz @ Ry @ Rx

    print(pcd)
    #mins = pcd.min(dim=0).values
    #maxs = pcd.max(dim=0).values
    #diameter = torch.norm(maxs - mins)

    t = (torch.rand(3, device=device) * 2 - 1) * tr_factor# * diameter

    pcd_t = pcd.pos.float() @ R.float().T + t

    # return transformed point 
    # cloud, relative distances (to centers)

    return pcd_t#, R, t



## predicted point positions <-chamfer loss-> actual point positions --calculated, loss function

def loss_function(pcd,pcd_pred):
    diff = pcd.unsqueeze(2)-pcd_pred.unsqueeze(1)    
    dist = torch.sum(diff**2,dim=-1)
    min_p_to_q,_ = torch.min(dist,dim=2)
    min_q_to_p,_ = torch.min(dist,dim=1)
    loss = min_p_to_q.mean(dim=1) + min_q_to_p(dim=1)
    return loss.mean()

model.train(True)

def run_epoch(partial_loader):
    for part in partial_loader:
        part['neighbor_graphs'] = part['neighbor_graphs'].to(device)
        part['graph'] = part['graph'].to(device)
        # ground truth distances: center of each cluster xyz - vantage point translation
        part['graph'].pos = random_transform(part)
        part['graph'].x[:,:3] = part['graph'].pos

        # load subgraph on locator model
        part_reg = model(part['graph'],part['neighbor_graphs'])
        loss = loss_function(part['graph'],part_reg)
        loss.backward()
        opt.step()

    return

loss_records = []
for epoch in range(EPOCHS):
    write_header = False
    loss, loss_dict = run_epoch(partial_loader)
    break
    row = {"epoch": epoch,**loss_dict}
    pd.DataFrame([row]).to_csv(csv_path,mode="a",header=write_header,index=False)
    loss_records.append({"epoch": epoch, "train_loss": loss_dict})
    if epoch % 10 == 0:
        print(f'epoch {epoch}',loss_dict)
    if epoch % 200 == 0:
        torch.save(model.state_dict(), f"../../../scratch/btuncay/cog/gnn_spatial_reasoning/model_recon/{alias}/model_weights_{epoch}.pt")
torch.save(model.state_dict(), f"../../../scratch/btuncay/cog/gnn_spatial_reasoning/model_recon/{alias}/model_weights.pt")
print('completed')

##LATER:
# make subgraph from points based on segmentation for matching
# subgraph - visible pcd <=> cosine similarity
# implement self-supervision function
# test kl divergence/regularization on latents (small autoencoder)
# transformer-based backbone