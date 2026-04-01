import torch
from torch_geometric.data import Data
import torch_geometric.nn.functional as F_geom
from torch_geometric.nn import SimpleConv, MLP, Linear
from torch.utils.data import Dataset
from torch_geometric.nn import pool
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

from preprocessing_src.dataloader_subgraph import make_loader_subgraph
from preprocessing_src.dataloader_partial import make_loader_partial
from preprocessing_src.dataloader_walk import make_loader_walk
from preprocessing_src.dataloader import make_loader

device = torch.device('cuda')
torch.set_default_dtype(torch.float32)
##TODO:
# use latent embedding from autoencoder for representations
# make graph with latents <-> align with transformation + large pcd portion
# partial reconstruction of large pcd <-> latent construction from pcd slices

class LatentLocationAE(nn.Module):
    def __init__(self,in_channels,pcd_dims,latent_dim):
        super().__init__()
        self.in_channels = in_channels
        self.pcd_dims = pcd_dims
        self.latent_dim = latent_dim
        self.locator = Locator(self.in_channels,self.latent_dim,2,3,4,2)
        #self.recon = ReconPCD()

    def forward(self,pcd,pcd_ei,pcd_ea,latents):
        #relative transform
        #visible point cloud portion
        point_location_pcd,node_translation,ei_n_t = self.locator(pcd,pcd_ei,pcd_ea,latents) 

        ##TODO: point cloud recon

        return point_location_pcd, node_translation, ei_n_t

# coarsen point cloud since graph is HUGE --> file saved

EPOCHS = 20000

alias = '0316'
print('starting training')
print(alias)
os.makedirs(f"../../../scratch/btuncay/cog/gnn_spatial_reasoning/model_locator/{alias}",exist_ok=True)
csv_path = f"../../../scratch/btuncay/cog/gnn_spatial_reasoning/model_locator/{alias}/losses.csv"
write_header = not os.path.exists(csv_path)


model = LatentLocationAE(in_channels=6,pcd_dims=6,latent_dim=128).to(device)
loader = make_loader('../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/rw_latents/z_img', batch_size=1, shuffle=True, num_workers=4)
pcd_graph = torch.load('../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/scan_pcd_graph/combined_aligned.pt',weights_only=False)
opt = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,0,-1)

#dataset = ClusteredDataset("../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/scan_pcd_graph/clusters/", pattern="*.pt")
#cluster_loader = DataLoader(dataset, batch_size=32, shuffle=True)
cluster_loader = make_loader_subgraph("../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/scan_pcd_graph/clusters/", pattern="*.pt", batch_size=4, shuffle=True, num_workers=4)
#partial_loader = make_loader_partial("../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/scan_pcd_graph/clusters/", pattern="*.pt", batch_size=4, shuffle=True, num_workers=4)
walk_loader = make_loader_walk('../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/rw_latents/z_img', batch_size=16, shuffle=True, num_workers=4)

def batch_equivariant_translation(t1, t2, v1, v2, eps=1e-9):
    # World displacement from cam1 to cam2
    t1 = t1.float()
    t2 = t2.float()
    v1 = v1.float()
    v2 = v2.float()
    dp = t2-t1
    # Cam1 forward (+z) in world
    z = F.normalize(v1, dim=-1, eps=eps)
    # World up
    B = t1.shape[0]
    uw = torch.tensor([0.0, 1.0, 0.0], device=t1.device, dtype=torch.float32)
    uw = uw.expand(B,3)
    # Right (+x) = normalize(uw x z); fallback if degenerate
    x = torch.cross(uw, z, dim=-1)
    x_norm = x.norm(dim=-1, keepdim=True)
    bad = x_norm.squeeze(-1) < eps

    if bad.any():
        # forward is ~parallel to world-up; choose a fallback up
        uw_fallback = torch.tensor([0.0, 0.9, 0.0], dtype=torch.float32).expand(B,3)
        x_fallback = torch.cross(uw_fallback, z, dim=-1)
        x_fallback_norm = x_fallback.norm(dim=-1, keepdim=True)

        x = torch.where(bad.unsqueeze(-1), x_fallback, x)
        x_norm = torch.where(bad.unsqueeze(-1), x_fallback_norm, x_norm)
        if x_norm < eps:
            raise ValueError("Degenerate orientation: cannot construct basis.")
    x /= x_norm
    # Up (+y) = z x x  (ensures right-handed basis)
    y = torch.cross(z, x, dim=-1)
    # Rotation camera --> world, columns are camera axes in world
    R_cw = torch.stack((x, y, z),dim=-1)
    # displacement in camera 1 coordinates: t = R_wc * dp = R_cw.T * dp
    t_c1 = torch.bmm(R_cw.transpose(1, 2), dp.unsqueeze(-1)).squeeze(-1)
    rot1 = v2 - v1
    return t_c1, rot1


## latent graph node pose <-edges in xyz-> pcd cluster mean xyz --calculated, loss function

def loss_function(walk,subgraph_mean,pred,pred_node):

    translation_cluster_obs = (walk['loc'][:, None, :] - subgraph_mean[None, :, :]).reshape(-1, 3)

    #translation_cluster_obs = walk['loc']-subgraph_mean
    dist_loss = F.mse_loss(translation_cluster_obs,pred)
    
    ## translation loss from latent edge attrs
    # get pairwise translation between latents
    dist = torch.cdist(walk['loc'], walk['loc'])
    dist.fill_diagonal_(1e6)
    dist_pairs = torch.stack([torch.arange(0,dist.shape[0]).to(device),dist.min(dim=1).indices],dim=1)
    # get translation between relative locations/view dirs in walks
    tr, vd = walk['loc'][dist_pairs], walk['viewdir'][dist_pairs] # [B, 2, 3]
    tr1, tr2, vd1, vd2 = tr[:,0],tr[:,1],vd[:,0],vd[:,1]
    node_tr, node_vd = batch_equivariant_translation(tr1, tr2, vd1, vd2)
    node_true = torch.stack([node_tr, node_vd],dim=1)
    node_true = node_true.view(-1,6).repeat(subgraph_mean.shape[0],1)
    

    translation_loss = F.mse_loss(node_true,pred_node)
    loss = 0.8 * dist_loss + 0.2 * translation_loss

    return loss, {'distance_loss':dist_loss.item(), 'node_loss':translation_loss.item()}


model.train(True)

def run_epoch(cluster_loader,walk_loader):
    for part in cluster_loader:
        part = part.to(device)
        mean_xyz = pool.global_mean_pool(part.pos, part.batch)
        #mean_xyz = part.pos.mean(dim=0)
        for walk in walk_loader:
            print(part.dtype,walk.dtype)
            walk = walk.to(device) # DataBatch(latent=[16, 128], loc=[16, 3], viewdir=[16, 3])
            # ground truth distances: center of each cluster xyz - vantage point translation
            #print(walk,type(walk))
            # load subgraph on locator model
            h,t,ei_walk = model(part, part.edge_index, part.edge_attr.float(),walk)
            print(walk,h.shape,t.shape)
            loss, loss_dict = loss_function(walk,mean_xyz,h,t) # h:[64,3]; t:[64,6]
            loss.backward()
            opt.step()
        #part_rep = h.mean(dim=0)
    return loss, loss_dict

loss_records = []
for epoch in range(EPOCHS):
    write_header = False
    loss, loss_dict = run_epoch(cluster_loader,walk_loader)
    row = {"epoch": epoch,**loss_dict}
    pd.DataFrame([row]).to_csv(csv_path,mode="a",header=write_header,index=False)
    loss_records.append({"epoch": epoch, "train_loss": loss_dict})
    if epoch % 10 == 0:
        print(f'epoch {epoch}',loss_dict)
    if epoch % 200 == 0:
        torch.save(model.state_dict(), f"../../../scratch/btuncay/cog/gnn_spatial_reasoning/model_locator/{alias}/model_weights_{epoch}.pt")
torch.save(model.state_dict(), f"../../../scratch/btuncay/cog/gnn_spatial_reasoning/model_locator/{alias}/model_weights.pt")
print('completed')

##LATER:
# make subgraph from points based on segmentation for matching
# subgraph - visible pcd <=> cosine similarity
# implement self-supervision function
# test kl divergence/regularization on latents (small autoencoder)
# transformer-based backbone