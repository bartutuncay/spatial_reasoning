#from comet_ml import start
#from comet_ml.integration.pytorch import log_model

import torch
from torch_geometric.data import Data
import torch_geometric.nn.functional as F_geom
from torch_geometric.nn import SimpleConv, MLP, Linear
from torch_geometric.nn import pool
from torch.nn import ModuleList
import torch_geometric.transforms as T
import torch.nn as nn
import math
import pandas as pd
from typing import Optional, Tuple, Dict
import os
import torch.nn.functional as F
from models.autoencoder_img_enc import ImageEncoder
from models.autoencoder_pcd_enc import PCDEncoder
from models.autoencoder_combined_dec import CombinedDecoder
from preprocessing_src.dataloader import MultiModalBatch, make_loader, collate_pt_dicts, PtDictFolderDataset

## Updated autoencoder pipeline:
# image/pcd --> latent --> image --> pcd
# all latents should be regularized (VAE)

## Autoencoder outline:
# Inputs:
# - image (2D array)
# - image rotation translation matrices
# - visible points pcd graph
# => collapsed into latent space
# Outputs:
# - image (2D array) --> visible points pcd graph

# Data format:
# image: [H,W,R,G,B]
# pcd: [x,y,z,R,G,B]

# Comet - logging
#experiment = start(api_key="7VD3oulgQdsrnNz60JDDhY86O",project_name="gnn-spatial-reasoning",workspace="btuncay")
#hyper_params = {'learning_rate': 1e-4,'steps': 10000,'batch_size':1}
#experiment.log_parameters(hyper_params)

device = torch.device('cuda')
torch.set_default_dtype(torch.float32)
alias = '0210_novis'

class ImageGraphAE(nn.Module):
    def __init__(self,latent_dim,use_z_norm=True,use_ss=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_ss = use_ss
        self.use_z_norm = use_z_norm
        basedim = 32
        self.img_enc = ImageEncoder(latent_dim=latent_dim, base_dim=basedim, token_dim=256)
        self.img_dec = ImageDecoder(latent_dim=latent_dim, base_dim=basedim, out_hw=(384,512), norm='group', groups=16, out_act='sigmoid')
        self.pcd_enc = PCDEncoder(latent_dim=64,layers=8,layers_mlp=2,z_dim=latent_dim)
        self.pcd_dec = PCDDecoder(nodes_dim=6,latent_dim=latent_dim,nodes_init=12000,nodes_k=3,layers=8)

        self.pcd_proj = MLP(in_channels=6,hidden_channels=latent_dim,out_channels=latent_dim,num_layers=2,act='relu',norm='layer')

        # visibility projection: [area, volume]
        self.vis_proj = MLP(in_channels=10,hidden_channels=32,out_channels=latent_dim,num_layers=4,act='relu',norm='layer')
        
        self.vis_head = nn.Linear(latent_dim,2)
        #self.pcd_node_head = nn.Linear(latent_dim,1) #for predicting number of nodes in the future
        return

    def cond(self, z, vis=None):
        if self.use_z_norm == True:
            z = F.layer_norm(z, (z.shape[-1],))
            vis = vis.squeeze(1)
            z = z + self.vis_proj(vis)
        return z

    def forward(self,img, pcd, batch, edge_index, edge_weights, vis=None):
        z_img = self.img_enc(img)
        z_pcd = self.pcd_enc(pcd,batch,edge_index,edge_weights)

        #print(vis.shape,self.vis_proj(vis).shape)
        if self.use_ss == True:
            z_img = self.cond(z_img,vis)
            z_pcd = self.cond(z_pcd,vis)

        ## image-->image, pcd-->pcd reconstruction
        img_img = self.img_dec(z_img)
        pcd_pcd = self.pcd_dec(z_pcd)

        ## image-->pcd, pcd-->image reconstruction
        img_pcd = self.pcd_dec(z_img)
        pcd_img = self.img_dec(z_pcd)

        out = dict(z_img=z_img,z_pcd=z_pcd,img_img=img_img,img_pcd=img_pcd,pcd_pcd=pcd_pcd,pcd_img=pcd_img)

        return out

###
def chamfer_distance(p: torch.Tensor, q: torch.Tensor):
    """
    p: [N,3], q: [M,3]
    returns scalar CD = mean_{p} min_q ||p-q||^2 + mean_{q} min_p ||q-p||^2
    """
    # [N,M]
    #print(p.dtype, q.dtype, p.shape, q.shape)
    d2 = torch.cdist(p, q, p=2.0) ** 2
    return d2.min(dim=1).values.mean() + d2.min(dim=0).values.mean()

def chamfer_with_color(p, p_rgb, q, q_rgb):
    d = torch.cdist(p, q)  # [N,M]
    nn_q = d.argmin(dim=1)  # for each p, nearest q
    nn_p = d.argmin(dim=0)  # for each q, nearest p

    cd = (d.gather(1, nn_q[:,None]).squeeze(1)**2).mean() + (d.gather(0, nn_p[None,:]).squeeze(0)**2).mean()

    # color consistency on the matched pairs (symmetrized)
    col1 = (p_rgb - q_rgb[nn_q]).abs().mean()
    col2 = (q_rgb - p_rgb[nn_p]).abs().mean()
    return cd, 0.5*(col1 + col2)

def edge_regularizers(x, edge_index, edge_prob, w_sparse=1.0, w_len=1.0):
    src, dst = edge_index
    pos = x[:, :3]
    lengths = (pos[src] - pos[dst]).norm(dim=-1)  # [E]
    L_sparse = edge_prob.mean()
    L_len = (edge_prob * lengths).mean()
    #print(lengths,L_sparse,L_len)
    return w_sparse*L_sparse + w_len*L_len

###
def rwpe_smoothness_profile_normed(
    rwpe: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
    eps: float = 1e-8,
    center: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      per_dim_n: [K]  (E[(rw_i-rw_j)^2] / (E[rw^2] + eps)) per dimension  (dimensionless)
      scalar_n:  []   mean over dims of per_dim_n (dimensionless, O(1))
    """
    if center:
        rwpe = rwpe - rwpe.mean(dim=0, keepdim=True)

    src, dst = edge_index
    diff2 = (rwpe[src] - rwpe[dst]).pow(2)   # [E, K]

    if edge_weight is not None:
        w = edge_weight / (edge_weight.mean() + eps)   # keep scale stable
        diff2 = diff2 * w[:, None]

    per_dim = diff2.mean(dim=0)                          # [K]
    energy = rwpe.pow(2).mean(dim=0) + eps               # [K]
    per_dim_n = per_dim / energy                         # [K] dimensionless
    scalar_n = per_dim_n.mean()                          # []  (not sum!)

    return per_dim_n, scalar_n


def rwpe_graph_compare_loss(
    pred_rwpe: torch.Tensor, pred_edge_index: torch.Tensor,
    gt_rwpe: torch.Tensor, gt_edge_index: torch.Tensor,
    pred_edge_weight: torch.Tensor | None = None,
    gt_edge_weight: torch.Tensor | None = None,
    mode: str = "mse",   # "mse", "rel", or "log"
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compare normalized smoothness profiles (permutation-invariant, light).
    Outputs a scalar with stable scale across graphs / RWPE magnitudes.
    """
    p_dim, p_s = rwpe_smoothness_profile_normed(pred_rwpe, pred_edge_index, pred_edge_weight, eps=eps)
    g_dim, g_s = rwpe_smoothness_profile_normed(gt_rwpe, gt_edge_index, gt_edge_weight, eps=eps)

    if mode == "mse":
        return F.mse_loss(p_dim, g_dim)  # already dimensionless and ~O(1)
    elif mode == "rel":
        # relative squared error: (p-g)^2 / (g^2 + eps)
        return (((p_dim - g_dim) ** 2) / (g_dim**2 + eps)).mean()
    elif mode == "log":
        # compare in log-space to compress scale
        return F.mse_loss(torch.log(p_dim + eps), torch.log(g_dim + eps))
    else:
        raise ValueError(f"Unknown mode={mode}")
###

def pcd_losses(pred_graphs, gt_graphs, w_cd=0.8, w_rgb=0.2, w_edge_reg=1e-3, w_pe=1e-5):
    """
    pred_graphs: list[Data] with fields x, edge_index, edge_prob
    gt_graphs: list[Data] with field x (at least)
    """
    L_cd = 0.0
    L_rgb = 0.0
    L_edge = 0.0
    L_pe = 0.0
    B = len(pred_graphs)
    #for pg, gg in zip(pred_graphs, gt_graphs): #use only if batched!
    pg,gg = pred_graphs, gt_graphs
    px = pg.x
    gx = gg.x
    ppos = px[:, :3]
    gpos = gx[:, :3]

    if hasattr(pg, "edge_weight") and pg.edge_weight is not None:
        pg.edge_weight = pg.edge_weight.detach()
    pg.edge_index = pg.edge_index.detach()

    ###
    #pe_lap = T.AddLaplacianEigenvectorPE(6,'laplacian',is_undirected=True)
    #pe_rw = T.AddRandomWalkPE(4,'randomwalk')
    #pred_graphs = pe_lap(pred_graphs).to(device)
    #pred_graphs = pe_rw(pred_graphs).to(device)
    #gt_graphs = pe_lap(gt_graphs).to(device)
    #gt_graphs = pe_rw(gt_graphs).to(device)
    #pe_pred = torch.cat([pred_graphs.laplacian,pred_graphs.randomwalk],dim=-1).to(device)
    #pe_gt = torch.cat([gt_graphs.laplacian,gt_graphs.randomwalk],dim=-1).to(device)
    #cd_pe = chamfer_distance(pred_graphs.randomwalk,gt_graphs.randomwalk)
    #cd_pe = L_pe = rwpe_graph_compare_loss(pred_graphs.randomwalk, pg.edge_index,
    #                           gt_graphs.randomwalk, gg.edge_index)
    #L_pe = L_pe + cd_pe
    ###

    cd = chamfer_distance(ppos, gpos)
    L_cd = L_cd + cd

    prgb = px[:, 3:6]
    grgb = gx[:, 3:6]
    # simple NN color via chamfer matching
    cd2, col = chamfer_with_color(ppos, prgb, gpos, grgb)
    # replace cd with cd2 if you want; or just use col term
    L_rgb = L_rgb + col

    # edge regularization if no GT edges
    #if hasattr(pg, "edge_prob") and pg.edge_prob is not None:
    #    L_edge = L_edge + edge_regularizers(px, pg.edge_index, pg.edge_prob,
    #                                        w_sparse=1.0, w_len=1.0)

    L_cd = L_cd / B
    L_rgb = L_rgb / B
    L_edge = L_edge / B
    L_pe = L_pe / B
    return w_cd*L_cd + w_rgb*L_rgb + w_edge_reg*L_edge + w_pe*L_pe


def losses(img_img,img_pcd,img,pcd_img,pcd_pcd,pcd,edge_index,edge_weights,z_img,z_pcd):
    #L_img_img = (img_img-img).abs().mean()
    #L_pcd_img = (pcd_img-img).abs().mean()
    L_img_img = F.l1_loss(img_img,img)
    L_pcd_img = F.l1_loss(pcd_img,img)

    pcd_data = Data()
    pcd_data.x = pcd
    pcd_data.edge_index = edge_index
    pcd_data.edge_attr = edge_weights
    #print(pcd_pcd,pcd_data)
    L_pcd_pcd = pcd_losses(pcd_pcd,pcd_data)
    L_img_pcd = pcd_losses(img_pcd,pcd_data)

    L_align = F.mse_loss(z_img,z_pcd)

    loss = (L_img_img+L_pcd_img)+(L_pcd_pcd+L_img_pcd)+0.2*L_align
    loss_dict = {'img_img':L_img_img.item(),'pcd_img':L_pcd_img.item(),'pcd_pcd':L_pcd_pcd.item(),'img_pcd':L_img_pcd.item(),'latent':L_align.item()}
    return loss, loss_dict

EPOCHS=20000

model = ImageGraphAE(latent_dim=128,use_z_norm=False,use_ss=False).to(device)
opt = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,0,-1)

loader = make_loader('../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/combined_dataset', batch_size=1, shuffle=True, num_workers=4)
#loader = make_loader('../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/single_sample', batch_size=1, shuffle=True, num_workers=4)


def run_epoch(loader):
    model.train(True)
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad(set_to_none=True)
        
        #TODO: define data inputs, write into file
        img_in = batch['img'].permute(0,3,1,2).to(torch.float32)
        #print(img_in.shape,img_in.dtype)
        #R,t -> None
        pcd_in = batch['pcd'].to(torch.float32)
        pcd_in[:,:3] = -pcd_in[:,:3]
        edge_index = batch['edge_index']
        edge_weights = batch['edge_weights'].to(torch.float32)
        vis_in = batch['vis']
        #print('loaded dataset')
        out = model(img_in,pcd_in,batch.batch,edge_index,edge_weights,vis_in)
        print('forward pass')
        #out = dict(z_img=z_img,z_pcd=z_pcd,img_img=img_img,img_pcd=img_pcd,pcd_pcd=pcd_pcd,pcd_img=pcd_img)
        #losses(img_img,img_pcd,img,pcd_img,pcd_pcd,pcd,z_img,z_pcd)
        loss, loss_dict = losses(out['img_img'],out['img_pcd'],img_in,out['pcd_img'],out['pcd_pcd'],pcd_in,edge_index,edge_weights,out['z_img'],out['z_pcd'])
        loss.backward()
        opt.step()
    return loss, loss_dict

print('starting training')
print(alias)
os.makedirs(f"../../../scratch/btuncay/cog/gnn_spatial_reasoning/model/{alias}",exist_ok=True)
csv_path = f"../../../scratch/btuncay/cog/gnn_spatial_reasoning/model/{alias}/losses.csv"
write_header = not os.path.exists(csv_path)

loss_records = []
for epoch in range(EPOCHS):
    write_header = False

    loss, loss_dict = run_epoch(loader)
    row = {"epoch": epoch,**loss_dict}
    pd.DataFrame([row]).to_csv(csv_path,mode="a",header=write_header,index=False)
    loss_records.append({"epoch": epoch, "train_loss": loss_dict})
    if epoch % 10 == 0:
        print(f'epoch {epoch}',loss_dict)
    if epoch % 200 == 0:
        torch.save(model.state_dict(), f"../../../scratch/btuncay/cog/gnn_spatial_reasoning/model/{alias}/model_weights_{epoch}.pt")
torch.save(model.state_dict(), f"../../../scratch/btuncay/cog/gnn_spatial_reasoning/model/{alias}/model_weights.pt")
print('completed')

#pd.DataFrame(loss_records).to_csv(f"../../../scratch/btuncay/cog/gnn_spatial_reasoning/model/{alias}/losses.csv", index=False)

##TODO
# fix pcd decoder => right shape, wrong colors
# pcd decoder has no color loss
# agent sees nothing, is slow
# pcd extractor extracts wrong views
# save losses at each step
# KL loss for latent

##future
# add BCELoss to reconstructed pcd edges -> not applicable without probabilities