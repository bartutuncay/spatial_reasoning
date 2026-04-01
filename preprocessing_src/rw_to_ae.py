import torch
from gnn_spatial_reasoning.models.image_encoder_no_cond import ImageEncoder
from gnn_spatial_reasoning.models.image_decoder import ImageDecoder
from gnn_spatial_reasoning.models.pcd_encoder import PCDEncoder
from gnn_spatial_reasoning.models.pcd_decoder_edges import PCDDecoder
from gnn_spatial_reasoning.preprocessing_src.dataloader import MultiModalBatch, make_loader, collate_pt_dicts, PtDictFolderDataset
from torch_geometric.data import Data
import torch_geometric.nn.functional as F_geom
from torch_geometric.nn import SimpleConv, MLP, Linear
from torch_geometric.nn import pool
from natsort import natsorted
from torch.nn import ModuleList
import torch.nn as nn
import glob
import pandas as pd
from typing import Optional, Tuple, Dict
import torch.nn.functional as F

device = torch.device('cpu')
torch.set_default_dtype(torch.float32)


## Model Definition
class ImageGraphAE(nn.Module):
    def __init__(self,latent_dim,use_z_norm=False,use_ss=False):
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

## Generate Results - Image Only - from Random Walk
loader = make_loader('../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/random_walks', batch_size=1, shuffle=False, num_workers=4)
rw_list = natsorted(glob.glob('../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/random_walks/*.pt'))

alias = '0210_novis'
model_weights = f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/model/{alias}/model_weights_19800.pt'
state = torch.load(model_weights, map_location=device)
model = ImageGraphAE(latent_dim=128,use_z_norm=False,use_ss=False).to(device)
model.load_state_dict(state)
model.eval()

for batch, name in zip(loader,rw_list):
    batch = batch.to(device)
    data = torch.load(name,weights_only=False)
    name = name.split('random_walks/')[1]
    img_in = batch['img'].permute(0,3,1,2).to(torch.float32)
    #print(img_in.shape,img_in.dtype)
    #R,t -> None
    pcd_in = batch['pcd'].to(torch.float32)
    pcd_in[:,:3] = -pcd_in[:,:3]
    edge_index = batch['edge_index']
    edge_weights = batch['edge_weights'].to(torch.float32)
    vis_in = torch.zeros(5,2)
    loc, viewdir = torch.tensor(data['loc']), torch.tensor(data['view_dir'])
    with torch.no_grad():
        pred = model(img_in,pcd_in,batch.batch,edge_index,edge_weights,vis_in)
        z_img = {'latent':pred['z_img'],'loc':loc,'viewdir':viewdir}
        z_pcd = {'latent':pred['z_pcd'],'loc':loc,'viewdir':viewdir}
        #print(f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/rw_latents/z_img/{name}')
        torch.save(z_img,f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/rw_latents/z_img/{name}')
        torch.save(z_pcd,f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/rw_latents/z_pcd/{name}')