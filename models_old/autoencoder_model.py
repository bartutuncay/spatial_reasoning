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
from models.image_encoder_no_cond import ImageEncoder
from models.image_decoder import ImageDecoder
from models.pcd_encoder import PCDEncoder
from models.pcd_decoder_edges import PCDDecoder
from preprocessing_src.dataloader import MultiModalBatch, make_loader, collate_pt_dicts, PtDictFolderDataset

## Autoencoder outline:
# Inputs:
# - image (2D array)
# - image rotation translation matrices
# - visible points pcd graph
# => collapsed into latent space
# Outputs:
# - image (2D array)
# - visible points pcd graph

# Data format:
# image: [H,W,R,G,B]
# pcd: [x,y,z,R,G,B]

# Comet - logging
#experiment = start(api_key="7VD3oulgQdsrnNz60JDDhY86O",project_name="gnn-spatial-reasoning",workspace="btuncay")
#hyper_params = {'learning_rate': 1e-4,'steps': 10000,'batch_size':1}
#experiment.log_parameters(hyper_params)

device = torch.device('cuda')
torch.set_default_dtype(torch.float32)

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