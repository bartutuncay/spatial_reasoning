import torch
from spatial_reasoning.models.autoencoder_img_enc import ImageEncoder
from spatial_reasoning.models.autoencoder_pcd_enc import PCDEncoder
from spatial_reasoning.models.autoencoder_combined_dec import ImageDecoder, PCDDecoder
from spatial_reasoning.preprocessing_src.dataloader_autoencoder import make_loader
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
class ImageGraphVAE(nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        basedim = 32
        self.img_enc = ImageEncoder(latent_dim=latent_dim, base_dim=basedim, token_dim=256)
        self.img_dec = ImageDecoder(latent_dim=latent_dim, base_dim=basedim, out_hw=(192,256), norm='group', groups=16, out_act='sigmoid')
        self.pcd_enc = PCDEncoder(latent_dim=latent_dim,layers_points=2,layers_camera=4,layers_mlp=2,z_dim=latent_dim)
        self.pcd_dec = PCDDecoder(nodes_dim=7,layers_mlp=2,latent_dim=latent_dim,nodes_k=3,layers_points=2,layers_camera=4)
        return

    def forward(self,img, pcd, batch, ei_points, ew_points, ei_camera, ea_camera):
        #t1 = time.time()
        z_img, mu_img, logvar_img = self.img_enc(img)
        #t2 = time.time()
        z_pcd, mu_pcd, logvar_pcd = self.pcd_enc(pcd,batch,ei_points,ew_points,ei_camera,ea_camera)

        out = dict(z_img=z_img,z_pcd=z_pcd,mu_pcd=mu_pcd,logvar_pcd=logvar_pcd,mu_img=mu_img,logvar_img=logvar_img)

        return out

## Generate Results - Image Only - from Random Walk
loader = make_loader('datasets_processed/anlieferung/random_walks', batch_size=1, shuffle=False, num_workers=4)
rw_list = natsorted(glob.glob('datasets_processed/anlieferung/random_walks/*.pt'))

alias = 'model_name'
model_weights = f'1_model/{alias}/model_weights_best.pt'
state = torch.load(model_weights, map_location=device)
model = ImageGraphVAE(latent_dim=256).to(device)
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
    ei_points = batch['edge_index']
    ew_points = batch['edge_weights'].to(torch.float32)
    ew_points = torch.exp(-ew_points**2/(2*ew_points.mean()**2))
    ei_camera = batch['ei_camera']
    ea_camera = batch['ea_camera'].to(torch.float32)
    depth_img = batch['depth'].to(torch.float32)
    #print(batch)
    #print('loaded dataset')
    loc, viewdir = torch.tensor(data['loc']), torch.tensor(data['view_dir'])
    with torch.no_grad():
        pred = model(img_in,pcd_in,batch.batch,ei_points,ew_points,ei_camera,ea_camera)
        z_img = {'latent':pred['z_img'],'loc':loc,'viewdir':viewdir}
        z_pcd = {'latent':pred['z_pcd'],'loc':loc,'viewdir':viewdir}
        torch.save(z_img,f'datasets_processed/anlieferung/rw_latents/z_img/{name}')
        torch.save(z_pcd,f'datasets_processed/anlieferung/rw_latents/z_pcd/{name}')