import torch
from torch.nn import ModuleList
import numpy as np
import torch.nn as nn
from scipy.spatial import cKDTree as KDTree
import os
import glob
import pandas as pd
import math
from typing import Optional, Tuple, Dict
import torch.nn.functional as F

device = torch.device('cuda')
torch.set_default_dtype(torch.float32)

## obtain relative translation between adjacent latent point pairs
# point pairs max. 2-3 hops apart
# convert translations to equivariant frame
# decode edge weights with MLP
# losses: translation, rotation

class RelTrLoc(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.translation_projection = nn.Sequential(
            nn.Linear(in_channels,in_channels),
            #nn.ReLU(),
            nn.Tanh(),
            nn.Linear(in_channels,in_channels),
            #nn.ReLU(),
            nn.Tanh(),
            nn.Linear(in_channels,out_channels))

    def forward(self,x1,x2):
        #inp = torch.cat([x1, x2, x2 - x1], dim=-1)
        inp = x2-x1
        transform = self.translation_projection(inp)
        return transform

def equivariant_translation(tx1, ty1, tz1, tx2, ty2, tz2, u1, v1, w1, u2, v2, w2, eps=1e-9):
    # World displacement from cam1 to cam2
    dp = np.array([tx2 - tx1, ty2 - ty1, tz2 - tz1], dtype=float)

    # Cam1 forward (+z) in world
    z = np.array([u1, v1, w1], dtype=float)
    z_norm = np.linalg.norm(z)
    z /= z_norm

    # World up
    uw = np.array([0.0, 1.0, 0.0], dtype=float)

    # Right (+x) = normalize(uw x z); fallback if degenerate
    x = np.cross(uw, z)
    x_norm = np.linalg.norm(x)
    if x_norm < eps:
        # forward is ~parallel to world-up; choose a fallback up
        uw_fallback = np.array([0.0, 0.9, 0.0], dtype=float)
        x = np.cross(uw_fallback, z)
        x_norm = np.linalg.norm(x)
        if x_norm < eps:
            raise ValueError("Degenerate orientation: cannot construct basis.")
    x /= x_norm

    # Up (+y) = z x x  (ensures right-handed basis)
    y = np.cross(z, x)

    # Rotation camera --> world, columns are camera axes in world
    R_cw = np.column_stack((x, y, z))

    # displacement in camera 1 coordinates: t = R_wc * dp = R_cw.T * dp
    t_c1 = R_cw.T @ dp
    t_c1 = torch.tensor(t_c1)
    rot1 = torch.tensor([u2-u1,v2-v1,w2-w1])
    return t_c1, rot1

# pairs already saved in rw_latents
def train_loader():
    rw = torch.randint(low=1,high=5,size=(1,))
    data = torch.load(f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/rw_latents/z_img_rw_{rw.item()}.pt')
    pairs = data.edge_index.T
    idx1, idx2 = pairs[torch.randint(low=0,high=pairs.shape[0],size=(1,)).item()]
    p1 = torch.load(f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/rw_latents/z_img/rw_{rw.item()}_{idx1}.pt')
    p2 = torch.load(f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/rw_latents/z_img/rw_{rw.item()}_{idx2}.pt')
    
    return p1['latent'], p2['latent'], p1['loc'], p2['loc'], p1['viewdir'], p2['viewdir']


def calculate_losses(t1,rot,pred):
    # t1 = translations in camera frame [3,]
    # rot = relative rotation (SO(3))   [3,]
    # pred = pred(t1),pred(rot)         [6,]
    #tr_loss = F.mse_loss(pred[:3],t1)
    tr_loss = F.smooth_l1_loss(pred[:3],t1)
    rot_loss = F.mse_loss(pred[3:],rot)
    loss = tr_loss + 0.01*rot_loss
    loss_dict = {'total':loss.item(),'translation':tr_loss.item(),'rotation':rot_loss.item()}
    return loss, loss_dict

EPOCHS=10000
model = RelTrLoc(in_channels=128,out_channels=6).to(device)
opt = torch.optim.Adam(model.parameters(),lr=2e-5,weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,0,-1)
alias = '0222_translation_rotation_tanh'

print('starting training')
print(alias)
os.makedirs(f"../../../scratch/btuncay/cog/gnn_spatial_reasoning/model_movement/{alias}",exist_ok=True)
csv_path = f"../../../scratch/btuncay/cog/gnn_spatial_reasoning/model_movement/{alias}/losses.csv"
write_header = not os.path.exists(csv_path)

def run_epoch():
    model.train(True)
    opt.zero_grad(set_to_none=True)
    for _ in range(200):
        l1,l2,p1loc,p2loc,p1dir,p2dir = train_loader()
        tx1,ty1,tz1 = p1loc
        tx2,ty2,tz2 = p2loc
        u1,v1,w1 = p1dir
        u2,v2,w2 = p2dir
        t_c1,rot = equivariant_translation(tx1,ty1,tz1,tx2,ty2,tz2,u1,v1,w1,u2,v2,w2)
        t_c1 = t_c1.to(torch.float32).to(device)
        rot = rot.to(torch.float32).to(device)
        l1 = l1.to(device)
        l2 = l2.to(device)
        pred = model(l1,l2)
        #print(t_c1.shape,rot.shape,pred.shape)
        loss, ld = calculate_losses(t_c1,rot,pred[0])
        loss.backward()
        opt.step()
    return loss, ld


loss_records = []
for epoch in range(EPOCHS):
    write_header = False

    loss, loss_dict = run_epoch()
    row = {"epoch": epoch,**loss_dict}
    pd.DataFrame([row]).to_csv(csv_path,mode="a",header=write_header,index=False)
    loss_records.append({"epoch": epoch, "train_loss": loss_dict})
    if epoch % 10 == 0:
        print(f'epoch {epoch}',loss_dict)
    if epoch % 200 == 0:
        torch.save(model.state_dict(), f"../../../scratch/btuncay/cog/gnn_spatial_reasoning/model_movement/{alias}/model_weights_{epoch}.pt")
torch.save(model.state_dict(), f"../../../scratch/btuncay/cog/gnn_spatial_reasoning/model_movement/{alias}/model_weights.pt")
print('completed')