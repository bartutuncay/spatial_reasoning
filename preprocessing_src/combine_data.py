import torch
from typing import Dict, List
from torch.utils.data import Dataset
import numpy as np
import cv2
import pandas as pd
import pathlib
import os
from typing import Dict
import torch

images_path = '../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/delivery_area/dslr_calibration_undistorted/images_parsed.csv'


def generate_train_file(filename, out_path:str) -> None:

    ##image
    img_path = f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/delivery_area/images/dslr_images_undistorted/{filename}.JPG'
    img_in = cv2.imread(img_path)
    img_cropped = cv2.resize(img_in,(512,384)) # cropped image shape: 512,384,3
    img = torch.from_numpy(img_cropped/255).to(torch.float32)

    ##image csv
    parsed_csv = pd.read_csv('../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/delivery_area/dslr_calibration_undistorted/images_parsed.csv')
    query_name = f'dslr_images_undistorted/{filename}.JPG'

    ##image R
    #vals_R = parsed_csv.loc[parsed_csv['image_name']==query_name,['qw','qx','qy','qz']]
    #R = torch.tensor(vals_R.iloc[0].to_numpy(dtype=float), dtype=torch.float32) # rotation matrix shape: 4,

    ##image t
    #vals_t = parsed_csv.loc[parsed_csv['image_name']==query_name,['tx','ty','tz']]
    #t = torch.tensor(vals_t.iloc[0].to_numpy(dtype=float), dtype=torch.float32) # translation matrix shape: 3,

    ##point cloud graph
    pcd_graph = torch.load(f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/pcd_graph/{filename}.pt',weights_only=False)
    pcd = torch.cat([pcd_graph.pos,pcd_graph.rgb],dim=-1) # point cloud shape N,6
    pcd_ei = pcd_graph.edge_index # point cloud graph edges
    pcd_ew = pcd_graph.edge_attr # point cloud graph edge weights

    ##visibility metrics
    # 2D: area, mean distance, max distance, min distance, area stdev
    # 3D: volume, mean depth, max depth, min depth, volume stdev
    stats_2d_csv = pd.read_csv('../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/vis_2d_camera.csv')
    stats_3d_csv = pd.read_csv(f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/vis_3d_camera_{int(filename.split("_")[1])}.csv')
    vals_2d = stats_2d_csv.loc[stats_2d_csv['img']==filename,['area_calc','mean_dist','max_dist','min_dist','dist_std']]
    vals_3d = stats_3d_csv.loc[stats_3d_csv['img']==filename,['vol_calc','mean_dist','max_dist','min_dist','dist_std']]
    a = torch.tensor(vals_2d.to_numpy(dtype=float), dtype=torch.float32)
    b = torch.tensor(vals_3d.to_numpy(dtype=float), dtype=torch.float32)
    vis = torch.cat([a, b], dim=1)


    packed = {
        "img": img,               # [M,H,W,3] uint8
        "pcd": pcd,                     # [sumN,6] float32
        "edge_index": pcd_ei,       # [2,sumE] long (global)
        "edge_weights": pcd_ew,   # [sumE] float32
        "vis": vis,                     # [M,10]
    }

    torch.save(packed, out_path)
    return

images_csv = pd.read_csv(images_path)

for _, file in images_csv.iterrows():
    filename = file['image_name'].split('/')[1].split('.JPG')[0]
    generate_train_file(filename,f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/combined_dataset/{filename}.pt')

print('completed')