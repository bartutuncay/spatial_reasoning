## Extract point clouds for the test dataset
# Note: camera intrinsics are set according to the available ETH3D data,
# this may require modifications

import numpy as np
import pandas as pd
from volume_methods import directional_raycast
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from torch_geometric.utils import is_undirected, to_undirected
import torch
import time
from scipy.spatial import KDTree
from torch_geometric.data import Data
import torch_geometric.typing as pyg_typing
pyg_typing.WITH_INDEX_SORT = False

## get slices of visible point cloud from each image

pcd_path = 'datasets_processed/anlieferung/delivery_area/scan_raw/combined_aligned.ply'
#pcd_path = '../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/delivery_area/scan_raw/scan2_no_camera.ply'
images_path = 'datasets_processed/anlieferung/delivery_area/dslr_calibration_undistorted/images_parsed.csv'
images_df = pd.read_csv(images_path)
pcd = o3d.io.read_point_cloud(pcd_path)
pcd_points = np.asarray(pcd.points)
pcd_colors = np.asarray(pcd.colors)

camera_intrinsics = {'W':6208,'H':4135,'fx':3408.59,'fy':3408.87,'cx':3117.24,'cy':2064.07}
fov_x = 2*np.arctan(camera_intrinsics['W']/(2*camera_intrinsics['fx']))
fov_y = 2*np.arctan(camera_intrinsics['H']/(2*camera_intrinsics['fy']))

def knn_connectivity(pcd,k):
    tree = KDTree(pcd)
    num_nodes = pcd.shape[0]
    dists, local_conn = tree.query(pcd,k=k+1) # first item is self-loop
    src = np.repeat(np.arange(num_nodes, dtype=np.int64), k)
    neighbors = local_conn[:,1:].reshape(-1)
    pairs = np.stack([src, neighbors], axis=1)
    pairs = np.sort(pairs, axis=1)
    pairs = np.unique(pairs, axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    weights = np.linalg.norm(pcd[pairs[:,0]]-pcd[pairs[:,1]],axis=1)
    #print(pairs,pairs.shape,weights,weights.shape)
    return pairs.T, weights

def make_graph(pcd_points,pcd_colors,ei,ew):
    data = Data()
    data.pos = torch.from_numpy(pcd_points)
    data.rgb = torch.from_numpy(pcd_colors)
    #data.edge_index = ei
    #data.edge_weights = ew
    ei = torch.from_numpy(ei)
    ew = torch.from_numpy(ew)

    edge_index,ew = to_undirected(ei,edge_attr=ew)
    #print('Undirected?' if is_undirected(edge_index) else 'Directed!')
    data.edge_attr = ew
    data.edge_index = edge_index
    return data

ndirs = 120000 #total number of directions for raycasting

for _, point in images_df.iterrows():
    t1 = time.time()
    img_name = str(point.loc['image_name']).split('.JPG')[0].split('/')[1]
    print(img_name)
    
    qx,qy,qz,qw = point.loc[['qx','qy','qz','qw']].to_numpy()
    R_wc = R.from_quat([qx, qy, qz, qw]).as_matrix()
    #translation from quaternion, translation matrices to camera coords
    tx,ty,tz = point.loc[['tx','tz','ty']].to_numpy()
    t_wc = np.array([tx, ty, tz])
    # camera center in world coordinates
    C = -R_wc.T @ t_wc
    d_cam = np.array([0.0, 0.0, 1.0])
    # camera view direction (center)
    d_world = R_wc.T @ d_cam
    d_world = d_world/(np.linalg.norm(d_world))
    t2 = time.time()
    print(t2-t1)
    pcd_visible, pcd_rgb, vol, mindist, maxdist, meandist, stdev_dist = directional_raycast([0,40],ndirs,d_world,fov_x,fov_y,pcd_points,pcd_colors,t_wc,[0,20],False)
    t3 = time.time()
    print(t3-t2)
    ei,ew = knn_connectivity(pcd_visible,3)
    t4 = time.time()
    print(t4-t3)
    data = make_graph(pcd_visible,pcd_rgb,ei,ew)
    t5 = time.time()
    print(t5-t4)

    row = pd.DataFrame([{'dirs': ndirs, 'max_distance' : 40, 'img': img_name, 'vol_calc': vol, 'mean_dist':meandist,
            'dist_std':stdev_dist,'min_dist':mindist,'max_dist':maxdist,'posX': C[0], 'posY': C[1]}])
    #row.to_csv(results_csv,mode='a',header=False,index=False)
    #torch.save(data,f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/pcd_graph/{img_name}.pt')
    #print(data)
    #print(pcd_visible)
