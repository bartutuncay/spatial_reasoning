import numpy as np
import pandas as pd
from volume_methods import directional_raycast
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from torch_geometric.utils import is_undirected, to_undirected
import torch
from scipy.spatial import KDTree
import argparse
from torch_geometric.data import Data
import torch_geometric.typing as pyg_typing
pyg_typing.WITH_INDEX_SORT = False

## get image index from slurm
parser = argparse.ArgumentParser()
parser.add_argument("--task-index",type=int,
                required=True,help="Row index from CSV")
args = parser.parse_args()
process_idx = args.task_index

## get slices of visible point cloud from each image

pcd_path = '../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/delivery_area/scan_raw/combined_aligned.ply'
#pcd_path = '../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/delivery_area/scan_raw/scan2_no_camera.ply'
images_path = '../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/delivery_area/dslr_calibration_undistorted/images_parsed.csv'
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

def rotation_a_to_b(a, b, eps=1e-12):
    """
    Returns R (3x3) such that R @ a == b (approximately),
    where a,b are 3D vectors (need not be unit).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / (np.linalg.norm(a) + eps)
    b = b / (np.linalg.norm(b) + eps)

    v = np.cross(a, b)
    c = float(np.dot(a, b))          # cos(theta)
    s = np.linalg.norm(v)            # sin(theta)

    # If a and b are (anti)parallel:
    if s < eps:
        if c > 0.0:
            return np.eye(3)         # already aligned
        # 180° rotation: pick any axis orthogonal to a
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        u = np.cross(a, axis)
        u = u / (np.linalg.norm(u) + eps)
        # Rodrigues for theta=pi: R = -I + 2 u u^T
        return -np.eye(3) + 2.0 * np.outer(u, u)

    # Rodrigues' rotation formula
    k = v / s
    K = np.array([[0.0,   -k[2],  k[1]],
                  [k[2],   0.0,  -k[0]],
                  [-k[1],  k[0],  0.0]])
    R = np.eye(3) + K * s + (K @ K) * (1.0 - c)
    return R

ndirs = 120000

results_csv = f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/vis_3d_camera_{process_idx}.csv'
res_df = pd.DataFrame(columns=['dirs','max_distance','img','vol_calc','mean_dist','dist_std','min_dist','max_dist',
                            'posX','posY'])
res_df.to_csv(results_csv,index=False)


for _, point in images_df.iterrows():
    img_name = str(point.loc['image_name']).split('.JPG')[0].split('/')[1]
    print(img_name,f'DSC_{process_idx}')
    if img_name != f'DSC_0{process_idx}':
        continue
    
    qx,qy,qz,qw = point.loc[['qx','qy','qz','qw']].to_numpy()
    R_wc = R.from_quat([qx, qy, qz, qw]).as_matrix()
    #translation from quaternion, translation matrices to camera coords
    tx,ty,tz = point.loc[['tx','ty','tz']].to_numpy()
    t_wc = np.array([tx, ty, tz])
    # camera center in world coordinates
    C = -R_wc.T @ t_wc
    #print(t_wc,C)
    d_cam = np.array([0.0, 0.0, 1.0])
    # camera view direction (center)
    d_world = R_wc.T @ d_cam
    d_world = d_world/(np.linalg.norm(d_world))
    pcd_visible, pcd_rgb, vol, mindist, maxdist, meandist, stdev_dist = directional_raycast([0,40],ndirs,d_world,fov_x,fov_y,pcd_points,pcd_colors,C,[0,20],False)
    e = np.array([1.0, 0.0, 0.0])  # target view direction
    R_align = rotation_a_to_b(d_world, e)
    pcd_visible_aligned = (R_align @ (pcd_visible - C).T).T
    ei,ew = knn_connectivity(pcd_visible_aligned,3)
    data = make_graph(pcd_visible_aligned,pcd_rgb,ei,ew)
    
    row = pd.DataFrame([{'dirs': ndirs, 'max_distance' : 20, 'img': img_name, 'vol_calc': vol, 'mean_dist':meandist,
            'dist_std':stdev_dist,'min_dist':mindist,'max_dist':maxdist,'posX': C[0], 'posY': C[1]}])
    row.to_csv(results_csv,mode='a',header=False,index=False)
    torch.save(data,f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/pcd_graph/{img_name}.pt')
    #break
    #print(data)
    #print(pcd_visible)
