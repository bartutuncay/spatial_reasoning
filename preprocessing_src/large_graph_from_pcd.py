## Helper function - graph from point clouds
# Converts point cloud inputs into data graphs usable by PyG
# Required for training and inference tasks of locator model

import math
import torch
import numpy as np
from scipy.spatial import KDTree, Delaunay
import open3d as o3d
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import to_undirected 
from torch_geometric.loader import ClusterData, ClusterLoader

def make_graph(pcd_points,pcd_colors,ei,ew):
    data = Data()
    data.pos = pcd_points
    data.rgb = pcd_colors
    data.x = torch.cat([pcd_points,pcd_coords],dim=1)
    #data.edge_index = ei
    #data.edge_weights = ew

    edge_index,ew = to_undirected(ei,edge_attr=ew)
    #print('Undirected?' if is_undirected(edge_index) else 'Directed!')
    data.edge_attr = ew
    data.edge_index = edge_index
    return data

scene =  'test_terrace/terrace'# 'pipes/pipes', 'relief/relief', 'hospital/.', 'break_room/kicker', 'anlieferung/delivery_area'

device = torch.device('cpu')
pcd = o3d.io.read_point_cloud(f'datasets_processed/{scene}/scan_raw/combined_aligned.ply')
#pcd = o3d.io.read_point_cloud(f'area_calc/Hospital2 1.pts')
#pcd_downsampled = pcd.voxel_down_sample(0.15)
pcd_downsampled = pcd.voxel_down_sample(0.05)
print(len(pcd.points),len(pcd_downsampled.points))
pcd_coords = torch.tensor(pcd_downsampled.points)
pcd_rgb = torch.tensor(pcd_downsampled.colors,dtype=torch.float32)
#ei,ew = knn_connectivity(pcd_coords,3)
#ei = radius_graph(pcd_coords,r=1)
ei = knn_graph(pcd_coords,k=3)
row, col = ei
ew = torch.norm(pcd_coords[row] - pcd_coords[col], dim=1)
data = make_graph(pcd_coords,pcd_rgb,ei,ew)

torch.save(data,f'datasets_processed/{scene.split('/')[0]}/scan_pcd_graph/combined_aligned.pt')

## Cluster the graph
N = data.num_nodes
part_size = 48000#1200
num_parts = math.ceil(N / part_size)
cluster_data = ClusterData(data, num_parts=num_parts, recursive=False)
cluster_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True)

for i, part in enumerate(cluster_loader):
    torch.save(part,f'datasets_processed/{scene.split('/')[0]}/scan_pcd_graph/clusters/{i}.pt')