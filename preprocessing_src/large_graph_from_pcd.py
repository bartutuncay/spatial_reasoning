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

device = torch.device('cpu')
pcd = o3d.io.read_point_cloud('../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/delivery_area/scan_raw/combined_aligned.ply')
pcd_downsampled = pcd.voxel_down_sample(0.15)
print(len(pcd.points),len(pcd_downsampled.points))
pcd_coords = torch.tensor(pcd_downsampled.points)
pcd_rgb = torch.tensor(pcd_downsampled.colors,dtype=torch.float32)
#ei,ew = knn_connectivity(pcd_coords,3)
ei = radius_graph(pcd_coords,r=1)
row, col = ei
ew = torch.norm(pcd_coords[row] - pcd_coords[col], dim=1)
data = make_graph(pcd_coords,pcd_rgb,ei,ew)

torch.save(data,'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/scan_pcd_graph/combined_aligned.pt')

## Cluster the graph
N = data.num_nodes
part_size = 1200
num_parts = math.ceil(N / part_size)
cluster_data = ClusterData(data, num_parts=num_parts, recursive=False)
cluster_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True)

for i, part in enumerate(cluster_loader):
    torch.save(part,f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/scan_pcd_graph/clusters/{i}.pt')