import torch
import glob
import pandas as pd
from scipy.spatial import KDTree

subgraphs_dir = '../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/scan_pcd_graph/clusters/*.pt'
subgraphs = sorted(glob.glob(subgraphs_dir))

#print(subgraphs)

mean_positions = []
for graph in subgraphs:
    data = torch.load(graph)
    data_vals = data.pos.mean(dim=0).detach()
    mean_positions.append(data_vals.tolist())
    
mean_df = pd.DataFrame(mean_positions)

tree = KDTree(mean_positions)
_, neighbors = tree.query(mean_positions,k=len(mean_positions))
print(neighbors.shape)
neighbors_df = pd.DataFrame(neighbors[:,1:])

mean_df.to_csv('../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/scan_pcd_graph/clusters/lookup.csv')
neighbors_df.to_csv('../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/scan_pcd_graph/clusters/neighbors.csv')