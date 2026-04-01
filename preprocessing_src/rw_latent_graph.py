import torch
from torch_geometric.data import Data
import torch_geometric.nn.functional as F_geom
import torch.nn as nn
import numpy as np
import math
from scipy.spatial import KDTree
from torch_geometric.utils import is_undirected, to_undirected
from typing import Optional, Tuple, Dict
import torch.nn.functional as F
from natsort import natsorted
import argparse
import glob
import torch_geometric.typing as pyg_typing
from gnn_spatial_reasoning.preprocessing_src.dataloader import MultiModalBatch, make_loader, PtDictFolderDataset
pyg_typing.WITH_INDEX_SORT = False

device = torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--task-index",type=int,
                required=True,help="Row index from CSV")
args = parser.parse_args()
idx = args.task_index
# load latents from one random walk

walk_list = natsorted(glob.glob(f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/rw_latents/z_img/rw_{idx}*'))
rw_list = natsorted(glob.glob(f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/random_walks/rw_{idx}*.pt'))

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

    #calculate edge attributes with equivariant translation


    #print(pairs,pairs.shape,weights,weights.shape)
    return pairs.T, weights

def make_graph(pcd_points,ei,ew):
    data = Data()
    data.x = pcd_points
    #data.edge_index = ei
    #data.edge_weights = ew
    ei = torch.from_numpy(ei)
    ew = torch.from_numpy(ew)

    edge_index,ew = to_undirected(ei,edge_attr=ew)
    #print('Undirected?' if is_undirected(edge_index) else 'Directed!')
    data.edge_attr = ew
    data.edge_index = edge_index
    return data

locs = []
latents = []

for batch in walk_list:
    batch = torch.load(batch,weights_only=False)
    locs.append(batch['loc'])
    latents.append(batch['latent'][0])
locs_np = np.asarray(locs)
latents_np = np.asarray(latents)
ei, ew = knn_connectivity(locs_np,4)
data = make_graph(latents_np,ei,ew)
print(data)

torch.save(data,f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/rw_latents/z_img_rw_{idx}.pt')