## Helper functions for random walk data processing
# kNN graph generator
# Rotation matrix solver

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