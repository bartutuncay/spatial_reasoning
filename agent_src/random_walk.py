import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import pandas as pd
from torch_geometric.utils import is_undirected, to_undirected
import torch
from scipy.spatial import KDTree
import argparse
from vision import raycast_img
from volume_methods import directional_raycast, directional_raycast_fast
from dataclasses import dataclass
from torch_geometric.data import Data
from torch.utils.data import Dataset
from pcd_slice_methods import knn_connectivity, make_graph, rotation_a_to_b
import torch_geometric.typing as pyg_typing
pyg_typing.WITH_INDEX_SORT = False

## function inputs
# position
# view direction
# point cloud

## outputs
# position
# view direction
# RGB image
# depth image
# point cloud slice
### 2d visibility metrics
### 3d visibility metrics

## list of functions:
# Rect: boundary data class
# in_union_rects: checks if point in boundary
# sample_xy_in_union: random point sampling for starting point
# smooth_heading_walk: generates walking trajectory

## get task index from slurm
parser = argparse.ArgumentParser()
parser.add_argument("--task-index",type=int,
                required=True,help="Row index from CSV")
args = parser.parse_args()
process_idx = args.task_index

np.random.seed(process_idx)
rng = np.random.default_rng(process_idx)

## set data class for navigable area (union of rectangles)
@dataclass(frozen=True)
class Rect:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def area(self) -> float:
        return max(0.0, self.xmax - self.xmin) * max(0.0, self.ymax - self.ymin)

def in_union_rects(xy: np.ndarray, rects: list[Rect]) -> np.ndarray:
    """
    xy: (N,2)
    returns mask: (N,)
    """
    x = xy[:, 0]
    y = xy[:, 1]
    mask = np.zeros(len(xy), dtype=bool)
    for r in rects:
        mask |= (x >= r.xmin) & (x <= r.xmax) & (y >= r.ymin) & (y <= r.ymax)
    return mask

def sample_xy_in_union(rects: list[Rect], n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Returns (n,2) points uniformly over the *mixture* of rectangles by area.
    NOTE: If rectangles overlap, the overlapping area will be oversampled.
    """
    areas = np.array([r.area for r in rects], dtype=float)
    if areas.sum() <= 0:
        raise ValueError("Total area is zero; check rectangle bounds.")
    probs = areas / areas.sum()

    # choose which rect each point comes from
    choices = rng.choice(len(rects), size=n, p=probs)

    pts = np.zeros((n, 2), dtype=float)
    for i, ridx in enumerate(choices):
        r = rects[ridx]
        pts[i, 0] = rng.uniform(r.xmin, r.xmax)
        pts[i, 1] = rng.uniform(r.ymin, r.ymax)
    return pts

def smooth_heading_walk(start_xy, rects, n_steps=300,
    step=0.1, sigma_theta=0.15, sigma_step=0.02,
    # view-direction params
    pitch=-0.1,          # radians; negative looks slightly down (Z up)
    sigma_view=0.07,      # view yaw noise (smaller = steadier)
    view_pull=0.1,        # how strongly view yaw is pulled toward motion theta each step
    rng=None
    ):
    """
    Returns:
      path_xy:  (n_steps, 2)
      view_dir: (n_steps, 3) unit vectors in world coords (Z up)
      view_yaw: (n_steps,)   yaw angles (radians)
    """
    rng = np.random.default_rng() if rng is None else rng
    xy = np.array(start_xy, dtype=float)

    def inside(p):
        x, y = p
        for r in rects:
            if r.xmin <= x <= r.xmax and r.ymin <= y <= r.ymax:
                return True
        return False
    if not inside(xy):
        raise ValueError("start_xy must be inside the union.")

    theta = rng.uniform(0, 2*np.pi)   # motion yaw
    view_yaw = theta                  # camera/view yaw (Markov state)
    path_xy  = np.zeros((n_steps, 2), float)
    view_dir = np.zeros((n_steps, 3), float)
    view_yaw_hist = np.zeros((n_steps,), float)
    cp, sp = np.cos(pitch), np.sin(pitch)

    for i in range(n_steps):
        # --- motion update (persistent heading) ---
        theta = theta + rng.normal(scale=sigma_theta)
        L = max(0.0, step + rng.normal(scale=sigma_step))
        proposal = xy + L * np.array([np.cos(theta), np.sin(theta)])

        if inside(proposal):
            xy = proposal
        else:
            # cheap bounce: reverse direction and try once more
            theta = theta + np.pi + rng.normal(scale=0.05)
            proposal = xy + L * np.array([np.cos(theta), np.sin(theta)])
            if inside(proposal):
                xy = proposal

        # --- view direction update (Markov, smooth) ---
        # Pull the view yaw toward the motion direction, plus small noise.
        # view_pull in [0,1]; smaller = more inertia/smoothing.
        view_yaw = (1.0 - view_pull) * view_yaw + view_pull * theta + rng.normal(scale=sigma_view)

        cy, sy = np.cos(view_yaw), np.sin(view_yaw)
        d = np.array([cp * cy, cp * sy, sp], dtype=float)
        d /= (np.linalg.norm(d) + 1e-12)
        path_xy[i] = xy
        view_dir[i] = d
        view_yaw_hist[i] = view_yaw

    return path_xy, view_dir, view_yaw_hist

rects = [Rect(xmin=-47,xmax=5,ymin=-4,ymax=0),
         Rect(xmin=-12,xmax=5,ymin=0,ymax=9),
         Rect(xmin=-11,xmax=-7,ymin=9,ymax=15),
         Rect(xmin=-3,xmax=2,ymin=9,ymax=14),
         Rect(xmin=-1,xmax=2,ymin=14,ymax=18)]

sampled_pts = sample_xy_in_union(rects,10000,rng=rng)
#print(sampled_pts)
#path = random_walk_xy(rng.choice(sampled_pts),rects,100,1,rng=rng)
path,dirs,_ = smooth_heading_walk(rng.choice(sampled_pts),rects,40,0.2,0.15,0.1,rng=rng)
#path = lazy_walk(rng.choice(sampled_pts),rects,100,0.1,0.9,0.05,rng=rng)
print(path,dirs)

pcd_path = '../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/delivery_area/scan_raw/combined_aligned.ply'
images_path = '../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung/delivery_area/dslr_calibration_undistorted/images_parsed.csv'
images_df = pd.read_csv(images_path)
pcd = o3d.io.read_point_cloud(pcd_path)
pcd_points = np.asarray(pcd.points)
pcd_colors = np.asarray(pcd.colors)
camera_intrinsics = {'W':6208,'H':4135,'fx':3408.59,'fy':3408.87,'cx':3117.24,'cy':2064.07}
fov_x = 2*np.arctan(camera_intrinsics['W']/(2*camera_intrinsics['fx']))
fov_y = 2*np.arctan(camera_intrinsics['H']/(2*camera_intrinsics['fy']))


for vantage_idx, point in enumerate(zip(path,dirs)):
    view_idx = f'{process_idx}_{vantage_idx}'
    out_path = f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/anlieferung/random_walks/rw_{process_idx}_{vantage_idx}.pt'
    point, viewdir = point
    tx, ty, tz = point[0], point[1], -3.5
    t_wc = np.array([tx, ty, tz])
    # convert to quaternion
    pcd_visible, pcd_rgb, vol, mindist, maxdist, meandist, stdev_dist = directional_raycast_fast([0,40],120000,viewdir,fov_x,fov_y,pcd_points,pcd_colors,t_wc,[0,20],False)
    view_rgb, view_depth = raycast_img([0,40],viewdir,fov_x,fov_y,pcd_points,pcd_colors*255,t_wc,384,512,(0,0,0))
    e = np.array([1.,0.,0.])
    R_align = rotation_a_to_b(viewdir,e)
    pcd_visible_aligned = (R_align @ (pcd_visible - t_wc).T).T
    ei, ew = knn_connectivity(pcd_visible_aligned,3)
    data_graph = make_graph(pcd_visible_aligned,pcd_rgb,ei,ew)
    img_rgb = torch.from_numpy(view_rgb/255).to(torch.float32)
    img_d = torch.from_numpy(view_depth/255).to(torch.float32)
    pcd_visible_aligned = torch.tensor(pcd_visible_aligned)
    pcd_rgb = torch.tensor(pcd_rgb)
    pcd_tensor = torch.cat([pcd_visible_aligned,pcd_rgb],dim=1)
    #a = torch.tensor([0,0,0,0,0]).to(torch.float32)
    #b = torch.tensor([vol,meandist,maxdist,mindist,stdev_dist])
    #vis = torch.cat([a,b],dim=1)
    packed = {'img': img_rgb, 'edge_index':torch.tensor(ei),
              'edge_weights':torch.tensor(ew), 'pcd':pcd_tensor, 'loc':t_wc, 'view_dir':viewdir}#,'vis':vis}
    torch.save(packed,out_path)
    #break
if False:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    #plt.plot(path[:,0],path[:,1])
    plt.imshow(view_rgb)
    #plt.axis('equal')
    plt.savefig(f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/logs/random_walk/test_{process_idx}_{vantage_idx}.png')