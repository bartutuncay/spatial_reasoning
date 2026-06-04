import open3d as o3d
import numpy as np
import pandas as pd
import torch
import argparse
from dataclasses import dataclass
from torch_geometric.data import Data
from pcd_slice_methods import knn_connectivity, rotation_a_to_b
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

def cartesian_to_spherical(vectors):
    if vectors.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)

    radii = np.linalg.norm(vectors, axis=1)
    xy_norm = np.linalg.norm(vectors[:, :2], axis=1)
    azimuth = np.arctan2(vectors[:, 1], vectors[:, 0])
    elevation = np.arctan2(vectors[:, 2], xy_norm)
    return np.stack([radii, azimuth, elevation], axis=1).astype(np.float32)

def spherical_edge_attr(points, edge_index):
    if edge_index.shape[1] == 0:
        return np.empty((0, 3), dtype=np.float32)

    rel_vecs = points[edge_index[1]] - points[edge_index[0]]
    return cartesian_to_spherical(rel_vecs)

def cartesian_to_norm(points,edge_index): #convert xyz coordinates to (dist*(uvw))
    vectors = points[edge_index[1]] - points[edge_index[0]]
    dist = np.linalg.norm(vectors, axis=1)
    dir_vec = vectors/dist[:,None]
    return np.concatenate([dist[:,None],dir_vec],axis=1).astype(np.float32)

def safe_knn_connectivity(points, k):
    num_nodes = points.shape[0]
    if num_nodes < 2:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)

    if num_nodes <= k:
        pairs = np.array(
            [[i, j] for i in range(num_nodes) for j in range(i + 1, num_nodes)],
            dtype=np.int64,
        )
        weights = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1).astype(np.float32)
        return pairs.T, weights

    return knn_connectivity(points, k)

def make_graph(pcd_points,pcd_colors,knn_edge_index,knn_edge_attr,visible_idx):
    pcd_points = pcd_points.astype(np.float32, copy=False)
    points_norm = np.linalg.norm(pcd_points,axis=1) # convert xyz to (dist,uvw)
    pcd_points = np.concatenate([points_norm[:,None],pcd_points/points_norm[:,None]],axis=1)

    pcd_colors = pcd_colors.astype(np.float32, copy=False)
    num_knn_nodes = pcd_points.shape[0]
    origin = np.zeros((1, 4), dtype=np.float32)
    origin_rgb = np.zeros((1, pcd_colors.shape[1]), dtype=np.float32)
    all_points = np.vstack([origin, pcd_points])
    all_colors = np.vstack([origin_rgb, pcd_colors])

    targets = np.arange(1, num_knn_nodes+1, dtype=np.int64)
    origin_sources = np.full(num_knn_nodes, 0, dtype=np.int64)
    origin_to_nodes = np.stack([origin_sources, targets], axis=0)
    nodes_to_origin = np.stack([targets, origin_sources], axis=0)
    origin_edge_index = np.concatenate([origin_to_nodes, nodes_to_origin], axis=1)

    #origin_edge_attr = spherical_edge_attr(all_points, origin_edge_index)
    origin_edge_attr = cartesian_to_norm(all_points[:,1:], origin_edge_index)

    data = Data()
    data.pos = torch.from_numpy(all_points)
    data.rgb = torch.from_numpy(all_colors)
    data.edge_index = torch.from_numpy(knn_edge_index+1).long()
    data.edge_attr = torch.from_numpy(knn_edge_attr).float()
    data.ei_camera = torch.from_numpy(origin_edge_index).long()
    data.ea_camera = torch.from_numpy(origin_edge_attr).float()
    data.origin_node_index = torch.tensor([0], dtype=torch.long)
    data.visible_point_indices = torch.from_numpy(visible_idx).long()
    return data

def raycast_img_with_points(trunc, view_dir, fov_x, fov_y, pcd_points, pcd_rgb, vantage, H, W,
    background=(0, 0, 0), return_depth=True):
    near, far = float(trunc[0]), float(trunc[1])
    forward = np.asarray(view_dir, dtype=np.float32)
    forward /= (np.linalg.norm(forward) + 1e-9)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(world_up, forward))) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right /= (np.linalg.norm(right) + 1e-9)
    up = np.cross(right, forward)

    rel = (pcd_points - vantage).astype(np.float32, copy=False)
    x = rel @ right
    y = rel @ up
    z = rel @ forward
    base_mask = (z > 1e-6) & (z >= near) & (z <= far)

    rgb_img = np.zeros((H * W, 3), dtype=np.uint8)
    rgb_img[:] = np.array(background, dtype=np.uint8)
    depth = np.full((H * W,), np.inf, dtype=np.float32) if return_depth else None

    empty_points = np.empty((0, 3), dtype=pcd_points.dtype)
    empty_colors = np.empty((0, pcd_rgb.shape[1]), dtype=pcd_rgb.dtype)
    empty_idx = np.empty((0,), dtype=np.int64)

    if not np.any(base_mask):
        rgb_img = rgb_img.reshape(H, W, 3)
        if return_depth:
            return rgb_img, depth.reshape(H, W), empty_idx, empty_points, empty_colors
        return rgb_img, empty_idx, empty_points, empty_colors

    base_idx = np.flatnonzero(base_mask)
    x = x[base_mask]
    y = y[base_mask]
    z = z[base_mask]

    tanx = np.tan(0.5 * float(fov_x))
    tany = np.tan(0.5 * float(fov_y))
    xn = x / z
    yn = y / z
    frustum_mask = (np.abs(xn) <= tanx) & (np.abs(yn) <= tany)

    if not np.any(frustum_mask):
        rgb_img = rgb_img.reshape(H, W, 3)
        if return_depth:
            return rgb_img, depth.reshape(H, W), empty_idx, empty_points, empty_colors
        return rgb_img, empty_idx, empty_points, empty_colors

    src_idx = base_idx[frustum_mask]
    xn = xn[frustum_mask]
    yn = yn[frustum_mask]
    z = z[frustum_mask]
    cols = pcd_rgb[src_idx]

    u = ((xn / tanx) * 0.5 + 0.5) * (W - 1)
    v = (0.5 - (yn / tany) * 0.5) * (H - 1)
    ui = np.clip(u.astype(np.int32), 0, W - 1)
    vi = np.clip(v.astype(np.int32), 0, H - 1)
    pix = vi * W + ui

    order = np.lexsort((z, pix))
    pix_s = pix[order]
    z_s = z[order]
    cols_s = cols[order]
    src_idx_s = src_idx[order]

    first = np.empty_like(pix_s, dtype=bool)
    first[0] = True
    first[1:] = pix_s[1:] != pix_s[:-1]

    pix_u = pix_s[first]
    z_u = z_s[first]
    cols_u = cols_s[first]
    visible_idx = src_idx_s[first]

    img_cols = cols_u
    if img_cols.dtype != np.uint8:
        img_cols = np.clip(img_cols * 255.0, 0, 255).astype(np.uint8)

    rgb_img[pix_u] = img_cols
    rgb_img = rgb_img.reshape(H, W, 3)

    if return_depth:
        depth[pix_u] = z_u.astype(np.float32)
        depth = depth.reshape(H, W)

    visible_order = np.argsort(visible_idx)
    visible_idx = visible_idx[visible_order]
    visible_points = pcd_points[visible_idx]
    visible_rgb = pcd_rgb[visible_idx]

    if return_depth:
        return rgb_img, depth, visible_idx, visible_points, visible_rgb
    return rgb_img, visible_idx, visible_points, visible_rgb

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

rects = [Rect(xmin=3.9,xmax=4.9,ymin=7.1,ymax=15.4),
         Rect(xmin=5.3,xmax=14.8,ymin=14.2,ymax=16.9),
         Rect(xmin=3.9,xmax=13.4,ymin=3.1,ymax=4.1),
         Rect(xmin=13.6,xmax=15.5,ymin=3.1,ymax=15.2),
         Rect(xmin=15.7,xmax=19.9,ymin=2.9,ymax=3.5)]

sampled_pts = sample_xy_in_union(rects,10000,rng=rng)
#print(sampled_pts)
path,dirs,_ = smooth_heading_walk(rng.choice(sampled_pts),rects,40,0.2,0.15,0.1,rng=rng)
print(path,dirs)

pcd_path = 'area_calc/Hospital2 1.pts'
pcd = o3d.io.read_point_cloud(pcd_path)
pcd_points = np.asarray(pcd.points)
pcd_colors = np.asarray(pcd.colors)
render_h = 192
render_w = 256
camera_intrinsics = {'W':6208,'H':4135,'fx':3408.59,'fy':3408.87}
fov_x = 2*np.arctan(camera_intrinsics['W']/(2*camera_intrinsics['fx']))
fov_y = 2*np.arctan(camera_intrinsics['H']/(2*camera_intrinsics['fy']))


for vantage_idx, point in enumerate(zip(path,dirs)):
    view_idx = f'{process_idx}_{vantage_idx}'
    out_path = f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/processed/hospital/random_walks/rw_{process_idx}_{vantage_idx}.pt'
    point, viewdir = point
    vantage = np.array([point[0], point[1], 0], dtype=np.float32)
    view_rgb, view_depth, visible_idx, pcd_visible_world, pcd_rgb = raycast_img_with_points(
        [0,40],
        viewdir,
        fov_x,
        fov_y,
        pcd_points,
        pcd_colors,
        vantage,
        render_h,
        render_w,
        (0,0,0),
    )
    e = np.array([1.,0.,0.])
    R_align = rotation_a_to_b(viewdir,e)
    pcd_visible_aligned = (R_align @ (pcd_visible_world - vantage).T).T
    ei, ew = safe_knn_connectivity(pcd_visible_aligned,3)
    data_graph = make_graph(pcd_visible_aligned,pcd_rgb,ei,ew,visible_idx)
    img_rgb = torch.from_numpy(view_rgb / 255.0).to(torch.float32)
    img_d = torch.from_numpy(view_depth).to(torch.float32)
    packed = {'img': img_rgb, 'depth': img_d, 'graph': data_graph,
              'loc': torch.from_numpy(vantage), 'view_dir': torch.from_numpy(viewdir).to(torch.float32),
              'ei_points':ei,'ew_points':ew,'ei_camera':data_graph.ei_camera,'ea_camera':data_graph.ea_camera}
    torch.save(packed,out_path)
    #break
if False:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    #plt.plot(path[:,0],path[:,1])
    plt.imshow(view_rgb)
    #plt.axis('equal')
    plt.savefig(f'../../../scratch/btuncay/cog/gnn_spatial_reasoning/logs/random_walk/test_{process_idx}_{vantage_idx}.png')
 