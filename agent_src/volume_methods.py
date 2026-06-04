import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, KDTree, cKDTree
from scipy.interpolate import interp1d
import pandas as pd
import open3d.core as o3c
import time
import shapely
from shapely import plotting
from scipy.interpolate import CubicSpline
from sklearn.neighbors import KNeighborsRegressor

def sort_points_counterclockwise(points, origin):
    ox, oy = origin
    return sorted(points, key=lambda p: np.arctan2(p[1] - oy, p[0] - ox))

def directional_raycast(trunc,ndirs,view_dir,fov_x,fov_y,pcd_points,pcd_rgb,vantage,heightlims,lim_height=False):
    near = trunc[0]
    far = trunc[1]
    rel_positions = pcd_points-vantage
    pcd_distances = np.linalg.norm(rel_positions,axis=1,keepdims=True)
    mask = (pcd_distances[:,0] >= near) & (pcd_distances[:,0] <= far)
    pcd_dirs = rel_positions/(pcd_distances+1e-5) #unit sphere
    pcd_dirs = pcd_dirs[mask]
    rel_positions = rel_positions[mask]
    pcd_distances = pcd_distances[mask]
    pcd_rgb = pcd_rgb[mask]

    indices = np.arange(0, ndirs, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/ndirs)
    theta = np.pi * (1 + 5**0.5) * indices
    dirs = np.vstack([np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),np.cos(phi)]).T.astype(np.float32)

    # truncate view directions according to intrinsics
    forward = view_dir / (np.linalg.norm(view_dir) + 1e-9)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if np.abs(np.dot(world_up, forward)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-9
    up = np.cross(right, forward)
    x = dirs @ right
    y = dirs @ up
    z = dirs @ forward 
    h_ang = np.arctan2(x, z)
    v_ang = np.arctan2(y, z)
    mask_dir = (z > 0) & (np.abs(h_ang) <= fov_x * 0.5) & (np.abs(v_ang) <= fov_y * 0.5)
    dirs = dirs[mask_dir]
    ndirs = dirs.shape[0]
    tree = KDTree(dirs, leafsize=8)
    _, idx = tree.query(pcd_dirs, k=1)

    print(len(dirs))
    print(mask.sum())

    dist_per_bin = np.ones(ndirs)*1e5
    np.minimum.at(dist_per_bin, idx, pcd_distances[:,0])

    if lim_height == True:
        # flatten floor and calculated ceiling height --> clip bins
        #heightlims = [floor,ceiling] --> relative to vantage point height
        #extrapolate heights from dist_per_bin, pcd_dirs
        rel_heights = dist_per_bin * dirs[:, 2]
        height_o_mask = rel_heights > (heightlims[1] - vantage[2])
        height_u_mask = rel_heights < (heightlims[0] - vantage[2])
        #calculate differences
        height_o_diff = np.abs(dist_per_bin[height_o_mask] - (heightlims[1] - vantage[2]))
        height_u_diff = np.abs(dist_per_bin[height_u_mask] + (heightlims[0] - vantage[2]))
        # safeguard against sqrt of negative values
        val_o = dist_per_bin[height_o_mask]**2 - height_o_diff**2
        val_u = dist_per_bin[height_u_mask]**2 - height_u_diff**2
        val_o[val_o < 0] = 0
        val_u[val_u < 0] = 0
        #get norm -> subtract difference from norm^2
        dist_per_bin[height_o_mask] = np.sqrt(val_o)
        dist_per_bin[height_u_mask] = np.sqrt(val_u)
    
    dist_per_bin[dist_per_bin>far] = 0
    ts = dist_per_bin
    all_idx   = np.arange(dist_per_bin.size)
    known_idx = all_idx[dist_per_bin != 0]
    #known_pts = pcd_points[known_idx]
    #pts_colors = pcd_rgb[known_idx]
    #rel_pts = rel_positions[known_idx]
    #print(rel_pts.shape)
    
    dist_per_bin[(~np.isfinite(dist_per_bin)) | (dist_per_bin > far)] = 0.0
    valid_rays = dist_per_bin > 0.0
    rel_pts = dirs[valid_rays] * dist_per_bin[valid_rays, None]
    
    tree_relpts = KDTree(rel_positions,leafsize=8)
    _, rgb_idx = tree_relpts.query(rel_pts,k=1)
    pts_colors = pcd_rgb[rgb_idx]

    alpha = 0.5 * fov_x
    beta  = 0.5 * fov_y
    Omega = 4.0 * np.arcsin(np.sin(alpha) * np.sin(beta))
    N = dist_per_bin.size
    vol = (Omega / (3.0 * N)) * np.sum(dist_per_bin**3)
    min_dist = dist_per_bin.min()
    max_dist = dist_per_bin.max()
    mean_dist = dist_per_bin.mean()
    dist_stdev = dist_per_bin.std()

    return rel_pts, pts_colors, vol, min_dist, max_dist, mean_dist, dist_stdev


def directional_raycast_fast(
    trunc, ndirs, view_dir, fov_x, fov_y,
    pcd_points, pcd_rgb, vantage,
    heightlims, lim_height=False,
    *,
    precomputed=None,          # optional dict with {"dirs":..., "tree":...}
    leafsize=16
):
    near, far = float(trunc[0]), float(trunc[1])
    near2, far2 = near * near, far * far

    # --- relative positions & cheap range mask using squared distance
    rel = pcd_points - vantage
    dist2 = np.einsum("ij,ij->i", rel, rel)  # fast squared norm
    mask = (dist2 >= near2) & (dist2 <= far2)

    if not np.any(mask):
        # no points in range -> no hits
        # (match your output schema; empty rel_pts, empty colors, zeros for stats)
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=pcd_rgb.dtype),
            0.0, 0.0, 0.0, 0.0, 0.0
        )

    rel = rel[mask]
    dist2 = dist2[mask]
    rgb = pcd_rgb[mask]

    # compute distances only for survivors
    dist = np.sqrt(dist2).astype(np.float32)
    inv_dist = (1.0 / (dist + 1e-5)).astype(np.float32)
    pcd_dirs = (rel * inv_dist[:, None]).astype(np.float32)

    # --- directions + KDTree (allow reuse)
    if precomputed is not None:
        dirs = precomputed["dirs"]
        tree = precomputed["tree"]
    else:
        indices = (np.arange(ndirs, dtype=np.float32) + 0.5)
        phi = np.arccos(1.0 - 2.0 * indices / ndirs)
        theta = np.pi * (1.0 + np.sqrt(5.0)) * indices
        dirs = np.vstack([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ]).T.astype(np.float32)

        # truncate view directions according to intrinsics
        forward = view_dir.astype(np.float32)
        forward /= (np.linalg.norm(forward) + 1e-9)

        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if np.abs(np.dot(world_up, forward)) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        right = np.cross(forward, world_up)
        right /= (np.linalg.norm(right) + 1e-9)
        up = np.cross(right, forward)

        x = dirs @ right
        y = dirs @ up
        z = dirs @ forward
        h_ang = np.arctan2(x, z)
        v_ang = np.arctan2(y, z)

        mask_dir = (z > 0) & (np.abs(h_ang) <= 0.5 * fov_x) & (np.abs(v_ang) <= 0.5 * fov_y)
        dirs = dirs[mask_dir]

        # rebuild ndirs after FOV crop
        ndirs = int(dirs.shape[0])

        tree = cKDTree(dirs, leafsize=leafsize)

    # --- assign each point to nearest ray direction
    # cKDTree.query returns shape (N,) if k=1
    _, idx = tree.query(pcd_dirs, k=1, workers=-1)
    idx = idx.astype(np.int64, copy=False)

    # --- compute per-ray closest distance AND which point caused it (no second KDTree)
    # Sort by (ray_idx, dist) and take first occurrence per ray
    order = np.lexsort((dist, idx))
    idx_s = idx[order]
    dist_s = dist[order]
    rgb_s = rgb[order]

    unique_rays, first_pos = np.unique(idx_s, return_index=True)

    dist_per_bin = np.zeros(ndirs, dtype=np.float32)
    hit_rgb = np.zeros((ndirs, rgb.shape[1]), dtype=rgb.dtype)

    # fill only rays that got at least one point
    dist_per_bin[unique_rays] = dist_s[first_pos]
    hit_rgb[unique_rays] = rgb_s[first_pos]

    # --- optional height limiting (same structure as yours)
    if lim_height:
        rel_heights = dist_per_bin * dirs[:, 2]
        height_o_mask = rel_heights > (heightlims[1] - vantage[2])
        height_u_mask = rel_heights < (heightlims[0] - vantage[2])

        # your original math (kept), but vectorized
        # NOTE: this logic is a bit odd dimensionally, but preserving behavior.
        height_o_diff = np.abs(dist_per_bin[height_o_mask] - (heightlims[1] - vantage[2]))
        height_u_diff = np.abs(dist_per_bin[height_u_mask] + (heightlims[0] - vantage[2]))

        val_o = dist_per_bin[height_o_mask] ** 2 - height_o_diff ** 2
        val_u = dist_per_bin[height_u_mask] ** 2 - height_u_diff ** 2
        val_o[val_o < 0] = 0
        val_u[val_u < 0] = 0

        dist_per_bin[height_o_mask] = np.sqrt(val_o, dtype=np.float32)
        dist_per_bin[height_u_mask] = np.sqrt(val_u, dtype=np.float32)

    # truncate far / invalid
    dist_per_bin[(~np.isfinite(dist_per_bin)) | (dist_per_bin > far)] = 0.0

    valid_rays = dist_per_bin > 0.0
    rel_pts = (dirs[valid_rays] * dist_per_bin[valid_rays, None]).astype(np.float32)
    pts_colors = hit_rgb[valid_rays]

    # --- volume/stats (same)
    alpha = 0.5 * fov_x
    beta = 0.5 * fov_y
    Omega = 4.0 * np.arcsin(np.sin(alpha) * np.sin(beta))
    N = dist_per_bin.size
    vol = (Omega / (3.0 * N)) * np.sum(dist_per_bin ** 3)

    min_dist = float(dist_per_bin.min(initial=0.0))
    max_dist = float(dist_per_bin.max(initial=0.0))
    mean_dist = float(dist_per_bin.mean())
    dist_stdev = float(dist_per_bin.std())

    return rel_pts, pts_colors, float(vol), min_dist, max_dist, mean_dist, dist_stdev