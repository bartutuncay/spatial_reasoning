## Helper Functions

# libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from shapely.geometry import Polygon
from shapely.ops import unary_union
import cv2
from scipy.spatial import KDTree
from sklearn.neighbors import KNeighborsRegressor
import time
from scipy.stats import qmc
from SALib.analyze.morris import analyze as morris_analyze


def ray_cast_points(segments, origin, num_rays=360, max_dist=20, view_center=np.pi, fov_x=2*np.pi): #similar to spherical 3D method; only linear segments
    start = time.time()
    resolution = 15/num_rays #rule of thumb
    
    ### exclude curves with both endpoints outside visible range
    segments_diffs = segments-origin
    segments_dists = np.hypot(segments_diffs[:,:,0],segments_diffs[:,:,1])
    segment_mask = (segments_dists[:,0] <= max_dist) | (segments_dists[:,1] <= max_dist)
    #proximity_mask = (segments_dists[:,0] <= num_rays/60) | (segments_dists[:,1] <= num_rays/60)
    segments = segments[segment_mask]
    ###
    #get norms of each curve
    norms_curves = np.linalg.norm(segments[:,1,:]-segments[:,0,:],axis=1)/resolution
    norms_curves = np.array(norms_curves,dtype=int)
    #create points
    idx    = np.arange(norms_curves.max())
    T = idx[None,:] / (norms_curves[:,None] - 1 + 1e-6)
    starts = segments[:, 0, :]
    deltas = segments[:, 1, :] - starts 
    pts = starts[:, None, :] + deltas[:, None, :] * T[:, :, None]
    #vstack points
    pts_flat  = pts.reshape(-1, 2) 
    mask = idx[None, :] < norms_curves[:, None]
    mask_flat = mask.flatten()
    valid_pts = pts_flat[mask_flat]
    #raycasting directions
    #angles = np.linspace(0,2*np.pi,num_rays)
    angles = np.linspace(view_center-(fov_x/2),view_center+(fov_x/2),num_rays)
    angles = np.vstack([np.sin(angles),np.cos(angles)]).T.astype(np.float32)
    #kdtree
    tree = KDTree(angles,leafsize=8)
    rel_positions = valid_pts-origin
    distances = np.linalg.norm(rel_positions,axis=1,keepdims=True)
    rel_dirs = rel_positions/(distances+1e-6)
    dist_mask = (distances[:,0] <= max_dist)
    rel_dirs = rel_dirs[dist_mask]
    rel_positions = rel_positions[dist_mask]
    distances = distances[dist_mask]
    _, idx = tree.query(rel_dirs, k=1)
    dist_per_bin = np.full(num_rays, 1e3) 
    np.minimum.at(dist_per_bin, idx, distances[:,0])
    dist_per_bin[dist_per_bin>max_dist] = max_dist
    known_pts = (angles*dist_per_bin[:,None])+origin
    end = time.time()
    return known_pts, angles

def sort_points_counterclockwise(points, origin):
    ox, oy = origin
    return sorted(points, key=lambda p: np.arctan2(p[1] - oy, p[0] - ox))

def compute_isovist_area(visible_points):
    if len(visible_points) < 3:
        return 0  # Not enough points to form an area
    polygon = Polygon(visible_points)
    return polygon.area

def compute_area_numpy(points):
    points = np.array(points,dtype=float)
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def distances(visible_points,observation_point):
    dist = []
    if len(visible_points) != 0:
        for point in visible_points:
            distance = (point[0]-observation_point[0])**2 + (point[1]-observation_point[1])**2
            dist.append(distance**0.5)
    return dist


# vectorized method from verification notebook
def compute_visibility_area_np(floor_lines, vantage_point, max_distance=100.0, num_rays=3600,view_center=np.pi,fov_x=2*np.pi):
    """
    Vectorized ray-to-segment intersection via Cramer's rule, without Shapely in the loop.
    """
    start = time.time()
    x0, y0 = vantage_point
    P = np.array([x0, y0])
    segments_diffs = floor_lines-vantage_point
    segments_dists = np.hypot(segments_diffs[:,:,0],segments_diffs[:,:,1])
    segment_mask = (segments_dists[:,0] <= max_distance) | (segments_dists[:,1] <= max_distance)
    floor_lines = floor_lines[segment_mask]
    # Extract 2D segment endpoints (n_segments × 2)
    A = floor_lines[:, 0]
    B = floor_lines[:, 1]
    D = B - A               # segment direction vectors (n_segments × 2)

    angles = np.linspace(view_center-(fov_x/2),view_center+(fov_x/2),num_rays,endpoint=False)
    #angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    hit_pts = []

    for θ in angles:
        # Ray direction and endpoint
        dir_vec = np.array([np.cos(θ), np.sin(θ)])
        ray_endpoint = P + dir_vec * max_distance

        # Right‑hand side: A − P
        rhs = A - P  # shape (n_segments, 2)

        # Compute determinant of the 2×2 system for each segment:
        # det = dir_x * (−D_y) − dir_y * (−D_x)
        det = dir_vec[0] * (-D[:, 1]) - dir_vec[1] * (-D[:, 0])

        # Cramer's numerators:
        # det_t = (A_x − x0)*(-D_y) − (A_y − y0)*(-D_x)
        det_t = rhs[:, 0] * (-D[:, 1]) - rhs[:, 1] * (-D[:, 0])
        # det_u = dir_x*(A_y − y0) − dir_y*(A_x − x0)
        det_u = dir_vec[0] * rhs[:, 1] - dir_vec[1] * rhs[:, 0]

        # Solve t and u where valid
        valid = det != 0
        t = np.full_like(det, np.inf, dtype=float)
        u = np.full_like(det, -1.0, dtype=float)

        t[valid] = det_t[valid] / det[valid]
        u[valid] = det_u[valid] / det[valid]

        # Keep intersections with 0 ≤ u ≤ 1 (on segment) and 0 < t < max_distance
        mask = (t > 0) & (t < max_distance) & (u >= 0) & (u <= 1)
        if np.any(mask):
            t_min = t[mask].min()
            hit = P + dir_vec * t_min
        else:
            hit = ray_endpoint

        hit_pts.append((hit[0], hit[1]))

    # Build the visibility polygon (points already in angular order)
    visibility_poly = Polygon(hit_pts)
    end = time.time()
    #print(f'Visibility area Numpy method, area calc duration: {end-start:.4f} s')
    return visibility_poly.area, visibility_poly, np.array(hit_pts)-vantage_point


# corner-based reconstruction method
#use points from discretized line -> get corners -> calculate concave volume
def visibility_polygon(segments, observer, max_distance = 100, num_samples = 120, view_center=np.pi,fov_x=2*np.pi):
    start = time.time()
    segments = np.array(segments, dtype=float) # shape (N, 2, 2)
    segments_diffs = segments-observer
    segments_dists = np.hypot(segments_diffs[:,:,0],segments_diffs[:,:,1])
    segment_mask = (segments_dists[:,0] <= max_distance) | (segments_dists[:,1] <= max_distance)
    valid_segments = segments_diffs[segment_mask]
    valid_dists = segments_dists[segment_mask]
    
    ### break up long segments
    norms_curves = np.linalg.norm(valid_segments[:,1,:]-valid_segments[:,0,:],axis=1)
    max_subd = 5
    subdivision_mask = (norms_curves >= max_subd)
    long_segments = valid_segments[subdivision_mask]
    short_segments = valid_segments[~subdivision_mask]
    long_norms = norms_curves[subdivision_mask]
    short_dists = valid_dists[~subdivision_mask]
    n_segs  = np.ceil(long_norms / max_subd).astype(int)
    start = long_segments[:, 0, :]
    end   = long_segments[:, 1, :]
    vec   = end - start
    t = np.concatenate([np.linspace(0, 1, ns+1)[:-1] for ns in n_segs])
    t_next = np.concatenate([np.linspace(0, 1, ns+1)[1:] for ns in n_segs])
    start_rep = np.repeat(start, n_segs, axis=0)
    vec_rep   = np.repeat(vec, n_segs, axis=0)
    seg_start = start_rep + vec_rep * t[:, None]
    seg_end   = start_rep + vec_rep * t_next[:, None]
    long_subdivided = np.stack([seg_start, seg_end], axis=1)
    long_dists = np.hypot(long_subdivided[:,:,0],long_subdivided[:,:,1])
    valid_segments = np.vstack([short_segments, long_subdivided])
    valid_dists = np.vstack([short_dists, long_dists])
    ###
    
    segments_dirs = valid_segments/valid_dists[:,:,None]
    segments_angles = np.arctan2(segments_dirs[:,:,1], segments_dirs[:,:,0])
    
    #each row in segments_angles goes col1 -> col2: CCW
    angle_diff = (segments_angles[:,1] - segments_angles[:,0]) % (2*np.pi)
    swap_mask = angle_diff < 0
    swap_mask |= angle_diff > np.pi
    out = segments_angles.copy()
    out_d = valid_dists.copy()
    out[swap_mask, 0] = segments_angles[:,1][swap_mask]
    out[swap_mask, 1] = segments_angles[:,0][swap_mask]
    #adjust distances array accordingly
    out_d[swap_mask,0] = valid_dists[:,1][swap_mask]
    out_d[swap_mask,1] = valid_dists[:,0][swap_mask]
    #sort by angle CCW
    idx = np.argsort((out[:, 0] - 0.0) % (2*np.pi), kind='mergesort')
    out_dists = out_d[idx]
    out_angles = out[idx]

    ### if difference in a row >= π and (+) to (-) -> split and flip direction
    out_flip_mask = (out_angles[:,0] > out_angles[:,1]) & (out_angles[:,0]/out_angles[:,1] < 0) & (np.abs(out_angles[:,0]-out_angles[:,1])>=np.pi)
    ordered_angles = out_angles[~out_flip_mask]
    flipped_angles = out_angles[out_flip_mask]
    ordered_dists = out_dists[~out_flip_mask]
    flipped_dists = out_dists[out_flip_mask]
    n = flipped_angles.shape[0]
    flipped_1 = np.column_stack([flipped_angles[:, 0], np.pi * np.ones(n)])
    flipped_2 = np.column_stack([-np.pi * np.ones(n), flipped_angles[:, 1]])
    flipped_angles = np.vstack([flipped_1, flipped_2])
    flipped_d1 = np.column_stack([flipped_dists[:, 0], flipped_dists[:,1]])
    flipped_d2 = np.column_stack([flipped_dists[:, 0], flipped_dists[:,1]])
    flipped_dists = np.vstack([flipped_d1, flipped_d2])
    out_angles = np.vstack([ordered_angles,flipped_angles])
    out_dists = np.vstack([ordered_dists,flipped_dists])
    ###
    
    ##calculate area
    #project onto a line 0 -> 2π
    #intersection of intervals -> select lower bound
    tri_points1 = np.column_stack((out_angles[:,0],out_dists[:,0]))
    tri_points2 = np.column_stack((out_angles[:,1],out_dists[:,1]))
    tri_points = np.column_stack((tri_points1,tri_points2))
    tri_points = np.concatenate((tri_points,np.array([[out_angles.min(), max_distance, out_angles.max(), max_distance]])))

    segments = list(zip(tri_points[:,:2], tri_points[:,2:]))
    
    xs_all = [x for seg in segments for (x,_) in seg]
    xmin, xmax = min(xs_all), max(xs_all)
    xs = np.linspace(xmin, xmax, num_samples)
    ys = np.full_like(xs, np.inf, dtype=float)

    for (x1, y1), (x2, y2) in segments:
        mask = (xs >= min(x1,x2)) & (xs <= max(x1,x2))
        if x2 != x1:
            slope = (y2-y1)/(x2-x1)
            ys_seg = y1 + slope*(xs[mask]-x1)
        else:
            ys_seg = np.full(np.sum(mask), min(y1,y2))
        ys[mask] = np.minimum(ys[mask], ys_seg)
    ys_clipped = np.clip(ys, 0, None)
    #area = np.trapz(ys_clipped, xs)
    area = (ys_clipped**2).mean()*np.pi

    end = time.time()
    #print(end-start)
    return area, ys_clipped



## Combined Function
def compute_area_directional(floor_plan,method,ndirs,vantage_point,max_distance,view_dir,fov_x):
    #start = time.time()
    view_center = np.arctan2(view_dir[1],view_dir[0])
    if method == 'segments_angle':
        visible_area,_,visible_points = compute_visibility_area_np(floor_plan,vantage_point,max_distance,ndirs,view_center,fov_x)
        bins_distance = np.linalg.norm(visible_points,axis=1)
    if method == 'disc_line':
        visible_points,_ = ray_cast_points(floor_plan,vantage_point,ndirs,max_distance,view_center,fov_x)
        visible_area = compute_isovist_area(visible_points)
    if method == 'corner':
        visible_area, bins_distance = visibility_polygon(floor_plan, vantage_point, max_distance, ndirs, view_center, fov_x) ##TODO: adapt directional calc
    return visible_area, bins_distance

