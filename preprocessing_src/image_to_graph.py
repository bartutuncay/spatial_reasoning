import numpy as np
import time
import json
import os
import itertools
import re
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import KDTree, Delaunay
from scipy.stats import qmc

## Relative Neighborhood Graph

def build_rng(pts, leaf_size=40):
    """
    Build the Relative Neighborhood Graph for a set of 3D points.
    Parameters:
    pts : (N,3) array of float, The 3D coordinates of N points.
    leaf_size : int, Leaf size for the KDTree; trades speed vs. memory.
    Returns:
    edges : List of tuple(int,int), Undirected edges (i,j), with i<j.
    """
    N = pts.shape[0]
    tree = KDTree(pts, leaf_size=leaf_size)
    tri = Delaunay(pts)
    delaunay_edges = set()
    for simplex in tri.simplices:
        # each simplex is a 4-tuple of point indices in 3D
        for i,j in itertools.combinations(simplex, 2):
            a, b = sorted((i,j))
            delaunay_edges.add((a,b))
    rng_edges = []
    for i, j in delaunay_edges:
        d_ij = np.linalg.norm(pts[i] - pts[j])
        # find all points within d_ij of both i and j
        # since most vertices of the Delaunay are “nearby”,
        # these radius queries tend to be very cheap.
        nbrs_i = set(tree.query_radius(pts[i:i+1], r=d_ij)[0])
        nbrs_j = set(tree.query_radius(pts[j:j+1], r=d_ij)[0])
        # remove the endpoints themselves
        nbrs_i.discard(i); nbrs_i.discard(j)
        nbrs_j.discard(i); nbrs_j.discard(j)
        if not (nbrs_i & nbrs_j):  # empty intersection ⇒ edge survives
            rng_edges.append((i, j))
    return sorted(rng_edges)

def plot_rng(pts3d, rng_edges):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    #ax.set_title("Relative Neighborhood Graph")
    # Scatter the 3D corner points

    mins = pts3d.min(axis=0)
    maxs = pts3d.max(axis=0)
    spans = maxs - mins
    pts3d = (pts3d - mins) / spans

    ax.scatter(pts3d[:, 0], pts3d[:, 1], s=2)
    # Draw an edge for each pair (i, j)
    for i, j in rng_edges:
        xs = [pts3d[i, 0], pts3d[j, 0]]
        ys = [pts3d[i, 1], pts3d[j, 1]]
        ax.plot(xs, ys, linewidth=0.5,color=(0.8,0,0))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.axis('equal')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.tight_layout()
    plt.show()

## Minimum Spanning Tree

def build_mst_edges(pts):
    """
    Build the Minimum Spanning Tree edges of a set of points.
    
    Parameters
    ----------
    pts : (N, d) array of float
        Coordinates in d dimensions (e.g. d=2 or d=3).

    Returns
    -------
    mst_edges : list of (i, j) tuples, i<j
        The index pairs of the MST edges.
    """
    # 1. Compute full pairwise distance matrix (N×N)
    D = distance_matrix(pts, pts)
    
    # 2. Compute the MST (returns a sparse matrix)
    mst_sparse = minimum_spanning_tree(D)
    
    # 3. Extract the nonzero entries as edge list
    #    mst_sparse is directed, but for Euclidean our graph is undirected
    coo = mst_sparse.tocoo()
    edges = []
    for u, v, w in zip(coo.row, coo.col, coo.data):
        # ensure i<j for consistency
        i, j = sorted((int(u), int(v)))
        edges.append((i, j))
    return edges

## Encoded probability map & cross-entropy calculation
def to_probability_map(canvas):
    """Normalize so sum=1, and add a tiny ε so log isn’t -inf."""
    p = canvas.astype(float)
    total = p.sum()
    if total == 0:
        return np.full_like(p, 1.0/p.size)
    p /= total
    # clamp to avoid zeros
    eps = 1e-12
    return np.clip(p, eps, 1.0)

## Return normalized points
def normalize_points(pts3d):
    mins = pts3d.min(axis=0)
    maxs = pts3d.max(axis=0)
    spans = maxs - mins
    pts2d_norm = (pts3d - mins) / spans
    return pts2d_norm

## Save output as JSON

def save_graph_to_json(path, X, mst_edges, rng_edges, mst_entropy, rng_entropy):
    """
    Save coordinates, edge-lists, and entropies all in one JSON file.
    
    Parameters
    ----------
    path : str
      Filename ending in .json
    X : (N,2) or (N,3) ndarray
      Node coordinates, assumed float
    mst_edges, rng_edges : list of (i,j) pairs
      Edge-lists
    mst_entropy, rng_entropy : float
      Precomputed entropy measures
    """
    # Convert arrays and edge-lists to Python structures
    payload = {
        "nodes": X.astype(float).tolist(),        # [[x0,y0], [x1,y1], …]
        "mst_edges": [[int(e[0]),int(e[1])] for e in mst_edges], # [[i,j], …]
        "rng_edges": [[int(e[0]),int(e[1])] for e in rng_edges],
        "mst_entropy": float(mst_entropy),
        "rng_entropy": float(rng_entropy),
    }

    # Write JSON
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    #print(f"Saved all graph data to {path}")

def radius_from_grad(g, min_r=5, max_r=20, eps=1e-3):
    return np.clip(max_r * ((g/eps)), min_r, max_r)

invariants = []

LHS_count = 5000
radius = 5 #search radius for normals
s_tol = 0.01 #similarity tolerance
trunc = 120 #max. distance
lb = 5 #min. distance

start = time.time()
sampler = qmc.LatinHypercube(2)
lhs_samples = sampler.random(LHS_count)
height, width = image.shape
samples_scaled = np.zeros_like(lhs_samples)
samples_scaled[:,0] = lhs_samples[:,0]*(height-1)
samples_scaled[:,1] = lhs_samples[:,1]*(width-1)
samples_idxs = samples_scaled.astype(int)
samples_idxs = np.unique(samples_idxs,axis=0)


dx,dy = np.gradient(grid_z)
#dx,dy = np.gradient(image)
#dx,dy = np.gradient(im_int)

dpxl = np.hypot(dx,dy)
dx2,dy2 = np.gradient(dpxl)
dpxl2 = np.hypot(dx2,dy2)
#get normals in 3D properly TODO
normals_unnorm = np.stack((-dx, -dy, np.ones_like(dx)), axis=-1)
length = np.linalg.norm(normals_unnorm, axis=2, keepdims=True)
epsilon = 1e-8
normals = normals_unnorm / (length + epsilon)

edgesc = cv2.Canny(image,20,255)/255

##neighborhood search for corners: try different scales and filtering
#select points which are most likely corners TODO - connected points with min. spanning trees?...
# ...skeletonize/retrive spatial info from neighborhood TODO
#identifier with statistical distribution TODO
#try edge DTW TODO
#descriptor: depth distribution in neighborhood, relative normals, edge features TODO
#tree = cKDTree(samples_scaled)
#_, idx1 = tree.query(samples_idxs)


def radius_from_grad(g, min_r=5, max_r=20, eps=1e-3):
    return np.clip(max_r * ((g/eps)), min_r, max_r)


#cull points on planes
filtered_results = []
planepts = []

results = []
for (r0,c0) in samples_idxs:
    g0   = dpxl[r0, c0]
    center_normal = normals[r0, c0]

    rad  = radius_from_grad(g0)
    rr,cc = disk((r0,c0), rad, shape=image.shape)
    neighbors = np.column_stack((rr, cc))
    nbr_normals = normals[rr, cc]
    rel_normals = nbr_normals - center_normal
    normals_norms = np.linalg.norm(rel_normals,axis=1)
    #get translation-invariant normals
    p5, p50, p95 = np.percentile(rel_normals, [5, 50, 95], axis=0)
    std_norm = np.std(rel_normals)
    IQRnorm = np.linalg.norm((p95-p5))
    if abs(g0) > 0 and IQRnorm > 1.6 and lb < image[r0,c0] < trunc: #choose points with most variation - ideal for renders
    #if abs(g0) > 0 and std_norm < 0.4 and lb < image[r0,c0] < trunc: #choose points with less variation - more robust vs. noise
        #filtered_results.append((r0,c0))
        filtered_results.append({
            "point": (r0, c0),
            "grad":  g0,
            "radius": rad,
            "neighbors": neighbors,
            "rel_p5":     p5,
            "rel_p50":    p50,
            "rel_p95":    p95,
            "var_norm":IQRnorm,
            "std_norm":std_norm})
        ##use IQR as metric
    if normals_norms.max() == 0 and image[r0,c0] < trunc:
        #get plane info TODO
        planepts.append((r0,c0))
planepts = np.array(planepts)
filtered_pts = np.array([d["point"] for d in filtered_results])
filtered_iqr = np.array([d["var_norm"] for d in filtered_results])
end = time.time()

#save_graph_to_json(f'80.json',filtered_norm,spanning_tree,neighborhood_graph,ent_mst,ent_rng)

print('duration:',end-start)
##plane detection with DBSCAN
#alpha = 10.0   # relative weighting of normals vs. coords
#features = np.hstack([ alpha * plane_normals,planepts ])
#db = DBSCAN(eps=0.5, min_samples=30).fit(features)
#labels = db.labels_
#unique_labels = [L for L in np.unique(labels) if L>=0]
#planes = []