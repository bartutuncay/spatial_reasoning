"""Render random walks from ScanNet meshes (generality dataset for the anatomy
matrix). Reuses the ETH3D raycaster; the only ScanNet-specific parts are:
  1. apply the .txt `axisAlignment` 4x4 so z points up (floor -> horizontal plane);
  2. auto-derive the navigable region — floor = lowest horizontal band of points,
     rasterized to a 0.1 m occupancy grid; walks stay on occupied floor cells;
  3. virtual camera 1.4 m above the floor (fixed FOV so every scene looks uniform
     to the probe, independent of ScanNet's per-scene intrinsics).

Output schema matches Bartu's walks exactly, so every probe works unchanged:
  datasets_scannet/walks/<scene>/random_walks/rw_<seed>_<step>.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent_src" / "rw_small_step"))
from pcd_slice_methods import rotation_a_to_b  # noqa: E402
from experiments.data.gen_branch_walks import (  # noqa: E402
    make_graph, raycast_img_with_points, safe_knn_connectivity,
)

RENDER_H, RENDER_W = 192, 256
FOV_X = float(np.deg2rad(70.0))          # fixed virtual camera (uniform across scenes)
FOV_Y = float(np.deg2rad(55.0))
PITCH = -0.1
CAM_HEIGHT = 1.4                          # metres above floor (typical handheld)
GRID = 0.10                               # occupancy cell size (m)


def load_axis_alignment(txt_path):
    for line in open(txt_path):
        if line.startswith("axisAlignment"):
            v = [float(x) for x in line.split("=")[1].split()]
            return np.array(v, dtype=np.float64).reshape(4, 4)
    return np.eye(4)


def floor_region(points):
    """Return (in_region(xy)->bool, floor_z, sample_xy(rng)->(x,y)) from the
    lowest horizontal band of an axis-aligned point cloud (z up)."""
    z = points[:, 2]
    floor_z = float(np.percentile(z, 2))
    band = points[(z >= floor_z - 0.05) & (z <= floor_z + 0.20)]   # near-floor slab
    if len(band) < 100:
        band = points[z <= np.percentile(z, 10)]
    xy = band[:, :2]
    xmin, ymin = xy.min(0)
    ncx = max(1, int(np.ceil((xy[:, 0].max() - xmin) / GRID)))
    ncy = max(1, int(np.ceil((xy[:, 1].max() - ymin) / GRID)))
    occ = np.zeros((ncx + 1, ncy + 1), dtype=bool)
    ix = np.clip(((xy[:, 0] - xmin) / GRID).astype(int), 0, ncx)
    iy = np.clip(((xy[:, 1] - ymin) / GRID).astype(int), 0, ncy)
    occ[ix, iy] = True
    occupied_xy = np.stack([xmin + ix * GRID, ymin + iy * GRID], 1)
    occupied_xy = np.unique(occupied_xy, axis=0)

    def in_region(p):
        cx = int((p[0] - xmin) / GRID); cy = int((p[1] - ymin) / GRID)
        return 0 <= cx <= ncx and 0 <= cy <= ncy and occ[cx, cy]

    def sample_xy(rng):
        return occupied_xy[rng.integers(0, len(occupied_xy))] + rng.uniform(0, GRID, 2)

    return in_region, floor_z, sample_xy


def smooth_heading_walk(start_xy, in_region, n_steps, step, rng,
                        sigma_theta=0.15, sigma_step=0.02, sigma_view=0.07, view_pull=0.1):
    xy = np.array(start_xy, float)
    theta = rng.uniform(0, 2 * np.pi); view_yaw = theta
    cp, sp = np.cos(PITCH), np.sin(PITCH)
    path, dirs = [], []
    for _ in range(n_steps):
        theta = theta + rng.normal(scale=sigma_theta)
        L = max(0.0, step + rng.normal(scale=sigma_step))
        prop = xy + L * np.array([np.cos(theta), np.sin(theta)])
        if in_region(prop):
            xy = prop
        else:
            theta = theta + np.pi + rng.normal(scale=0.05)
            prop = xy + L * np.array([np.cos(theta), np.sin(theta)])
            if in_region(prop):
                xy = prop
        view_yaw = (1 - view_pull) * view_yaw + view_pull * theta + rng.normal(scale=sigma_view)
        cy, sy = np.cos(view_yaw), np.sin(view_yaw)
        d = np.array([cp * cy, cp * sy, sp]); d /= (np.linalg.norm(d) + 1e-12)
        path.append(xy.copy()); dirs.append(d)
    return np.array(path), np.array(dirs)


def render_and_save(xy, viewdir, floor_z, pts, cols, out_path):
    vantage = np.array([xy[0], xy[1], floor_z + CAM_HEIGHT], dtype=np.float32)
    rgb, depth, vis_idx, vis_world, vis_rgb = raycast_img_with_points(
        [0, 40], viewdir, FOV_X, FOV_Y, pts, cols, vantage, RENDER_H, RENDER_W, (0, 0, 0))
    if vis_world.shape[0] < 3:
        return False
    R = rotation_a_to_b(viewdir, np.array([1., 0., 0.]))
    aligned = (R @ (vis_world - vantage).T).T
    ei, ew = safe_knn_connectivity(aligned, 3)
    g = make_graph(aligned, vis_rgb, ei, ew, vis_idx)
    torch.save({"img": torch.from_numpy(rgb / 255.0).to(torch.float32),
                "depth": torch.from_numpy(depth).to(torch.float32), "graph": g,
                "loc": torch.from_numpy(vantage),
                "view_dir": torch.from_numpy(viewdir.astype(np.float32)),
                "ei_points": ei, "ew_points": ew,
                "ei_camera": g.ei_camera, "ea_camera": g.ea_camera}, out_path)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)                 # e.g. scene0000_00
    ap.add_argument("--scannet-root", default="datasets_scannet/scans")
    ap.add_argument("--out-root", default="datasets_scannet/walks")
    ap.add_argument("--seeds", type=int, default=26)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--step-len", type=float, default=0.12)
    args = ap.parse_args()

    sc = args.scene
    sroot = Path(args.scannet_root) / sc
    ply = sroot / f"{sc}_vh_clean_2.ply"
    out_dir = Path(args.out_root) / sc / "random_walks"
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_point_cloud(str(ply))
    pts = np.asarray(mesh.points)
    cols = np.asarray(mesh.colors)
    if len(cols) != len(pts):                                 # some meshes lack colors
        cols = np.full((len(pts), 3), 0.5, dtype=np.float32)
    A = load_axis_alignment(sroot / f"{sc}.txt")
    pts = (A[:3, :3] @ pts.T).T + A[:3, 3]                     # gravity-align (z up)
    print(f"{sc}: {len(pts)} pts, z[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]", flush=True)

    in_region, floor_z, sample_xy = floor_region(pts)
    pts = pts.astype(np.float32); cols = cols.astype(np.float32)

    saved = 0
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        for _ in range(50):
            start = sample_xy(rng)
            if in_region(start):
                break
        path, dirs = smooth_heading_walk(start, in_region, args.steps, args.step_len, rng)
        for t, (xy, vd) in enumerate(zip(path, dirs)):
            if render_and_save(xy, vd, floor_z, pts, cols, out_dir / f"rw_{seed}_{t}.pt"):
                saved += 1
    print(f"{sc}: saved {saved}/{args.seeds * args.steps} frames to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
