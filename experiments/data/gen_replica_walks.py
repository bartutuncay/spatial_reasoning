"""Render random walks from Replica v1 meshes (multi-room geometric-evidence
dataset for the anatomy matrix: apartments/hotels give the pairwise-distance
variance that single-room ScanNet crops lack).

Reuses the ScanNet pipeline (dense surface sampling of the textured mesh + the
ETH3D raycaster). Replica-specific parts:
  1. no axisAlignment file: gravity axis auto-detected (bbox axis with the
     smallest extent - rooms are wider than tall), sign chosen so the denser
     slab (floor + furniture) sits at the bottom; --up/--flip override.
  2. meshes are watertight-ish and dense, so renders should fill ~fully
     (unlike ~30% ScanNet fill).

Output schema matches Bartu's walks exactly:
  datasets_replica/walks/<scene>/random_walks/rw_<seed>_<step>.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))                                  # repo root for experiments.*
from experiments.data.gen_scannet_walks import (  # noqa: E402
    floor_region, render_and_save, smooth_heading_walk,
)


def gravity_align(pts, up="auto", flip="auto"):
    """Permute/flip axes so gravity is -z (floor at low z). Returns pts."""
    if up == "auto":
        ext = pts.max(0) - pts.min(0)
        up_ax = int(np.argmin(ext))            # rooms are wider than tall
    else:
        up_ax = {"x": 0, "y": 1, "z": 2}[up]
    if up_ax != 2:
        order = [i for i in range(3) if i != up_ax] + [up_ax]
        pts = pts[:, order]
    z = pts[:, 2]
    if flip == "auto":
        lo = ((z <= z.min() + 0.5)).sum()      # floor slab: floor + furniture bases
        hi = ((z >= z.max() - 0.5)).sum()      # ceiling slab: mostly bare plane
        do_flip = hi > lo
    else:
        do_flip = flip == "yes"
    if do_flip:
        pts[:, 2] = -pts[:, 2]
    print(f"gravity_align: up_ax={up_ax} flipped={do_flip}", flush=True)
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)                 # e.g. frl_apartment_0
    ap.add_argument("--replica-root", default="datasets_replica/replica_v1")
    ap.add_argument("--out-root", default="datasets_replica/walks")
    ap.add_argument("--seeds", type=int, default=26)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--step-len", type=float, default=0.15)
    ap.add_argument("--n-points", type=int, default=3_000_000)  # big multi-room scenes
    ap.add_argument("--up", default="auto", choices=["auto", "x", "y", "z"])
    ap.add_argument("--flip", default="auto", choices=["auto", "yes", "no"])
    args = ap.parse_args()

    sc = args.scene
    ply = Path(args.replica_root) / sc / "mesh.ply"
    out_dir = Path(args.out_root) / sc / "random_walks"
    out_dir.mkdir(parents=True, exist_ok=True)

    tm = o3d.io.read_triangle_mesh(str(ply))
    if not tm.has_vertex_colors():
        tm.paint_uniform_color([0.5, 0.5, 0.5])
    pcd = tm.sample_points_uniformly(number_of_points=args.n_points)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    pts = gravity_align(pts, args.up, args.flip)
    print(f"{sc}: {len(pts)} pts, z[{pts[:,2].min():.2f},{pts[:,2].max():.2f}] "
          f"xy extent {pts[:,0].max()-pts[:,0].min():.1f}x{pts[:,1].max()-pts[:,1].min():.1f}m",
          flush=True)

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
