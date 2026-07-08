"""Generate BRANCHING walks for action-conditioned rollout (Wave-3 A2).

From each anchor pose, B=3 branches with distinct heading actions (-45/0/+45 deg)
roll K deterministic steps: the future frame is undetermined without the action,
so a positive rollout action-gap becomes achievable and meaningful.

Output: <out-root>/<scene>/branch_walks/bw_<anchor>_<branch>_<step>.pt with the
SAME sample schema as Bartu's random walks (img float 0-1, depth, graph, loc,
view_dir, ei/ew_points, ei/ea_camera). Step 0 is the shared anchor frame,
identical across the three branches by construction.

Rendering/graph helpers are copied verbatim from
agent_src/rw_small_step/random_walk_office.py (that script executes generation at
import time, so it cannot be imported as a library).
"""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from torch_geometric.data import Data
import torch_geometric.typing as pyg_typing

pyg_typing.WITH_INDEX_SORT = False

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent_src" / "rw_small_step"))
from pcd_slice_methods import knn_connectivity, rotation_a_to_b  # noqa: E402

RENDER_H, RENDER_W = 192, 256
INTR = {"W": 6208, "H": 4135, "fx": 3408.59, "fy": 3408.87}
FOV_X = 2 * np.arctan(INTR["W"] / (2 * INTR["fx"]))
FOV_Y = 2 * np.arctan(INTR["H"] / (2 * INTR["fy"]))
Z_HEIGHT = -0.2
PITCH = -0.1
BRANCH_DEG = (-45.0, 0.0, 45.0)


@dataclass(frozen=True)
class Rect:
    xmin: float
    ymin: float
    xmax: float
    ymax: float


def _in_union(xy, rects):
    x, y = xy
    return any(r.xmin <= x <= r.xmax and r.ymin <= y <= r.ymax for r in rects)


def _sample_xy(rects, rng):
    areas = np.array([max(0.0, r.xmax - r.xmin) * max(0.0, r.ymax - r.ymin) for r in rects])
    r = rects[rng.choice(len(rects), p=areas / areas.sum())]
    return np.array([rng.uniform(r.xmin, r.xmax), rng.uniform(r.ymin, r.ymax)])


# --- per-scene configs, values verbatim from agent_src/rw_small_step scripts ---
SCENES = {
    "office": {
        "rects": [Rect(xmin=-3, xmax=-2, ymin=-2, ymax=0),
                  Rect(xmin=-2, xmax=-0.5, ymin=-2.5, ymax=0),
                  Rect(xmin=0, xmax=1, ymin=-3, ymax=-0.5)],
        "ply": "office/office/scan_raw/combined_aligned.ply", "step": 0.05,
    },
    "break_room": {
        "rects": [Rect(xmin=-1, xmax=0, ymin=-4, ymax=-1.5),
                  Rect(xmin=-2.5, xmax=2.5, ymin=-1.5, ymax=-0.5),
                  Rect(xmin=-2.5, xmax=-0.8, ymin=-0.5, ymax=0.2),
                  Rect(xmin=-1, xmax=3.4, ymin=-0.4, ymax=0.8)],
        "ply": "break_room/kicker/scan_raw/combined_aligned.ply", "step": 0.02,
    },
    "pipes": {
        "rects": [Rect(xmin=-2, xmax=2, ymin=-5, ymax=-3),
                  Rect(xmin=1, xmax=2, ymin=-3, ymax=0)],
        "ply": "pipes/pipes/scan_raw/scan1.ply", "step": 0.2,
    },
    "relief": {
        "rects": [Rect(xmin=5.3, xmax=6.9, ymin=-16.3, ymax=-11.1),
                  Rect(xmin=3.4, xmax=5.3, ymin=-13.3, ymax=-7.7),
                  Rect(xmin=1.8, xmax=3.4, ymin=-9.8, ymax=-4),
                  Rect(xmin=-0.1, xmax=1.8, ymin=-6, ymax=-0.4),
                  Rect(xmin=1.8, xmax=3.9, ymin=-2.2, ymax=0.1),
                  Rect(xmin=3.9, xmax=6.5, ymin=-1.2, ymax=1.7),
                  Rect(xmin=6.5, xmax=9.1, ymin=0, ymax=3),
                  Rect(xmin=9.1, xmax=11.2, ymin=1.2, ymax=4.5),
                  Rect(xmin=11.2, xmax=13.3, ymin=2.5, ymax=5.3),
                  Rect(xmin=13.3, xmax=15.5, ymin=3.8, ymax=6.5),
                  Rect(xmin=15.5, xmax=18, ymin=5, ymax=7.6)],
        "ply": "relief/relief/scan_raw/combined_aligned.ply", "step": 0.02,
    },
}
# hospital excluded: no rw_small_step config exists upstream for it.


# --- helpers copied verbatim from random_walk_office.py ---
def cartesian_to_norm(points, edge_index):
    vectors = points[edge_index[1]] - points[edge_index[0]]
    dist = np.linalg.norm(vectors, axis=1)
    dir_vec = vectors / dist[:, None]
    return np.concatenate([dist[:, None], dir_vec], axis=1).astype(np.float32)


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


def make_graph(pcd_points, pcd_colors, knn_edge_index, knn_edge_attr, visible_idx):
    pcd_points = pcd_points.astype(np.float32, copy=False)
    points_norm = np.linalg.norm(pcd_points, axis=1)
    pcd_points = np.concatenate([points_norm[:, None], pcd_points / points_norm[:, None]], axis=1)

    pcd_colors = pcd_colors.astype(np.float32, copy=False)
    num_knn_nodes = pcd_points.shape[0]
    origin = np.zeros((1, 4), dtype=np.float32)
    origin_rgb = np.zeros((1, pcd_colors.shape[1]), dtype=np.float32)
    all_points = np.vstack([origin, pcd_points])
    all_colors = np.vstack([origin_rgb, pcd_colors])

    targets = np.arange(1, num_knn_nodes + 1, dtype=np.int64)
    origin_sources = np.full(num_knn_nodes, 0, dtype=np.int64)
    origin_to_nodes = np.stack([origin_sources, targets], axis=0)
    nodes_to_origin = np.stack([targets, origin_sources], axis=0)
    origin_edge_index = np.concatenate([origin_to_nodes, nodes_to_origin], axis=1)
    origin_edge_attr = cartesian_to_norm(all_points[:, 1:], origin_edge_index)

    data = Data()
    data.pos = torch.from_numpy(all_points)
    data.rgb = torch.from_numpy(all_colors)
    data.edge_index = torch.from_numpy(knn_edge_index + 1).long()
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
    x = x[base_mask]; y = y[base_mask]; z = z[base_mask]

    tanx = np.tan(0.5 * float(fov_x)); tany = np.tan(0.5 * float(fov_y))
    xn = x / z; yn = y / z
    frustum_mask = (np.abs(xn) <= tanx) & (np.abs(yn) <= tany)

    if not np.any(frustum_mask):
        rgb_img = rgb_img.reshape(H, W, 3)
        if return_depth:
            return rgb_img, depth.reshape(H, W), empty_idx, empty_points, empty_colors
        return rgb_img, empty_idx, empty_points, empty_colors

    src_idx = base_idx[frustum_mask]
    xn = xn[frustum_mask]; yn = yn[frustum_mask]; z = z[frustum_mask]
    cols = pcd_rgb[src_idx]

    sx = (xn / tanx) * 0.5 + 0.5
    sy = 0.5 - (yn / tany) * 0.5
    ui = np.clip(np.floor(sx * W).astype(np.int32), 0, W - 1)
    vi = np.clip(np.floor(sy * H).astype(np.int32), 0, H - 1)
    pix = vi * W + ui

    order = np.lexsort((z, pix))
    pix_s = pix[order]; z_s = z[order]; cols_s = cols[order]; src_idx_s = src_idx[order]

    first = np.empty_like(pix_s, dtype=bool)
    first[0] = True
    first[1:] = pix_s[1:] != pix_s[:-1]

    pix_u = pix_s[first]; z_u = z_s[first]; cols_u = cols_s[first]
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
# --- end copied helpers ---


def view_dir_from_yaw(yaw):
    cp, sp = np.cos(PITCH), np.sin(PITCH)
    d = np.array([cp * np.cos(yaw), cp * np.sin(yaw), sp], dtype=np.float32)
    return d / (np.linalg.norm(d) + 1e-12)


def branch_poses(anchor_xy, theta, rects, step, k):
    """Poses per branch: step 0 = shared anchor pose; steps 1..k deterministic.
    Returns None if ANY branch leaves the navigable union (caller resamples)."""
    branches = []
    for off_deg in BRANCH_DEG:
        th = theta + np.deg2rad(off_deg)
        poses = [(np.array(anchor_xy, dtype=float), theta)]   # shared anchor (view = theta)
        xy = np.array(anchor_xy, dtype=float)
        for _ in range(k):
            xy = xy + step * np.array([np.cos(th), np.sin(th)])
            if not _in_union(xy, rects):
                return None
            poses.append((xy.copy(), th))
        branches.append(poses)
    return branches


def render_and_save(xy, yaw, pcd_points, pcd_colors, out_path):
    vantage = np.array([xy[0], xy[1], Z_HEIGHT], dtype=np.float32)
    viewdir = view_dir_from_yaw(yaw)
    view_rgb, view_depth, visible_idx, pcd_visible_world, pcd_rgb = raycast_img_with_points(
        [0, 40], viewdir, FOV_X, FOV_Y, pcd_points, pcd_colors, vantage,
        RENDER_H, RENDER_W, (0, 0, 0))
    e = np.array([1., 0., 0.])
    R_align = rotation_a_to_b(viewdir, e)
    pcd_visible_aligned = (R_align @ (pcd_visible_world - vantage).T).T
    ei, ew = safe_knn_connectivity(pcd_visible_aligned, 3)
    data_graph = make_graph(pcd_visible_aligned, pcd_rgb, ei, ew, visible_idx)
    packed = {"img": torch.from_numpy(view_rgb / 255.0).to(torch.float32),
              "depth": torch.from_numpy(view_depth).to(torch.float32),
              "graph": data_graph,
              "loc": torch.from_numpy(vantage),
              "view_dir": torch.from_numpy(viewdir).to(torch.float32),
              "ei_points": ei, "ew_points": ew,
              "ei_camera": data_graph.ei_camera, "ea_camera": data_graph.ea_camera}
    torch.save(packed, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=list(SCENES))
    ap.add_argument("--anchors", type=int, default=60)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--ply-root",
                    default="/cluster/scratch/btuncay/cog/gnn_spatial_reasoning/datasets")
    ap.add_argument("--out-root", default="data_branch")
    args = ap.parse_args()

    cfg = SCENES[args.scene]
    out_dir = Path(args.out_root) / args.scene / "branch_walks"
    out_dir.mkdir(parents=True, exist_ok=True)

    pcd = o3d.io.read_point_cloud(str(Path(args.ply_root) / cfg["ply"]))
    pcd_points = np.asarray(pcd.points)
    pcd_colors = np.asarray(pcd.colors)
    print(f"{args.scene}: {len(pcd_points)} points loaded", flush=True)

    for a in range(args.anchors):
        if (out_dir / f"bw_{a}_2_{args.k}.pt").exists():
            continue                                          # idempotent
        rng = np.random.default_rng(1000 + a)                 # per-anchor determinism
        branches = None
        for _ in range(200):
            anchor_xy = _sample_xy(cfg["rects"], rng)
            theta = rng.uniform(0, 2 * np.pi)
            branches = branch_poses(anchor_xy, theta, cfg["rects"], cfg["step"], args.k)
            if branches is not None:
                break
        if branches is None:
            print(f"anchor {a}: no in-bounds branch set after 200 tries, skipped", flush=True)
            continue
        for b, poses in enumerate(branches):
            for s, (xy, yaw) in enumerate(poses):
                render_and_save(xy, yaw, pcd_points, pcd_colors, out_dir / f"bw_{a}_{b}_{s}.pt")
        if a % 10 == 0:
            print(f"anchor {a}/{args.anchors} done", flush=True)
    n = len(list(out_dir.glob("*.pt")))
    print(f"{args.scene}: {n} branch-walk samples in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
