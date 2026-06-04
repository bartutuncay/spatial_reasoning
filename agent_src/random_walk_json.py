import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import open3d as o3d
import torch as torch
import torch_geometric.typing as pyg_typing
from torch_geometric.data import Data

from pcd_slice_methods import knn_connectivity, rotation_a_to_b 

@dataclass(frozen=True)
class Rect:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def area(self) -> float:
        return max(0.0, self.xmax - self.xmin) * max(0.0, self.ymax - self.ymin)


def cartesian_to_norm(points, edge_index):
    vectors = points[edge_index[1]] - points[edge_index[0]]
    dist = np.linalg.norm(vectors, axis=1)
    dir_vec = vectors / np.maximum(dist[:, None], 1e-8)
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
    unit_points = pcd_points / np.maximum(points_norm[:, None], 1e-8)
    pcd_points = np.concatenate([points_norm[:, None], unit_points], axis=1)

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


def raycast_img_with_points(
    trunc,
    view_dir,
    fov_x,
    fov_y,
    pcd_points,
    pcd_rgb,
    vantage,
    height,
    width,
    background=(0, 0, 0),
    return_depth=True,
):
    near, far = float(trunc[0]), float(trunc[1])
    forward = np.asarray(view_dir, dtype=np.float32)
    forward /= np.linalg.norm(forward) + 1e-9
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(world_up, forward))) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-9
    up = np.cross(right, forward)

    rel = (pcd_points - vantage).astype(np.float32, copy=False)
    x = rel @ right
    y = rel @ up
    z = rel @ forward
    base_mask = (z > 1e-6) & (z >= near) & (z <= far)

    rgb_img = np.zeros((height * width, 3), dtype=np.uint8)
    rgb_img[:] = np.array(background, dtype=np.uint8)
    depth = np.full((height * width,), np.inf, dtype=np.float32) if return_depth else None

    empty_points = np.empty((0, 3), dtype=pcd_points.dtype)
    empty_colors = np.empty((0, pcd_rgb.shape[1]), dtype=pcd_rgb.dtype)
    empty_idx = np.empty((0,), dtype=np.int64)

    if not np.any(base_mask):
        rgb_img = rgb_img.reshape(height, width, 3)
        if return_depth:
            return rgb_img, depth.reshape(height, width), empty_idx, empty_points, empty_colors
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
        rgb_img = rgb_img.reshape(height, width, 3)
        if return_depth:
            return rgb_img, depth.reshape(height, width), empty_idx, empty_points, empty_colors
        return rgb_img, empty_idx, empty_points, empty_colors

    src_idx = base_idx[frustum_mask]
    xn = xn[frustum_mask]
    yn = yn[frustum_mask]
    z = z[frustum_mask]
    cols = pcd_rgb[src_idx]

    u = ((xn / tanx) * 0.5 + 0.5) * (width - 1)
    v = (0.5 - (yn / tany) * 0.5) * (height - 1)
    ui = np.clip(u.astype(np.int32), 0, width - 1)
    vi = np.clip(v.astype(np.int32), 0, height - 1)
    pix = vi * width + ui

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
    rgb_img = rgb_img.reshape(height, width, 3)

    if return_depth:
        depth[pix_u] = z_u.astype(np.float32)
        depth = depth.reshape(height, width)

    visible_order = np.argsort(visible_idx)
    visible_idx = visible_idx[visible_order]
    visible_points = pcd_points[visible_idx]
    visible_rgb = pcd_rgb[visible_idx]

    if return_depth:
        return rgb_img, depth, visible_idx, visible_points, visible_rgb
    return rgb_img, visible_idx, visible_points, visible_rgb


def sample_xy_in_union(rects, n, rng):
    areas = np.array([r.area for r in rects], dtype=float)
    if areas.sum() <= 0:
        raise ValueError("Total area is zero; check rectangle bounds.")
    probs = areas / areas.sum()
    choices = rng.choice(len(rects), size=n, p=probs)

    pts = np.zeros((n, 2), dtype=float)
    for i, ridx in enumerate(choices):
        r = rects[ridx]
        pts[i, 0] = rng.uniform(r.xmin, r.xmax)
        pts[i, 1] = rng.uniform(r.ymin, r.ymax)
    return pts


def smooth_heading_walk(
    start_xy,
    rects,
    n_steps=300,
    step=0.1,
    sigma_theta=0.15,
    sigma_step=0.02,
    pitch=-0.1,
    sigma_view=0.07,
    view_pull=0.1,
    rng=None,
):
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

    theta = rng.uniform(0, 2 * np.pi)
    view_yaw = theta
    path_xy = np.zeros((n_steps, 2), float)
    view_dir = np.zeros((n_steps, 3), float)
    view_yaw_hist = np.zeros((n_steps,), float)
    cp, sp = np.cos(pitch), np.sin(pitch)

    for i in range(n_steps):
        theta = theta + rng.normal(scale=sigma_theta)
        step_len = max(0.0, step + rng.normal(scale=sigma_step))
        proposal = xy + step_len * np.array([np.cos(theta), np.sin(theta)])

        if inside(proposal):
            xy = proposal
        else:
            theta = theta + np.pi + rng.normal(scale=0.05)
            proposal = xy + step_len * np.array([np.cos(theta), np.sin(theta)])
            if inside(proposal):
                xy = proposal

        view_yaw = (1.0 - view_pull) * view_yaw + view_pull * theta + rng.normal(scale=sigma_view)

        cy, sy = np.cos(view_yaw), np.sin(view_yaw)
        d = np.array([cp * cy, cp * sy, sp], dtype=float)
        d /= np.linalg.norm(d) + 1e-12
        path_xy[i] = xy
        view_dir[i] = d
        view_yaw_hist[i] = view_yaw

    return path_xy, view_dir, view_yaw_hist


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON dictionary keyed by scene name.")
    return data


def rects_from_config(scene_cfg):
    rects = []
    for rect in scene_cfg["rectangles"]:
        rects.append(
            Rect(
                xmin=float(rect["xmin"]),
                xmax=float(rect["xmax"]),
                ymin=float(rect["ymin"]),
                ymax=float(rect["ymax"]),
            )
        )
    return rects


def run_scene(scene_name, scene_cfg, process_idx, args):
    np.random.seed(process_idx)
    rng = np.random.default_rng(process_idx)

    rects = rects_from_config(scene_cfg)
    sampled_pts = sample_xy_in_union(rects, args.num_samples, rng=rng)
    path, dirs, _ = smooth_heading_walk(
        rng.choice(sampled_pts),
        rects,
        args.num_steps,
        args.step,
        args.sigma_theta,
        args.sigma_step,
        pitch=args.pitch,
        sigma_view=args.sigma_view,
        view_pull=args.view_pull,
        rng=rng,
    )
    print(path, dirs)

    pcd = o3d.io.read_point_cloud(scene_cfg["pcd_path"])
    pcd_points = np.asarray(pcd.points)
    pcd_colors = np.asarray(pcd.colors)
    if len(pcd_points) == 0:
        raise ValueError(f"No points loaded from {scene_cfg['pcd_path']}")

    os.makedirs(scene_cfg["out_dir"], exist_ok=True)

    camera_intrinsics = scene_cfg.get(
        "camera_intrinsics",
        {"W": 6208, "H": 4135, "fx": 3408.59, "fy": 3408.87},
    )
    fov_x = 2 * np.arctan(camera_intrinsics["W"] / (2 * camera_intrinsics["fx"]))
    fov_y = 2 * np.arctan(camera_intrinsics["H"] / (2 * camera_intrinsics["fy"]))
    render_h = int(scene_cfg.get("render_h", args.render_h))
    render_w = int(scene_cfg.get("render_w", args.render_w))
    trunc = scene_cfg.get("trunc", args.trunc)
    z = float(scene_cfg["z"])

    for vantage_idx, point in enumerate(zip(path, dirs)):
        xy, viewdir = point
        out_path = Path(scene_cfg["out_dir"]) / f"rw_{process_idx}_{vantage_idx}.pt"
        vantage = np.array([xy[0], xy[1], z], dtype=np.float32)
        view_rgb, view_depth, visible_idx, pcd_visible_world, pcd_rgb = raycast_img_with_points(
            trunc,
            viewdir,
            fov_x,
            fov_y,
            pcd_points,
            pcd_colors,
            vantage,
            render_h,
            render_w,
            (0, 0, 0),
        )
        r_align = rotation_a_to_b(viewdir, np.array([1.0, 0.0, 0.0]))
        pcd_visible_aligned = (r_align @ (pcd_visible_world - vantage).T).T
        ei, ew = safe_knn_connectivity(pcd_visible_aligned, args.knn_k)
        data_graph = make_graph(pcd_visible_aligned, pcd_rgb, ei, ew, visible_idx)
        packed = {
            "img": torch.from_numpy(view_rgb / 255.0).to(torch.float32),
            "depth": torch.from_numpy(view_depth).to(torch.float32),
            "graph": data_graph,
            "loc": torch.from_numpy(vantage),
            "view_dir": torch.from_numpy(viewdir).to(torch.float32),
            "ei_points": ei,
            "ew_points": ew,
            "ei_camera": data_graph.ei_camera,
            "ea_camera": data_graph.ea_camera,
            "scene": scene_name,
        }
        torch.save(packed, out_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True, help="Seed/task index from Slurm.")
    parser.add_argument("--scene", help="Scene key from the JSON config.")
    parser.add_argument(
        "--config",
        default=Path(__file__).with_name("random_walk_configs.json"),
        type=Path,
        help="JSON dictionary with scene directories and rectangles.",
    )
    parser.add_argument("--list-scenes", action="store_true")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--step", type=float, default=0.2)
    parser.add_argument("--sigma-theta", type=float, default=0.15)
    parser.add_argument("--sigma-step", type=float, default=0.1)
    parser.add_argument("--pitch", type=float, default=-0.1)
    parser.add_argument("--sigma-view", type=float, default=0.07)
    parser.add_argument("--view-pull", type=float, default=0.1)
    parser.add_argument("--render-h", type=int, default=192)
    parser.add_argument("--render-w", type=int, default=256)
    parser.add_argument("--trunc", type=float, nargs=2, default=[0, 40])
    parser.add_argument("--knn-k", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.list_scenes:
        print("\n".join(sorted(config)))
        return
    if not args.scene:
        raise ValueError("Set --scene, or use --list-scenes to print available scene keys.")
    if args.scene not in config:
        available = ", ".join(sorted(config))
        raise KeyError(f"Unknown scene '{args.scene}'. Available scenes: {available}")
    run_scene(args.scene, config[args.scene], args.task_index, args)


if __name__ == "__main__":
    main()
