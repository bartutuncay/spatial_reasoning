import numpy as np
import torch
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis
from torch_geometric.data import Data
import glob
import os
import argparse
from scipy.spatial import KDTree
import cv2

HABITAT_TO_ZUP = np.array([0, 2, 1], dtype=np.int64)
MAX_PCD_POINTS = 12000

def habitat_to_zup(vec):
    return np.asarray(vec, dtype=np.float32)[HABITAT_TO_ZUP]

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

    tree = KDTree(points)
    _, local_conn = tree.query(points, k=k + 1)
    src = np.repeat(np.arange(num_nodes, dtype=np.int64), k)
    neighbors = local_conn[:, 1:].reshape(-1)
    pairs = np.stack([src, neighbors], axis=1)
    pairs = np.sort(pairs, axis=1)
    pairs = np.unique(pairs, axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    weights = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1).astype(np.float32)
    return pairs.T, weights

def make_graph(pcd_points, pcd_colors, knn_edge_index, knn_edge_attr, visible_idx):
    pcd_points = pcd_points.astype(np.float32, copy=False)
    radii = np.linalg.norm(pcd_points, axis=1)
    unit = pcd_points / np.maximum(radii[:, None], 1e-8)
    pcd_pos = np.concatenate([radii[:, None], unit], axis=1).astype(np.float32)

    pcd_colors = pcd_colors.astype(np.float32, copy=False)
    origin = np.zeros((1, 4), dtype=np.float32)
    origin_rgb = np.zeros((1, pcd_colors.shape[1]), dtype=np.float32)
    all_points = np.vstack([origin, pcd_pos])
    all_colors = np.vstack([origin_rgb, pcd_colors])

    num_nodes = pcd_points.shape[0]
    targets = np.arange(1, num_nodes + 1, dtype=np.int64)
    origin_sources = np.zeros(num_nodes, dtype=np.int64)
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

def depth_to_camera_pcd(depth, rgb, hfov_degrees, max_points=MAX_PCD_POINTS):
    depth = np.asarray(depth, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    height, width = depth.shape
    valid = np.isfinite(depth) & (depth > 0.0)
    valid_count = int(valid.sum())
    if valid_count == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    stride = max(1, int(np.ceil(np.sqrt(valid_count / max_points))))
    grid_mask = np.zeros_like(valid, dtype=bool)
    grid_mask[::stride, ::stride] = True
    valid &= grid_mask

    v, u = np.nonzero(valid)
    z = depth[v, u]
    hfov = np.deg2rad(float(hfov_degrees))
    fx = 0.5 * width / np.tan(0.5 * hfov)
    fy = fx
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    x_cam = (u.astype(np.float32) - cx) * z / fx
    y_cam = -(v.astype(np.float32) - cy) * z / fy
    pcd_points = np.stack([z, x_cam, y_cam], axis=1).astype(np.float32)
    pcd_rgb = rgb[v, u, :3].astype(np.float32)
    visible_idx = (v * width + u).astype(np.int64)
    return pcd_points, pcd_rgb, visible_idx

parser = argparse.ArgumentParser()
parser.add_argument('--task-index',type=int)
args = parser.parse_args()

scenes_list = sorted(glob.glob('../../../scratch/btuncay/cog/baselines/datasets/habitat/train/*'))
print(scenes_list)
seed = args.task_index

glb_path = glob.glob(f"{scenes_list[seed]}/*.glb")[0]
navmesh_path = glob.glob(f"{scenes_list[seed]}/*.navmesh")[0]
out_dir = f"{scenes_list[seed]}/random_walks"
png_dir = f"{scenes_list[seed]}/rw_images"
os.makedirs(out_dir,exist_ok=True)
os.makedirs(png_dir,exist_ok=True)

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = glb_path
sim_cfg.enable_physics = False

rgb = habitat_sim.CameraSensorSpec()
rgb.uuid = "color_sensor"
rgb.sensor_type = habitat_sim.SensorType.COLOR
rgb.resolution = [192, 256]
rgb.position = [0.0, 1.5, 0.0]
rgb.hfov = 90.0

depth = habitat_sim.CameraSensorSpec()
depth.uuid = "depth_sensor"
depth.sensor_type = habitat_sim.SensorType.DEPTH
depth.resolution = [192, 256]
depth.position = [0.0, 1.5, 0.0]
depth.hfov = rgb.hfov

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [rgb, depth]

sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
sim.pathfinder.load_nav_mesh(navmesh_path)

agent = sim.initialize_agent(0)
rng = np.random.default_rng(0)

num_walks = 10
steps_per_walk = 40
step_size = 0.1

for walk_id in range(num_walks):
    pos = sim.pathfinder.get_random_navigable_point()
    yaw = rng.uniform(-np.pi, np.pi)

    for step_id in range(steps_per_walk):
        yaw += rng.normal(scale=0.2)

        # Habitat coordinates are Y-up.
        direction = np.array([np.sin(yaw), 0.0, -np.cos(yaw)], dtype=np.float32)
        proposal = pos + step_size * direction

        # Project/slides proposed motion along the navmesh.
        new_pos = sim.pathfinder.try_step(pos, proposal)
        pos = np.array(new_pos)

        state = habitat_sim.AgentState()
        state.position = pos
        state.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
        agent.set_state(state)

        obs = sim.get_sensor_observations()

        img = torch.from_numpy(obs["color_sensor"][..., :3]).float() / 255.0
        d = torch.from_numpy(obs["depth_sensor"]).float()
        pcd_points, pcd_rgb, visible_idx = depth_to_camera_pcd(
            obs["depth_sensor"],
            obs["color_sensor"][..., :3],
            rgb.hfov,
        )
        ei, ew = safe_knn_connectivity(pcd_points, 3)
        data_graph = make_graph(pcd_points, pcd_rgb, ei, ew, visible_idx)

        packed = {
            "img": img,
            "depth": d,
            "graph": data_graph,
            "pcd": torch.cat([data_graph.pos, data_graph.rgb], dim=1),
            "loc": torch.from_numpy(habitat_to_zup(pos)).float(),
            "view_dir": torch.from_numpy(habitat_to_zup(direction)).float(),
            "ei_points": ei,
            "ew_points": ew,
            "ei_camera": data_graph.ei_camera,
            "ea_camera": data_graph.ea_camera,
        }

        stem = f"rw_{walk_id}_{step_id}"
        torch.save(packed, f"{out_dir}/{stem}.pt")
        cv2.imwrite(
            f"{png_dir}/{stem}.png",
            cv2.cvtColor(obs["color_sensor"][..., :3], cv2.COLOR_RGB2BGR),
        )

sim.close()
